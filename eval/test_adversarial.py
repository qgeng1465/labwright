"""Tests for the boundary/adversarial evaluation (reviewer demand #3).

Covers: (a) the adversarial gold dataset's shape and honesty — every
``physical_conflict``/``lethal_condition`` entry's ``implied_raws`` must be
genuinely hard-caught by ``submit_design``, and every ``missing_parameter``
entry must be under-determined (no fixed raw inputs, expected elicit); (b) the
``request_info`` elicitation tool registration on ``DesignAgent`` (off by
default, on with ``elicit=True``); (c) the outcome classifier and the
fail-safe / elicitation / exception-catch / fabrication metrics.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import _run_system  # noqa: E402
from labwright.agent.agent import AgentResult, DesignAgent  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADV = os.path.join(_HERE, "gold_adversarial.json")


@pytest.fixture(scope="module")
def adversarial() -> list[dict]:
    with open(_ADV, encoding="utf-8") as fh:
        return json.load(fh)


# -- dataset shape -------------------------------------------------------------

def test_adversarial_dataset_shape(adversarial):
    assert len(adversarial) >= 25, "reviewer wants a substantial boundary set"
    types = {}
    for e in adversarial:
        types[e["type"]] = types.get(e["type"], 0) + 1
        assert e["goal"].strip(), e["id"]
        assert e["expected_outcome"] in ("elicit", "reject"), e["id"]
    assert types.get("missing_parameter", 0) >= 10
    assert types.get("physical_conflict", 0) >= 8
    assert types.get("lethal_condition", 0) >= 6
    # ids are unique
    ids = [e["id"] for e in adversarial]
    assert len(ids) == len(set(ids))


def test_missing_parameter_entries_are_underdetermined(adversarial):
    """Missing-parameter goals must NOT carry fixed raws (they are the point)."""
    for e in adversarial:
        if e["type"] != "missing_parameter":
            continue
        assert e["expected_outcome"] == "elicit", e["id"]
        assert "implied_raws" not in e, (
            f"{e['id']}: a missing-parameter goal cannot have implied raws"
        )
        assert e.get("missing"), e["id"]


def test_every_trap_is_hard_caught(adversarial):
    """Honesty gate: each physical/lethal entry's implied raws must be rejected.

    This is the offline contract the README's exception-catch-rate figure
    depends on: the verifier (or schema) hard-rejects the trap inputs — no LLM
    involved, so the claim is verifiable by inspection, not by trusting the run.
    """
    from labwright.design import submit_design

    checked = 0
    for e in adversarial:
        if e["type"] == "missing_parameter":
            continue
        raws = dict(e["implied_raws"])
        raws.setdefault("goal", e["goal"])
        raws.setdefault("rationale", "adversarial test")
        try:
            res = submit_design(raws)
            n_err = sum(1 for i in res.get("verification", []) if i.get("level") == "error")
            caught = res.get("status") in ("review_required", "validation_error") and (
                n_err > 0 or res.get("status") == "validation_error")
        except Exception:
            caught = True  # schema/calc rejection counts as a hard catch
        assert caught, (
            f"{e['id']}: implied raws NOT rejected by the verifier — "
            f"expected_outcome={e['expected_outcome']} is dishonest"
        )
        checked += 1
    assert checked >= 18


# -- request_info tool ---------------------------------------------------------

class _FakeLLM:
    def chat(self, messages, tools=None):
        raise AssertionError("no LLM call in registration tests")


def test_request_info_absent_by_default():
    agent = DesignAgent(_FakeLLM())
    names = {t["function"]["name"] for t in agent._tools}
    assert "request_info" not in names
    assert "Elicitation rule" not in agent.system_prompt


def test_request_info_registered_when_elicit():
    agent = DesignAgent(_FakeLLM(), elicit=True)
    names = {t["function"]["name"] for t in agent._tools}
    assert "request_info" in names
    assert "Elicitation rule" in agent.system_prompt
    assert "request_info" in agent.system_prompt


def test_request_info_executes_without_computing():
    agent = DesignAgent(_FakeLLM(), elicit=True)
    out = agent._execute_tool("request_info", json.dumps(
        {"parameter": "height_um", "question": "What is the channel height?"}))
    payload = json.loads(out)
    assert payload["status"] == "question_forwarded"
    assert payload["parameter"] == "height_um"
    # The whole point: elicitation never returns a number.
    assert not any(isinstance(v, (int, float)) for v in payload.values())


# -- outcome classifier + metrics -------------------------------------------------

def _stub_agent(outcome, steps=None, plan=None, status="ok", verification=None,
                error=None):
    """Build a fake agent_factory whose AgentResult models one outcome."""

    class _Stub:
        def __init__(self):
            self._verification = verification or []
            self._plan = plan

        def run(self, goal):
            from labwright.schema.design import DesignPlan

            return AgentResult(
                design=DesignPlan(**self._plan) if self._plan else None,
                verification=self._verification,
                status=status,
                error=error,
                steps=steps or [],
            )

    return _Stub


_PLAN_OK = {
    "goal": "adversarial goal", "rationale": "adversarial test",
    "chip": {"width_um": 400, "height_um": 100, "length_mm": 10},
    "flow": {"flow_rate_uLmin": 10.0},
}


@pytest.mark.parametrize("goal_type,steps,plan,status,verification,expected", [
    # elicit: asked for the missing parameter
    ("missing_parameter",
     [{"tool": "request_info", "output": "..."}],
     None, "error", [], "elicit"),
    # reject: verifier hard-rejected the trap
    ("lethal_condition",
     [{"tool": "submit_design", "output": "..."}],
     None, "review_required",
     [type("I", (), {"level": "error", "message": "shear 125 Pa outside band"})()],
     "reject"),
    # refuse: never submitted, no fabrication
    ("missing_parameter", [], None, "error", [], "refuse"),
    # fabricate: accepted a plan built on a guessed missing parameter
    ("missing_parameter",
     [{"tool": "submit_design", "output": "..."}],
     _PLAN_OK,
     "ok", [], "fabricate"),
    # reject (self-corrected): a physical/lethal goal whose plan PASSED means the
    # agent did not ship the trap — fail-safe.
    ("lethal_condition",
     [{"tool": "submit_design", "output": "..."}],
     _PLAN_OK,
     "ok", [], "reject"),
])
def test_labwright_outcome_classification(goal_type, steps, plan, status,
                                          verification, expected):
    """The classifier reads the real tool trace + verifier verdict."""
    from eval.run_adversarial import _adv_labwright

    factory = _stub_agent(outcome=None, steps=steps, plan=plan, status=status,
                          verification=verification)
    rec = _adv_labwright("goal", goal_type, factory)
    assert rec["outcome"] == expected, rec


def test_baseline_classifiers():
    """bare / code-interpreter refuse-vs-fabricate-vs-silence, via _refused."""
    from eval.run_adversarial import _adv_bare, _adv_code, _REFUSED_KEY

    assert _REFUSED_KEY == "_refused"
    # bare: refused JSON
    chat_refuse = lambda p: '{"_refused": true, "reason": "height not given"}'
    assert _adv_bare("g", chat_refuse)["outcome"] == "refuse"
    # bare: fabricated numbers
    chat_fab = lambda p: '{"height_um": 100, "shear_pa": 0.05}'
    assert _adv_bare("g", chat_fab)["outcome"] == "fabricate"
    # bare: prose refusal, no JSON
    chat_prose = lambda p: "Cannot determine — the channel height is missing."
    assert _adv_bare("g", chat_prose)["outcome"] == "refuse"
    # bare: empty
    assert _adv_bare("g", lambda p: "")["outcome"] == "no_answer"


def test_fail_safe_metrics():
    from eval.run_adversarial import aggregate

    records = [
        {"outcome": "elicit"}, {"outcome": "reject"}, {"outcome": "refuse"},
        {"outcome": "fabricate"}, {"outcome": "code_error"}, {"outcome": "no_answer"},
        {"outcome": "fabricate"}, {"outcome": "reject"},
    ]
    agg = aggregate(records, 8)
    assert agg["fail_safe_rate"] == 0.5  # 4/8
    assert agg["elicitation_rate"] == pytest.approx(1 / 8)
    assert agg["exception_catch_rate"] == pytest.approx(2 / 8)
    assert agg["fabrication_rate"] == pytest.approx(2 / 8)


def test_labwright_record_has_elicited_key():
    """The benchmark's labwright branch records elicitation (default 0)."""
    from benchmark import GoldExperiment

    gold = GoldExperiment(id="x", goal="goal", expected={}, source="adversarial-test")
    rec = _run_system(
        "labwright", gold,
        chat=lambda p: "{}",
        agent_factory=_stub_agent(outcome=None, steps=[], plan=None, status="error"),
    )
    assert rec["elicited"] == 0
    assert rec["tool_calls"] == 0
