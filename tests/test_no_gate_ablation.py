"""The ``tool_no_gate`` ablation — proof that the verifier is not circular.

The "circular verification" criticism runs: the calculators derive the numbers
and the verifier recomputes them from the *same* calculators, so the 0.000
hallucination rate is an artifact of the loop. The no-gate ablation answers it
empirically: run the exact same model, calculators and ReAct loop with the
verification layer switched off (``submit_design(verify=False)`` accepts
unconditionally, and the agent's system prompt carries no verification
discipline), then judge the resulting plans post-hoc with the *identical*
``_score_design`` rules. The verifier's independent value is the gap between
the two.

These tests pin the mechanics of that gap:

1. ``submit_design(verify=False)`` accepts a physically absurd design that the
   verifier would reject — status ``ok``, no verification issues — while the
   same input with the gate on is ``review_required``.
2. A no-gate agent finalizes an unverified, unphysical design as ``ok``; the
   gate-on agent reports it as ``review_required``.
3. Post-hoc, the benchmark's ``tool_no_gate`` record carries the verifier's
   *would-have-rejected* verdict (hallucination > 0, not usable), so the
   ablation quantifies what the verifier catches.
"""

import json
from types import SimpleNamespace

import pytest

from labwright.agent.agent import DesignAgent, NO_VERIFY_SYSTEM_PROMPT
from labwright.design import submit_design
from labwright.verify.checker import has_errors, verify_design

# Same raw-only design as test_gate_security, but with an absurdly shallow
# channel (1 µm tall). Every number is a legal positive input — the schema
# accepts it and the calculators happily derive 500 Pa wall shear — so only the
# verifier's physiological sanity band rejects it. That is exactly the class of
# error the verifier catches that the calculators alone cannot: calculators
# don't know physiology, the verifier does.
_RAW_GOOD = {
    "goal": "Perfused liver-chip model of drug-induced injury",
    "rationale": "Sinusoidal shear target 0.05 Pa; HepG2 at 1e5/cm2",
    "chip": {"width_um": 400, "height_um": 100, "length_mm": 20,
             "channel_count": 1, "material": "PDMS"},
    "flow": {"flow_rate_uLmin": 2, "viscosity_pas": 0.001, "density_kgm3": 1000},
    "cells": {"cell_type": "HepG2", "seeding_density_cells_cm2": 100000,
              "culture_area_cm2": 0.08, "doubling_time_h": 35,
              "culture_duration_h": 72},
    "dosing": {"compound": "Acetaminophen", "molecular_weight_g_mol": 151.16,
               "stock_mM": 100, "working_mM": 0.1, "vehicle_control": True,
               "exposure_h": 24},
    "stats": {"effect_size": 1.0, "std_dev": 1.0, "alpha": 0.05, "power": 0.80},
    "caveats": ["confirm shear from literature"],
}
_RAW_BAD = {**_RAW_GOOD, "chip": {**_RAW_GOOD["chip"], "height_um": 1}}


def _bad_plan() -> tuple[dict, bool]:
    """Return the plan built from the unphysical raw, plus whether it verifies."""
    from labwright.schema.design import DesignPlan

    plan = DesignPlan(**submit_design(_RAW_BAD, verify=False)["design"])
    return plan, not has_errors(verify_design(plan))


def test_no_gate_accepts_what_the_verifier_rejects():
    """Same unphysical raw: gate on → review_required; gate off → ok."""
    gated = submit_design(_RAW_BAD, verify=True)
    assert gated["status"] == "review_required"
    assert gated["verification"], "verifier must flag the 500 Pa shear"

    open = submit_design(_RAW_BAD, verify=False)
    assert open["status"] == "ok"
    assert open["verification"] == []
    assert open["verification_summary"] == ""


def test_no_gate_agent_finalizes_unphysical_design_as_ok():
    """A no-gate agent stops on its first (unverified) submission with status ok."""
    plan, verifies = _bad_plan()
    assert not verifies, "sanity band must reject the 500 Pa shear"

    def call(name, arguments):
        return SimpleNamespace(id=name,
                               function=SimpleNamespace(name=name, arguments=arguments))

    class DirtyLLM:
        def chat(self, messages, tools=None, **kwargs):
            return SimpleNamespace(
                content=None,
                tool_calls=[call("submit_design", json.dumps(_RAW_BAD))],
            )

    agent = DesignAgent(llm=DirtyLLM(), max_iterations=5, verify_gate=False)
    result = agent.run("design a perfused liver-chip")
    assert result.status == "ok"
    assert result.verification == []
    assert result.design is not None


def test_gate_on_agent_reports_same_submission_as_review_required():
    """The control: the identical submission under the gate is not silently ok."""

    def call(name, arguments):
        return SimpleNamespace(id=name,
                               function=SimpleNamespace(name=name, arguments=arguments))

    class DirtyLLM:
        def chat(self, messages, tools=None, **kwargs):
            return SimpleNamespace(
                content=None,
                tool_calls=[call("submit_design", json.dumps(_RAW_BAD))],
            )

    agent = DesignAgent(llm=DirtyLLM(), max_iterations=5, verify_gate=True)
    result = agent.run("design a perfused liver-chip")
    assert result.status == "review_required"
    assert result.verification, "the gate surfaces the sanity violation"


def test_no_gate_record_scored_posthoc_with_verifier():
    """The benchmark judges the no-gate plan with the identical rules — a plan
    the verifier would have rejected reads as hallucination > 0 / not usable."""
    from eval.benchmark import _run_system, load_gold

    gold = load_gold("eval/gold_experiments.json")[:1]

    def call(name, arguments):
        return SimpleNamespace(id=name,
                               function=SimpleNamespace(name=name, arguments=arguments))

    class DirtyLLM:
        def chat(self, messages, tools=None, **kwargs):
            return SimpleNamespace(
                content=None,
                tool_calls=[call("submit_design", json.dumps(_RAW_BAD))],
            )

    def factory_nogate():
        return DesignAgent(llm=DirtyLLM(), max_iterations=5, verify_gate=False)

    rec = _run_system("tool_no_gate", gold[0], chat=lambda p: "", agent_factory=lambda: None,
                      agent_factory_nogate=factory_nogate)
    # The plan was accepted by the no-gate submit, but the post-hoc verifier
    # rejects it: the gate's independent value is visible in the record.
    assert rec["hallucination_rate"] > 0.0
    assert not rec["valid"]


def test_no_gate_prompt_keeps_tools_but_drops_verification_discipline():
    """The ablation prompt keeps the calculators and physiology anchors but
    removes every promise the verifier backs: no 'never invent a number', no
    review_required fix loop, no 'derived fields computed and verified'."""
    assert "calculator" in NO_VERIFY_SYSTEM_PROMPT
    assert "Hepatic sinusoidal shear" in NO_VERIFY_SYSTEM_PROMPT
    for forbidden in ("never invent", "review_required", "numbers you type are not trusted",
                      "derived fields are computed"):
        assert forbidden not in NO_VERIFY_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# labwright_iter: the review_required fix-and-resubmit loop.
# ---------------------------------------------------------------------------


def _mk_llm(script: list[dict]):
    """A mock LLM that replays a script of tool calls, one per chat() call."""
    calls = []

    def call(name, arguments):
        return SimpleNamespace(id=f"c{len(calls)}",
                               function=SimpleNamespace(name=name, arguments=arguments))

    class ScriptLLM:
        def chat(self, messages, tools=None, **kwargs):
            calls.append(messages)
            return SimpleNamespace(content=None, tool_calls=[call(**script[len(calls) - 1])])

    return ScriptLLM(), calls


def test_iter_agent_fixes_review_required_and_resubmits():
    """A review_required verdict feeds back; the agent fixes and resubmits ok."""
    script = [
        {"name": "submit_design", "arguments": json.dumps(_RAW_BAD)},  # 500 Pa → review_required
        {"name": "submit_design", "arguments": json.dumps(_RAW_GOOD)},  # fixed
    ]
    llm, messages = _mk_llm(script)
    agent = DesignAgent(llm=llm, max_iterations=6, verify_gate=True,
                        max_submission_attempts=3)
    result = agent.run("design a perfused liver-chip")
    assert result.status == "ok"
    assert result.design is not None
    # The nudge was injected between the two submissions.
    assert any(
        isinstance(m, dict) and "review_required" in (m.get("content") or "")
        for hist in messages[1:]
        for m in hist
    )
    fix_rounds = sum(1 for s in result.steps
                     if isinstance(s, dict) and s.get("type") == "review_required")
    assert fix_rounds == 1


def test_iter_agent_exhausts_attempts_keeps_honest_verdict():
    """A model that never fixes still ends with the honest review_required."""
    script = [{"name": "submit_design", "arguments": json.dumps(_RAW_BAD)}] * 5
    llm, _ = _mk_llm(script)
    agent = DesignAgent(llm=llm, max_iterations=6, verify_gate=True,
                        max_submission_attempts=3)
    result = agent.run("design a perfused liver-chip")
    assert result.status == "review_required"
    assert result.verification, "the dirty plan's real findings must be carried"
    fix_rounds = sum(1 for s in result.steps
                     if isinstance(s, dict) and s.get("type") == "review_required")
    assert fix_rounds == 2  # attempts 1 and 2 feed back; attempt 3 is terminal


def test_first_submit_default_unchanged():
    """Default max_submission_attempts=1 keeps the historical first-submit loop."""
    script = [{"name": "submit_design", "arguments": json.dumps(_RAW_BAD)}] * 3
    llm, _ = _mk_llm(script)
    agent = DesignAgent(llm=llm, max_iterations=6, verify_gate=True)
    result = agent.run("design a perfused liver-chip")
    assert result.status == "review_required"
    fix_rounds = sum(1 for s in result.steps
                     if isinstance(s, dict) and s.get("type") == "review_required")
    assert fix_rounds == 0  # first review_required is terminal, as before
