"""Benchmark harness for the Labwright paper.

See ``eval/README.md`` for the experiment plan and for the provenance rules on
gold-standard entries (no fabricated literature numbers — every entry must cite
a source a reviewer can open).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from labwright.calc import microfluidics as mf
from labwright.schema.design import DesignPlan
from labwright.verify.checker import has_errors, verify_design

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(_HERE, "gold_experiments.json")


@dataclass
class GoldExperiment:
    """A ground-truth experiment with the key numbers a design should recover."""

    id: str
    goal: str
    expected: dict[str, float]  # e.g. {"shear_pa": 0.05, "flow_rate_uLmin": 2.0}
    source: str  # DOI / paper reference — mandatory for inclusion


def load_gold(path: str = GOLD_PATH) -> list[GoldExperiment]:
    with open(path) as fh:
        return [GoldExperiment(**item) for item in json.load(fh)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def relative_error(got: float | None, expected: float) -> float:
    """|got - expected| / expected; inf if got is None."""
    if got is None:
        return float("inf")
    return abs(got - expected) / abs(expected)


def parameter_recovery(gold: GoldExperiment, plan: DesignPlan) -> dict[str, float]:
    """Relative error of the design vs the gold standard on key parameters."""
    errs: dict[str, float] = {}
    if "shear_pa" in gold.expected:
        errs["shear_pa"] = relative_error(plan.derived.shear_pa, gold.expected["shear_pa"])
    if "flow_rate_uLmin" in gold.expected:
        errs["flow_rate_uLmin"] = relative_error(plan.flow.flow_rate_uLmin, gold.expected["flow_rate_uLmin"])
    if "seed_count" in gold.expected:
        errs["seed_count"] = relative_error(plan.cells.seed_count, gold.expected["seed_count"])
    if "n_per_group" in gold.expected and plan.stats is not None:
        errs["n_per_group"] = relative_error(float(plan.stats.n_per_group), gold.expected["n_per_group"])
    return errs


def hallucination_rate(plan: DesignPlan) -> float:
    """Fraction of derived fields that the verifier rejects.

    A bare-LLM design that invents shear stress will be caught here; a
    Labwright design (derived numbers from calculators) cannot fail on errors,
    by construction.
    """
    issues = verify_design(plan)
    if not has_errors(issues):
        return 0.0
    # count distinct derived fields with errors
    errored = {i.field.split(".")[-1] for i in issues if i.level == "error"}
    derived_fields = {"shear_pa", "reynolds", "pressure_drop_pa", "residence_time_s",
                      "channel_volume_ul", "mean_velocity_mms"}
    return len(errored & derived_fields) / len(derived_fields)


# ---------------------------------------------------------------------------
# Runners (injected so tests can stub the LLM)
# ---------------------------------------------------------------------------


def run_bare_llm(goal: str, chat: Callable) -> DesignPlan | None:
    """Ask a raw LLM for a full design JSON (no calculators). Returns a plan or None."""
    import json as _json

    prompt = (
        "You are a wet-lab design expert. Given this goal, return a JSON object that is a "
        "complete DesignPlan: chip geometry (width_um,height_um,length_mm), flow "
        "(flow_rate_uLmin, viscosity_pas), cells (cell_type, seeding_density_cells_cm2, "
        "culture_area_cm2, seed_count), and derived flow metrics (shear_pa, reynolds, "
        "pressure_drop_pa, residence_time_s, channel_volume_ul, mean_velocity_mms), plus "
        "stats (effect_size, std_dev, alpha, power, n_per_group) if applicable. "
        "Compute the numbers yourself.\n\nGoal: " + goal
    )
    try:
        text = chat(prompt)
        data = _extract_json(text)
        return DesignPlan(**data)
    except Exception:
        return None


def run_labwright(goal: str, agent_factory: Callable) -> DesignPlan | None:
    """Run the real Labwright pipeline."""
    result = agent_factory().run(goal)
    return result.design


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced {...} block out of model prose."""
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def evaluate(
    gold: list[GoldExperiment],
    agent_factory: Callable,
    chat: Callable,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run both systems on every gold experiment and aggregate the metrics."""
    summary: dict[str, Any] = {
        "n_gold": len(gold),
        "bare": {"recovery": {}, "hallucination_rate": []},
        "labwright": {"recovery": {}, "hallucination_rate": []},
    }

    for g in gold:
        if progress:
            progress(f"[{g.id}] bare-LLM ...")
        bare = run_bare_llm(g.goal, chat)
        if progress:
            progress(f"[{g.id}] Labwright ...")
        lw = run_labwright(g.goal, agent_factory)

        for system, plan, bucket in (("bare", bare, summary["bare"]), ("labwright", lw, summary["labwright"])):
            if plan is None:
                bucket["hallucination_rate"].append(1.0)  # produced nothing -> fully unusable
                for key in g.expected:
                    bucket["recovery"].setdefault(key, []).append(float("inf"))
                continue
            for key, err in parameter_recovery(g, plan).items():
                bucket["recovery"].setdefault(key, []).append(err)
            bucket["hallucination_rate"].append(hallucination_rate(plan))

    for bucket in (summary["bare"], summary["labwright"]):
        bucket["recovery"] = {k: _mean(v) for k, v in bucket["recovery"].items()}
        bucket["hallucination_rate"] = _mean(bucket["hallucination_rate"])
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


__all__ = ["GoldExperiment", "load_gold", "evaluate", "parameter_recovery", "hallucination_rate"]
