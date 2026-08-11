"""Benchmark harness for the Labwright paper.

See ``eval/README.md`` for the experiment plan and the provenance rules on
gold-standard entries (no fabricated literature numbers — every entry must cite
a source a reviewer can open).

The two systems are compared fairly but asymmetrically *in favour of the bare
LLM*:

* **bare-LLM** is asked for the design numbers once (no calculators), and its
  *reported* numbers are extracted leniently — any JSON nesting is searched for
  the key names. A reported derived number counts as *consistent* with its own
  geometry/flow up to a generous ±5 %, and a design with no geometry/flow to
  back its numbers is judged untrustworthy.
* **Labwright** runs the normal verified pipeline; its derived numbers come from
  the calculators, so they must agree to the verifier's machine-precision
  tolerance (1e-6).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

from labwright.calc import microfluidics as mf
from labwright.schema.design import DesignPlan
from labwright.verify.checker import has_errors, verify_design

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(_HERE, "gold_experiments.json")

#: Tolerance for a bare-LLM reported number to count as "consistent" with its
#: own geometry/flow. Generous on purpose — the bare model does the arithmetic
#: by hand; Labwright uses calculators and must match to 1e-6.
BARE_CONSISTENCY_TOL = 0.05

_DERIVED_KEYS = ["shear_pa", "reynolds", "pressure_drop_pa", "residence_time_s",
                 "channel_volume_ul", "mean_velocity_mms"]
_RAW_KEYS = ["width_um", "height_um", "length_mm", "flow_rate_uLmin", "viscosity_pas",
             "density_kgm3", "seed_count", "seeding_density_cells_cm2",
             "culture_area_cm2", "dmso_fraction_vv", "n_per_group"]


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
# Metrics (DesignPlan path — Labwright)
# ---------------------------------------------------------------------------


def relative_error(got: float | None, expected: float) -> float:
    """|got - expected| / expected; inf if got is None."""
    if got is None:
        return float("inf")
    return abs(got - expected) / abs(expected)


def parameter_recovery(gold: GoldExperiment, plan: DesignPlan) -> dict[str, float]:
    """Relative error of the design vs the gold standard on key parameters."""
    errs: dict[str, float] = {}
    d = plan.derived
    derived_map = {
        "shear_pa": d.shear_pa,
        "reynolds": d.reynolds,
        "pressure_drop_pa": d.pressure_drop_pa,
        "residence_time_s": d.residence_time_s,
        "channel_volume_ul": d.channel_volume_ul,
        "mean_velocity_mms": d.mean_velocity_mms,
    }
    for key, value in derived_map.items():
        if key in gold.expected:
            errs[key] = relative_error(value, gold.expected[key])
    if "flow_rate_uLmin" in gold.expected:
        errs["flow_rate_uLmin"] = relative_error(plan.flow.flow_rate_uLmin, gold.expected["flow_rate_uLmin"])
    if "seed_count" in gold.expected:
        errs["seed_count"] = relative_error(plan.cells.seed_count, gold.expected["seed_count"])
    if "dmso_fraction_vv" in gold.expected and plan.dosing is not None:
        errs["dmso_fraction_vv"] = relative_error(plan.dosing.dmso_fraction_vv, gold.expected["dmso_fraction_vv"])
    if "n_per_group" in gold.expected and plan.stats is not None:
        errs["n_per_group"] = relative_error(float(plan.stats.n_per_group), gold.expected["n_per_group"])
    return errs


def hallucination_rate(plan: DesignPlan) -> float:
    """Fraction of derived fields that the verifier rejects (Labwright path).

    A Labwright design's derived numbers come from the calculators, so this is
    0 by construction; the metric is what makes that checkable.
    """
    issues = verify_design(plan)
    if not has_errors(issues):
        return 0.0
    errored = {i.field.split(".")[-1] for i in issues if i.level == "error"}
    return len(errored & set(_DERIVED_KEYS)) / len(_DERIVED_KEYS)


# ---------------------------------------------------------------------------
# Bare-LLM path (lenient number extraction)
# ---------------------------------------------------------------------------

#: Raw inputs a bare model must report for its derived numbers to be checkable.
_CONSISTENCY_KEYS = ["width_um", "height_um", "length_mm", "flow_rate_uLmin", "viscosity_pas", "density_kgm3"]


def bare_prompt_for(gold: GoldExperiment) -> str:
    """A tailored prompt asking for exactly the numbers this goal needs.

    Keep the key set minimal so the model's reasoning stays short enough to
    finish within the token budget — a light prompt is fairer than a 17-key one.
    """
    keys = sorted(set(gold.expected) | set(_CONSISTENCY_KEYS))
    return (
        "You are a wet-lab design expert. For the goal below, compute the design "
        "numbers yourself and return a single flat JSON object with ONLY these keys "
        "(use exactly these names; do the arithmetic; omit nothing):\n"
        + ", ".join(keys)
        + ".\n"
        "Return ONLY the JSON object (no prose, no markdown fences).\n\nGoal: "
    )


def _find_key(data: Any, key: str, depth: int = 0) -> float | None:
    """Recursively find a numeric field by exact name anywhere in a JSON tree."""
    if depth > 12:
        return None
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                try:
                    f = float(v)
                    if math.isfinite(f):
                        return f
                except (TypeError, ValueError):
                    pass
            found = _find_key(v, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_key(item, key, depth + 1)
            if found is not None:
                return found
    return None


def run_bare_llm(gold: GoldExperiment, chat: Callable, attempts: int = 3) -> dict[str, float | None]:
    """Ask a raw LLM for the design numbers; return a lenient extraction.

    Retries on empty/unparseable responses (the models intermittently spend the
    whole token budget on hidden reasoning and emit nothing — a transient
    budget artifact, not a competence verdict). Returns a dict mapping every
    key to a float, or ``None`` when the model never reported it.
    """
    keys = sorted(set(gold.expected) | set(_CONSISTENCY_KEYS))
    prompt = bare_prompt_for(gold)
    empty = {k: None for k in keys}
    for _ in range(attempts):
        text = chat(prompt) or ""
        if not text.strip():
            continue
        try:
            data = _extract_json(text)
        except Exception:
            continue
        extracted = {k: _find_key(data, k) for k in keys}
        if any(v is not None for v in extracted.values()):
            return extracted
    return empty


def bare_recovery(extracted: dict[str, float | None], gold: GoldExperiment) -> dict[str, float]:
    """Relative error of the *reported* numbers vs the gold standard."""
    return {key: relative_error(extracted.get(key), expected) for key, expected in gold.expected.items()}


def bare_hallucination(extracted: dict[str, float | None]) -> float:
    """Fraction of reported derived numbers inconsistent with the model's own
    geometry/flow. No geometry+flow → 1.0 (numbers cannot be trusted)."""
    chip = (extracted.get("width_um"), extracted.get("height_um"), extracted.get("length_mm"))
    flow_rate = extracted.get("flow_rate_uLmin")
    if None in chip or flow_rate is None:
        return 1.0
    viscosity = extracted.get("viscosity_pas") or 1e-3
    density = extracted.get("density_kgm3") or 1000.0
    w, h, L = chip[0], chip[1], chip[2]
    computed = {
        "shear_pa": mf.wall_shear_stress(flow_rate, w, h, viscosity),
        "reynolds": mf.reynolds_number(flow_rate, w, h, viscosity, density),
        "pressure_drop_pa": mf.pressure_drop(flow_rate, w, h, L, viscosity),
        "residence_time_s": mf.residence_time(flow_rate, w, h, L),
        "channel_volume_ul": mf.channel_volume(w, h, L),
        "mean_velocity_mms": mf.mean_velocity(flow_rate, w, h),
    }
    claimed = {k: extracted.get(k) for k in _DERIVED_KEYS}
    present = [k for k in _DERIVED_KEYS if claimed[k] is not None]
    if not present:
        return 0.0  # nothing claimed → nothing to contradict (recovery catches silence)
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


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
        "per_entry": [],
    }

    for g in gold:
        if progress:
            progress(f"[{g.id}] bare-LLM ...")
        extracted = run_bare_llm(g, chat)
        if progress:
            progress(f"[{g.id}] Labwright ...")
        lw = run_labwright(g.goal, agent_factory)

        bare_rec = bare_recovery(extracted, g)
        bare_hall = bare_hallucination(extracted)
        for key, err in bare_rec.items():
            summary["bare"]["recovery"].setdefault(key, []).append(err)
        summary["bare"]["hallucination_rate"].append(bare_hall)

        lw_rec: dict[str, float] = {}
        lw_hall = 1.0
        if lw is not None:
            lw_rec = parameter_recovery(g, lw)
            lw_hall = hallucination_rate(lw)
        for key, err in lw_rec.items():
            summary["labwright"]["recovery"].setdefault(key, []).append(err)
        summary["labwright"]["hallucination_rate"].append(lw_hall)

        summary["per_entry"].append(
            {
                "id": g.id,
                "bare": {
                    "reported": {k: v for k, v in extracted.items() if v is not None},
                    "recovery": {k: round(v, 6) for k, v in bare_rec.items()},
                    "hallucination_rate": round(bare_hall, 6),
                    "valid": bare_hall == 0.0 and all(err <= 0.05 for err in bare_rec.values()),
                },
                "labwright": {
                    "plan": lw is not None,
                    "recovery": {k: round(v, 6) for k, v in lw_rec.items()},
                    "hallucination_rate": round(lw_hall, 6),
                    # usable: a plan that verifies AND recovers every gold target.
                    "valid": (
                        lw is not None
                        and lw_hall == 0.0
                        and bool(lw_rec)
                        and all(err <= 0.05 for err in lw_rec.values())
                    ),
                },
            }
        )

    for bucket in (summary["bare"], summary["labwright"]):
        rates = bucket["hallucination_rate"]
        bucket["recovery"] = {k: _mean(v) for k, v in bucket["recovery"].items()}
        bucket["hallucination_rate"] = _mean(rates)
        # self_consistent_rate = fraction of entries with zero verifier errors.
        bucket["self_consistent_rate"] = _mean([1.0 if r == 0.0 else 0.0 for r in rates])
        # usable_rate = self-consistent AND recovers every gold target (±5 %).
        bucket["usable_design_rate"] = _mean(
            [1.0 if e["valid"] else 0.0 for e in summary["per_entry"]]
        )
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


__all__ = [
    "GoldExperiment",
    "load_gold",
    "evaluate",
    "parameter_recovery",
    "hallucination_rate",
    "bare_recovery",
    "bare_hallucination",
    "run_bare_llm",
    "bare_prompt_for",
    "BARE_CONSISTENCY_TOL",
]
