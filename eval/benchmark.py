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
             "culture_area_cm2", "dmso_fraction_vv", "n_per_group",
             "plate_format", "wells"]

#: Plate-culture raw inputs and derived fields (WS1 domain).
_CULTURE_RAW_KEYS = [
    "plate_format", "wells", "seeding_density_cells_cm2", "viability_pct",
    "confluent_density_cells_cm2", "doubling_time_h", "culture_duration_h",
]
_CULTURE_DERIVED_KEYS = [
    "seed_per_well", "total_seed_count", "medium_volume_per_well_ml",
    "total_medium_ml", "expected_confluence_pct",
]
#: The minimal raw set a bare model must report for its culture numbers to be
#: cross-checkable (analogue of _CONSISTENCY_KEYS for the plate domain).
_CULTURE_CONSISTENCY_KEYS = ["plate_format", "seeding_density_cells_cm2", "wells"]


@dataclass
class GoldExperiment:
    """A ground-truth experiment with the key numbers a design should recover."""

    id: str
    goal: str
    expected: dict[str, float]  # e.g. {"shear_pa": 0.05, "flow_rate_uLmin": 2.0}
    source: str  # DOI / paper reference — mandatory for inclusion
    # Blind-set only: "cold" = target not in goal text nor the system prompt
    # (pure domain recall); "prompt-backed" = the Labwright system prompt lists a
    # range containing the answer, so the model must still select the right value.
    blind_strength: str | None = None


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
    if plan.derived is not None:
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
    if "flow_rate_uLmin" in gold.expected and plan.flow is not None:
        errs["flow_rate_uLmin"] = relative_error(plan.flow.flow_rate_uLmin, gold.expected["flow_rate_uLmin"])
    if "seed_count" in gold.expected and plan.cells is not None:
        errs["seed_count"] = relative_error(plan.cells.seed_count, gold.expected["seed_count"])
    if "dmso_fraction_vv" in gold.expected and plan.dosing is not None:
        errs["dmso_fraction_vv"] = relative_error(plan.dosing.dmso_fraction_vv, gold.expected["dmso_fraction_vv"])
    if "n_per_group" in gold.expected and plan.stats is not None:
        errs["n_per_group"] = relative_error(float(plan.stats.n_per_group), gold.expected["n_per_group"])
    # Plate-culture domain (WS1): expected keys map onto CulturePlan fields.
    if plan.culture is not None:
        culture_map = {
            "seed_per_well": plan.culture.seed_per_well,
            "total_seed_count": plan.culture.total_seed_count,
            "medium_volume_per_well_ml": plan.culture.medium_volume_per_well_ml,
            "total_medium_ml": plan.culture.total_medium_ml,
            "expected_confluence_pct": plan.culture.expected_confluence_pct,
            "wells": float(plan.culture.wells),
        }
        for key, value in culture_map.items():
            if key in gold.expected and value is not None:
                errs[key] = relative_error(value, gold.expected[key])
    return errs


#: Every derived field the verifier can reject, across all domains. The flow
#: six plus cell seeding, dosing, statistics and the plate-culture set.
#: ``hallucination_rate`` counts errors on whichever of these the plan actually
#: carries.
_DERIVED_FIELDS = [
    "derived.shear_pa", "derived.reynolds", "derived.pressure_drop_pa",
    "derived.residence_time_s", "derived.channel_volume_ul", "derived.mean_velocity_mms",
    "cells.seed_count", "dosing.dmso_fraction_vv", "stats.n_per_group",
    "culture.seed_per_well", "culture.total_seed_count",
    "culture.medium_volume_per_well_ml", "culture.total_medium_ml",
    "culture.expected_confluence_pct",
]


def hallucination_rate(plan: DesignPlan) -> float:
    """Fraction of the plan's derived fields that the verifier rejects (Labwright path).

    A Labwright design's derived numbers come from the calculators, so this is
    0 by construction; the metric is what makes that checkable. Fields the plan
    does not carry (no dosing/stats, no plate culture) are excluded from the
    denominator.
    """
    issues = verify_design(plan)
    if not has_errors(issues):
        return 0.0
    errored = {i.field for i in issues if i.level == "error"}
    present = set(_DERIVED_FIELDS)
    if plan.derived is None or plan.chip is None or plan.flow is None:
        for f in ("derived.shear_pa", "derived.reynolds", "derived.pressure_drop_pa",
                  "derived.residence_time_s", "derived.channel_volume_ul",
                  "derived.mean_velocity_mms"):
            present.discard(f)
    if plan.cells is None:
        present.discard("cells.seed_count")
    if plan.dosing is None:
        present.discard("dosing.dmso_fraction_vv")
    if plan.stats is None:
        present.discard("stats.n_per_group")
    if plan.culture is None:
        for f in ("culture.seed_per_well", "culture.total_seed_count",
                  "culture.medium_volume_per_well_ml", "culture.total_medium_ml",
                  "culture.expected_confluence_pct"):
            present.discard(f)
    return len(errored & present) / max(len(present), 1)


# ---------------------------------------------------------------------------
# Bare-LLM path (lenient number extraction)
# ---------------------------------------------------------------------------

#: Raw inputs a bare model must report for its derived numbers to be checkable.
_CONSISTENCY_KEYS = ["width_um", "height_um", "length_mm", "flow_rate_uLmin", "viscosity_pas", "density_kgm3"]


def _is_culture_gold(gold: GoldExperiment) -> bool:
    """True when the gold's expected keys are plate-culture derived numbers."""
    return bool(set(gold.expected) & set(_CULTURE_DERIVED_KEYS))


def _prompt_keys_for(gold: GoldExperiment) -> list[str]:
    """Key set a bare model must report, chosen per gold domain.

    Flow goals need geometry+flow raws (``_CONSISTENCY_KEYS``); plate-culture
    goals need plate_format + seeding density + wells to make the derived
    culture numbers re-checkable. Everything else is the goal's own targets.
    """
    if _is_culture_gold(gold):
        return sorted(set(gold.expected) | set(_CULTURE_CONSISTENCY_KEYS))
    return sorted(set(gold.expected) | set(_CONSISTENCY_KEYS))


def bare_prompt_for(gold: GoldExperiment) -> str:
    """A tailored prompt asking for exactly the numbers this goal needs.

    Keep the key set minimal so the model's reasoning stays short enough to
    finish within the token budget — a light prompt is fairer than a 17-key one.
    """
    keys = _prompt_keys_for(gold)
    return (
        "You are a wet-lab design expert. For the goal below, compute the design "
        "numbers yourself and return a single flat JSON object with ONLY these keys "
        "(use exactly these names; do the arithmetic; omit nothing):\n"
        + ", ".join(keys)
        + ".\n"
        "Return ONLY the JSON object (no prose, no markdown fences).\n\nGoal: "
        + gold.goal
    )


def soft_gate_prompt_for(gold: GoldExperiment) -> str:
    """The bare prompt plus a 'check yourself' instruction — the soft gate.

    This is the naive thing a user actually tries instead of Labwright: ask the
    LLM to re-derive its own numbers before finalising. No calculators, no
    deterministic verifier. If this works, the whole hard-gate machinery is
    unnecessary; the benchmark exists to show it does not.
    """
    keys = _prompt_keys_for(gold)
    if _is_culture_gold(gold):
        check = (
            "BEFORE you finalize: re-derive every derived culture number "
            f"({', '.join(_CULTURE_DERIVED_KEYS)}) from your own plate_format/"
            "seeding_density_cells_cm2/wells using the standard multi-well plate "
            "dimensions, hemocytometer and viability formulas, and correct any "
            "value that does not match."
        )
    else:
        check = (
            "BEFORE you finalize: re-derive every derived flow number "
            f"({', '.join(_DERIVED_KEYS)}) from your own width_um/height_um/"
            "length_mm/flow_rate_uLmin/viscosity_pas/density_kgm3 using the standard "
            "rectangular-channel formulas, and correct any value that does not match."
        )
    return (
        "You are a wet-lab design expert. For the goal below, compute the design "
        "numbers yourself and return a single flat JSON object with ONLY these keys "
        "(use exactly these names; do the arithmetic; omit nothing):\n"
        + ", ".join(keys)
        + ".\n"
        + check
        + "\n"
        "Return ONLY the JSON object (no prose, no markdown fences).\n\nGoal: "
        + gold.goal
    )


def self_verify_prompt_for(raw: dict[str, float | None], derived_keys: list[str]) -> str:
    """Stage-2 prompt: hand the model its own raw inputs and ask it to recompute.

    The second LLM pass plays the role of verifier. It sees only the raw inputs
    the first pass reported — never the deterministic answers — so any agreement
    is the model's own arithmetic, not a leak.
    """
    present = {k: raw[k] for k in _CONSISTENCY_KEYS if raw.get(k) is not None}
    rendered = ", ".join(f"{k}={present[k]}" for k in _CONSISTENCY_KEYS if k in present)
    return (
        "A design proposed these raw inputs: " + rendered + ".\n"
        "Using the standard rectangular-channel microfluidic formulas, recompute "
        "EXACTLY these derived values yourself (" + ", ".join(derived_keys) + ") "
        "from those inputs, and return a single flat JSON object with ONLY those "
        "keys (use exactly these names; do the arithmetic):\n"
        "Return ONLY the JSON object (no prose, no markdown fences)."
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
    keys = _prompt_keys_for(gold)
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


def run_soft_gate(gold: GoldExperiment, chat: Callable, attempts: int = 3) -> dict[str, float | None]:
    """Ask the raw LLM for the design numbers under a 'check yourself' prompt.

    Identical retry/extraction logic to :func:`run_bare_llm`; only the prompt
    changes. So any measured difference between bare and soft-gate is caused by
    the instruction to self-check, not by parsing or scoring.
    """
    keys = _prompt_keys_for(gold)
    prompt = soft_gate_prompt_for(gold)
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


def run_self_verify(gold: GoldExperiment, chat: Callable, attempts: int = 2) -> dict[str, float | None]:
    """Two-stage 'LLM as its own verifier': propose, then recompute.

    Stage 1 is exactly the bare prompt (propose numbers from memory). Stage 2
    hands the model its own reported raw inputs back and asks it to recompute the
    derived flow numbers *itself*. The final answer is the stage-2 numbers where
    returned; if the verifier pass returns nothing checkable, the proposal stands
    unverified (scored exactly like a bare answer). This is the naive alternative
    to Labwright's deterministic verifier: can a second LLM pass correct the
    first LLM's arithmetic? The benchmark shows it cannot reliably.
    """
    extracted = run_bare_llm(gold, chat, attempts=attempts)
    raw = {k: extracted.get(k) for k in _CONSISTENCY_KEYS}
    if None in raw.values():
        return extracted  # no geometry+flow → nothing for a verifier to check
    prompt = self_verify_prompt_for(raw, _DERIVED_KEYS)
    for _ in range(attempts):
        text = chat(prompt) or ""
        if not text.strip():
            continue
        try:
            data = _extract_json(text)
        except Exception:
            continue
        stage2 = {k: _find_key(data, k) for k in _DERIVED_KEYS}
        if any(v is not None for v in stage2.values()):
            merged = dict(extracted)
            merged.update(stage2)
            return merged
    return extracted  # verifier returned nothing checkable → proposal stands


def bare_recovery(extracted: dict[str, float | None], gold: GoldExperiment) -> dict[str, float]:
    """Relative error of the *reported* numbers vs the gold standard."""
    return {key: relative_error(extracted.get(key), expected) for key, expected in gold.expected.items()}


def bare_checkable(extracted: dict[str, float | None]) -> bool:
    """Whether the bare answer reported enough to cross-check any derived number.

    Flow-verifiable = geometry + flow *and* at least one derived flow metric.
    Culture-verifiable = plate_format + seeding density (+ wells) *and* at
    least one derived culture number. A bare answer that only states a headline
    number (e.g. ``seed_count``) without the raw inputs that produce it cannot
    be cross-checked.
    """
    chip = (extracted.get("width_um"), extracted.get("height_um"), extracted.get("length_mm"))
    flow_rate = extracted.get("flow_rate_uLmin")
    if None not in chip and flow_rate is not None:
        return any(extracted.get(k) is not None for k in _DERIVED_KEYS)
    if extracted.get("plate_format") and extracted.get("seeding_density_cells_cm2") is not None:
        return any(extracted.get(k) is not None for k in _CULTURE_DERIVED_KEYS)
    return False


def _flow_hallucination(extracted: dict[str, float | None]) -> float | None:
    """Cross-check reported flow numbers against the model's own geometry/flow.

    Returns the error fraction, or ``None`` when the answer is not
    flow-verifiable (no geometry+flow, or geometry+flow but no derived flow
    numbers at all — the "nothing checkable" convention maps to 1.0 upstream).
    """
    chip = (extracted.get("width_um"), extracted.get("height_um"), extracted.get("length_mm"))
    flow_rate = extracted.get("flow_rate_uLmin")
    if None in chip or flow_rate is None:
        return None
    viscosity = extracted.get("viscosity_pas") or 1e-3
    density = extracted.get("density_kgm3") or 1000.0
    w, h, L = chip[0], chip[1], chip[2]
    try:
        computed = {
            "shear_pa": mf.wall_shear_stress(flow_rate, w, h, viscosity),
            "reynolds": mf.reynolds_number(flow_rate, w, h, viscosity, density),
            "pressure_drop_pa": mf.pressure_drop(flow_rate, w, h, L, viscosity),
            "residence_time_s": mf.residence_time(flow_rate, w, h, L),
            "channel_volume_ul": mf.channel_volume(w, h, L),
            "mean_velocity_mms": mf.mean_velocity(flow_rate, w, h),
        }
    except ValueError:
        # Degenerate reported inputs (flow 0, or negative / non-finite geometry
        # or flow) cannot be cross-checked — unverifiable.
        return None
    if not all(math.isfinite(v) for v in computed.values()):
        return None
    claimed = {k: extracted.get(k) for k in _DERIVED_KEYS}
    present = [k for k in _DERIVED_KEYS if claimed[k] is not None]
    if not present:
        return None  # no derived flow number reported → nothing to check
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


def _culture_hallucination(extracted: dict[str, float | None]) -> float | None:
    """Cross-check reported plate-culture numbers against the model's own raws.

    Recomputes seed_per_well / total_seed_count / medium_volume_per_well_ml /
    total_medium_ml from the reported plate_format + seeding density (+ wells)
    with the culture calculators. Returns the error fraction, or ``None`` when
    the answer is not culture-verifiable.
    """
    from labwright.calc import culture as calc_culture

    plate = extracted.get("plate_format")
    density = extracted.get("seeding_density_cells_cm2")
    if not plate or density is None:
        return None
    try:
        wells = extracted.get("wells") if extracted.get("wells") is not None else 1
        per_well = calc_culture.cells_per_well(density, plate)
        med = calc_culture.medium_volume_per_well(plate)
        computed = {
            "seed_per_well": per_well,
            "total_seed_count": per_well * wells,
            "medium_volume_per_well_ml": med,
            "total_medium_ml": med * wells,
        }
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in computed.values()):
        return None
    claimed = {k: extracted.get(k) for k in _CULTURE_DERIVED_KEYS}
    present = [k for k in _CULTURE_DERIVED_KEYS if claimed[k] is not None]
    if not present:
        return None  # no derived culture number reported → nothing to check
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


def bare_hallucination(extracted: dict[str, float | None]) -> float:
    """Fraction of reported derived numbers inconsistent with the model's own raw inputs.

    Checks whichever domain the answer is verifiable in (flow, then culture).
    An answer that is not verifiable in either — no geometry+flow, or
    geometry+flow but **no derived flow numbers at all**, or no plate+density
    and no culture numbers — is scored 1.0. The second case matters: a design
    whose every number is typed from memory and cannot be re-derived from the
    model's own inputs is exactly the case Labwright refuses to trust ("numbers
    you type are not trusted"). This mirrors the Labwright convention where a
    run that never submits a plan is scored hallucination 1.0.
    """
    rate = _flow_hallucination(extracted)
    if rate is not None:
        return rate
    rate = _culture_hallucination(extracted)
    if rate is not None:
        return rate
    return 1.0


def run_labwright(goal: str, agent_factory: Callable) -> tuple[DesignPlan | None, str | None]:
    """Run the real Labwright pipeline.

    Returns ``(design, error)``. When the agent produced no design (``plan:
    false``), ``error`` carries the agent's own failure reason so a silent
    refusal is auditable rather than an unexplained blank.
    """
    result = agent_factory().run(goal)
    return result.design, result.error


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced {...} block out of model prose."""
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

#: Which system names the generic evaluate() loop knows how to run. Each returns
#: a flat ``reported`` dict (bare, soft-gate, self-verify) scored with the same
#: lenient bare metrics, or a verified design path (Labwright).
_SYSTEM_RUNNERS: dict[str, Callable] = {
    "bare": lambda g, chat, af: run_bare_llm(g, chat),
    "soft_gate": lambda g, chat, af: run_soft_gate(g, chat),
    "self_verify": lambda g, chat, af: run_self_verify(g, chat),
}


def _score_reported(reported: dict[str, float | None], gold: GoldExperiment) -> dict[str, Any]:
    """Score a flat 'reported numbers' answer with the bare-LLM convention.

    Shared by bare, soft-gate and self-verify so the three competitors are
    judged by identical extraction, tolerance and verifiability rules — only the
    prompt/stage structure differs.
    """
    rec = bare_recovery(reported, gold)
    hall = bare_hallucination(reported)
    return {
        "reported": {k: v for k, v in reported.items() if v is not None},
        "verifiable": bare_checkable(reported),
        "recovery": {k: round(v, 6) for k, v in rec.items()},
        "hallucination_rate": round(hall, 6),
        "valid": hall == 0.0 and all(err <= 0.05 for err in rec.values()),
    }


def _run_system(name: str, gold: GoldExperiment, chat: Callable, agent_factory: Callable) -> dict[str, Any]:
    """Run one named system on one gold entry and return its scored record."""
    if name == "labwright":
        lw, lw_error = run_labwright(gold.goal, agent_factory)
        lw_rec: dict[str, float] = {}
        lw_hall = 1.0
        if lw is not None:
            lw_rec = parameter_recovery(gold, lw)
            lw_hall = hallucination_rate(lw)
        return {
            "plan": lw is not None,
            "error": lw_error,
            "recovery": {k: round(v, 6) for k, v in lw_rec.items()},
            "hallucination_rate": round(lw_hall, 6),
            # usable: a plan that verifies AND recovers every gold target.
            "valid": (
                lw is not None
                and lw_hall == 0.0
                and bool(lw_rec)
                and all(err <= 0.05 for err in lw_rec.values())
            ),
        }
    return _score_reported(_SYSTEM_RUNNERS[name](gold, chat, agent_factory), gold)


def evaluate(
    gold: list[GoldExperiment],
    agent_factory: Callable,
    chat: Callable,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    systems: tuple[str, ...] = ("bare", "labwright"),
) -> dict[str, Any]:
    """Run the requested systems on every gold experiment and aggregate metrics.

    ``systems`` names which systems to run (bare / soft_gate / self_verify /
    labwright, any subset). The default keeps the historical bare-vs-Labwright
    comparison; the competitor baselines are extra systems scored by the same
    rules.
    """
    summary: dict[str, Any] = {"n_gold": len(gold), "per_entry": []}
    for name in systems:
        summary[name] = {"recovery": {}, "hallucination_rate": []}

    for g in gold:
        entry: dict[str, Any] = {"id": g.id}
        for name in systems:
            if progress:
                progress(f"[{g.id}] {name} ...")
            rec = _run_system(name, g, chat, agent_factory)
            entry[name] = rec
            summary[name]["hallucination_rate"].append(rec["hallucination_rate"])
            for key, err in rec["recovery"].items():
                summary[name]["recovery"].setdefault(key, []).append(err)
        summary["per_entry"].append(entry)
        if checkpoint:
            checkpoint(summary)

    for name in systems:
        bucket = summary[name]
        rates = bucket["hallucination_rate"]
        bucket["recovery"] = {k: _mean(v) for k, v in bucket["recovery"].items()}
        bucket["hallucination_rate"] = _mean(rates)
        # self_consistent_rate = fraction of entries with zero verifier errors.
        bucket["self_consistent_rate"] = _mean([1.0 if r == 0.0 else 0.0 for r in rates])
        # usable_rate = self-consistent AND recovers every gold target (±5 %).
        bucket["usable_design_rate"] = _mean(
            [1.0 if e[name]["valid"] else 0.0 for e in summary["per_entry"]]
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
    "run_soft_gate",
    "run_self_verify",
    "bare_prompt_for",
    "soft_gate_prompt_for",
    "self_verify_prompt_for",
    "BARE_CONSISTENCY_TOL",
]
