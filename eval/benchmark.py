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

#: Per-domain raw/derived/consistency key sets. The single source of truth is
#: :mod:`labwright.blocks` — one ``Block`` per design domain declares its raw,
#: derived and consistency keys — and these aliases are re-exported here as
#: lists so the existing ``_CULTURE_*`` / ``_SPHEROID_*`` / flow key names keep
#: working. Adding a domain to the benchmark means adding a ``Block`` entry;
#: these names then exist automatically.
from labwright.blocks import ALL_FIELD_MAP, ALL_RAW_KEYS, BLOCKS

_DERIVED_KEYS = list(BLOCKS["flow"].derived_keys)
_RAW_KEYS = sorted(ALL_RAW_KEYS)
_CONSISTENCY_KEYS = list(BLOCKS["flow"].consistency_keys)

#: Plate-culture raw inputs and derived fields (WS1 domain).
_CULTURE_RAW_KEYS = list(BLOCKS["culture"].raw_keys)
_CULTURE_DERIVED_KEYS = list(BLOCKS["culture"].derived_keys)
#: The minimal raw set a bare model must report for its culture numbers to be
#: cross-checkable (analogue of _CONSISTENCY_KEYS for the plate domain).
_CULTURE_CONSISTENCY_KEYS = list(BLOCKS["culture"].consistency_keys)

#: Spheroid / 3D-culture raw inputs and derived fields (3D domain).
_SPHEROID_RAW_KEYS = list(BLOCKS["spheroid"].raw_keys)
_SPHEROID_DERIVED_KEYS = list(BLOCKS["spheroid"].derived_keys)
#: The minimal raw set a bare model must report for its spheroid numbers to be
#: cross-checkable.
_SPHEROID_CONSISTENCY_KEYS = list(BLOCKS["spheroid"].consistency_keys)

#: Perfused-system pharmacokinetics raw inputs and derived fields (PK domain).
_PK_RAW_KEYS = list(BLOCKS["pk"].raw_keys)
_PK_DERIVED_KEYS = list(BLOCKS["pk"].derived_keys)
#: The minimal raw set a bare model must report for its PK numbers to be
#: cross-checkable: inlet/outlet concentrations and the flow are enough to
#: re-derive extraction ratio and clearance.
_PK_CONSISTENCY_KEYS = list(BLOCKS["pk"].consistency_keys)


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
    #: Real-world failure mode the goal exercises: "complete-info" (all numbers
    #: given, answer follows by calculation), "partial-info" (a parameter must
    #: come from a standard reference/default), "unit-ambiguity" (a target is
    #: stated in a non-canonical unit that must be converted), "multi-target"
    #: (two or more targets must be hit jointly).
    scenario: str = "complete-info"


def load_gold(path: str = GOLD_PATH) -> list[GoldExperiment]:
    with open(path) as fh:
        return [GoldExperiment(**item) for item in json.load(fh)]


# ---------------------------------------------------------------------------
# Metrics (DesignPlan path — Labwright)
# ---------------------------------------------------------------------------


def relative_error(got: float | None, expected: float) -> float:
    """|got - expected| / expected; inf if got is None.

    A zero-valued target (a spheroid with no anoxic core, unidirectional flow
    with OSI 0) cannot be expressed as a *relative* error — the denominator
    would vanish. For ``expected == 0`` the error is the absolute deviation
    ``|got|``, which the ``<= 0.05`` usable-tolerance then judges directly.
    """
    if got is None:
        return float("inf")
    if expected == 0:
        return abs(got)
    return abs(got - expected) / abs(expected)


def classify_failure(rec: dict, gold: GoldExperiment) -> str:
    """Why an entry failed (or succeeded), as a category.

    - ``"ok"`` — usable: verified/consistent and recovers every gold target.
    - ``"silence"`` — the model produced nothing checkable (no plan for
      Labwright; no reported numbers for the memory systems). The 3
      ``plan:false`` flash refusals are silence.
    - ``"calculation_error"`` — numbers were produced but are internally
      inconsistent / unverifiable (the memory systems' typed-from-memory
      numbers; a Labwright plan the verifier rejects).
    - ``"wrong_target"`` — the answer is internally consistent (or verified)
      but misses the gold value: the model picked the wrong magnitude, the
      wrong formula, or the wrong organ's number.
    """
    if "plan" in rec:  # Labwright record
        if not rec["plan"]:
            return "silence"
        if rec["hallucination_rate"] > 0:
            return "calculation_error"
        return "ok" if rec["valid"] else "wrong_target"
    # memory-system record (bare / soft-gate / self-verify)
    if not rec.get("reported"):
        return "silence"
    if rec["hallucination_rate"] > 0:
        return "calculation_error"
    return "ok" if rec["valid"] else "wrong_target"


#: Gold keys (bare, as used in ``gold.expected``) -> canonical verifier field
#: name. ``classify_unit_misread`` looks the field up in the unit audit table,
#: which is keyed by the verifier's issue names (``derived.shear_pa``), so a
#: bare ``shear_pa`` must be mapped before the alias check can run. The mapping
#: is the union of every design domain's ``Block.field_map`` (declared once in
#: :mod:`labwright.blocks`), first declaration wins — so the shared
#: ``total_medium_ml`` key maps to culture's field, exactly as before.
_FIELD_MAP: dict[str, str] = ALL_FIELD_MAP


def unit_misreads(claimed: dict[str, float | str | None], gold: GoldExperiment) -> dict[str, dict]:
    """Probable unit misreads among the reported numbers.

    For each gold target the system reported, run :func:`classify_unit_misread`
    — a claimed value that is a clean multiple of a known alias ratio (e.g.
    dyn/cm² vs Pa = 10×) with the *right* magnitude is a unit error, not an
    arithmetic one. Returns ``{gold_key: misread_record}``.
    """
    from labwright.verify.units import classify_unit_misread

    out: dict[str, dict] = {}
    for key, expected in gold.expected.items():
        value = claimed.get(key)
        if value is None:
            continue
        field = _FIELD_MAP.get(key, key)
        m = classify_unit_misread(value, expected, field)
        if m:
            out[key] = m
    return out


def _primary_key(gold: GoldExperiment) -> str:
    """The goal's headline target — the first listed expected key."""
    return next(iter(gold.expected)) if gold.expected else ""


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
    # Spheroid domain (3D): expected keys map onto SpheroidPlan fields.
    if plan.spheroid is not None:
        spheroid_map = {
            "spheroid_volume_ul": plan.spheroid.spheroid_volume_ul,
            "expected_diameter_um": plan.spheroid.expected_diameter_um,
            "cells_total": plan.spheroid.cells_total,
            "medium_volume_per_spheroid_ul": plan.spheroid.medium_volume_per_spheroid_ul,
            "total_medium_ml": plan.spheroid.total_medium_ml,
            "cells_per_spheroid": plan.spheroid.cells_per_spheroid,
            "spheroid_count": float(plan.spheroid.spheroid_count),
            "expected_cells_after_growth": plan.spheroid.expected_cells_after_growth,
        }
        for key, value in spheroid_map.items():
            if key in gold.expected and value is not None:
                errs[key] = relative_error(value, gold.expected[key])
    # PK domain: expected keys map onto PkPlan fields. ``flow_rate_uLmin`` is
    # resolved from the PK plan's own field (checked after the flow block so a
    # PK gold is scored against the circuit flow actually used in the PK math).
    if plan.pk is not None:
        pk_map = {
            "extraction_ratio": plan.pk.extraction_ratio,
            "clearance_uLmin": plan.pk.clearance_uLmin,
            "half_life_h": plan.pk.half_life_h,
            "accumulation_ratio": plan.pk.accumulation_ratio,
            "mass_cleared_ug_h": plan.pk.mass_cleared_ug_h,
            "inlet_concentration_uM": plan.pk.inlet_concentration_uM,
            "outlet_concentration_uM": plan.pk.outlet_concentration_uM,
            "flow_rate_uLmin": plan.pk.flow_rate_uLmin,
            "system_volume_uL": plan.pk.system_volume_uL,
            "dose_interval_h": plan.pk.dose_interval_h,
            "molecular_weight_g_mol": plan.pk.molecular_weight_g_mol,
        }
        for key, value in pk_map.items():
            if key in gold.expected and value is not None:
                errs[key] = relative_error(value, gold.expected[key])
    # Seven post-v1 domains (barrier / oxygen / pumpless / breathing /
    # pulsatile / scaling / gradient): score whichever gold keys the plan
    # actually carries, mirroring the culture/spheroid/pk blocks above.
    for key, value in _new_domain_claims(plan).items():
        if key in gold.expected and value is not None:
            errs[key] = relative_error(value, gold.expected[key])
    return errs


def _new_domain_claims(plan: DesignPlan) -> dict[str, float]:
    """Bare gold keys -> values for the seven post-v1 design domains.

    Mirrors the culture/spheroid/pk blocks above: each new block's derived
    fields are read off its plan, ``None`` optional values skipped so a gold
    key that the design does not carry is simply not reported.
    """
    out: dict[str, float] = {}

    def add(fields: tuple[str, ...], obj) -> None:
        for k in fields:
            v = getattr(obj, k)
            if v is not None and not isinstance(v, bool):
                out[k] = float(v)

    if plan.barrier is not None:
        add(("teer_ohm_cm2", "papp_cm_s", "clearance_mL_min"), plan.barrier)
    if plan.oxygen is not None:
        add(("dissolved_o2_mM", "penetration_depth_um", "necrotic_fraction",
             "demand_umol_min"), plan.oxygen)
    if plan.pumpless is not None:
        add(("hydrostatic_head_pa", "driven_flow_rate_uLmin", "peak_wall_shear_pa",
             "volume_per_half_cycle_ul", "oscillatory_shear_index", "cycles_per_hour",
             "shear_ratio_to_target"), plan.pumpless)
    if plan.breathing is not None:
        add(("breaths_per_minute", "cyclic_displacement_um", "strain_rate_per_s",
             "total_cycles", "stretch_duty_fraction", "ali_liquid_film_um"),
            plan.breathing)
    if plan.pulsatile is not None:
        add(("womersley_number", "oscillatory_shear_index", "peak_shear_pa",
             "pulsatility_index"), plan.pulsatile)
    if plan.scaling is not None:
        add(("organ_flow_fraction", "organ_flow_rate_mlmin", "cells_in_organ",
             "allometric_scale", "transit_time_s", "residence_time_match_error_s"),
            plan.scaling)
    if plan.gradient is not None:
        add(("steepness_um_per_mm", "midpoint_conc_um", "relaxation_time_s",
             "flux_mol_m2s"), plan.gradient)
    return out


def _design_claimed(plan: DesignPlan, gold: GoldExperiment) -> dict[str, float | str | None]:
    """The design's value for each gold target key (mirror of parameter_recovery)."""
    out: dict[str, float | str | None] = {}
    if plan.derived is not None:
        d = plan.derived
        mapping = {
            "shear_pa": d.shear_pa, "reynolds": d.reynolds,
            "pressure_drop_pa": d.pressure_drop_pa, "residence_time_s": d.residence_time_s,
            "channel_volume_ul": d.channel_volume_ul, "mean_velocity_mms": d.mean_velocity_mms,
        }
        for key in gold.expected:
            if key in mapping:
                out[key] = mapping[key]
    if "flow_rate_uLmin" in gold.expected and plan.flow is not None:
        out["flow_rate_uLmin"] = plan.flow.flow_rate_uLmin
    if "seed_count" in gold.expected and plan.cells is not None:
        out["seed_count"] = plan.cells.seed_count
    if "dmso_fraction_vv" in gold.expected and plan.dosing is not None:
        out["dmso_fraction_vv"] = plan.dosing.dmso_fraction_vv
    if "n_per_group" in gold.expected and plan.stats is not None:
        out["n_per_group"] = float(plan.stats.n_per_group)
    if plan.culture is not None:
        c = plan.culture
        cmap = {
            "seed_per_well": c.seed_per_well, "total_seed_count": c.total_seed_count,
            "medium_volume_per_well_ml": c.medium_volume_per_well_ml,
            "total_medium_ml": c.total_medium_ml,
            "expected_confluence_pct": c.expected_confluence_pct, "wells": float(c.wells),
        }
        for key in gold.expected:
            if key in cmap and cmap[key] is not None:
                out[key] = cmap[key]
    if plan.spheroid is not None:
        s = plan.spheroid
        smap = {
            "spheroid_volume_ul": s.spheroid_volume_ul,
            "expected_diameter_um": s.expected_diameter_um,
            "cells_total": s.cells_total,
            "medium_volume_per_spheroid_ul": s.medium_volume_per_spheroid_ul,
            "total_medium_ml": s.total_medium_ml,
            "cells_per_spheroid": s.cells_per_spheroid,
            "spheroid_count": float(s.spheroid_count),
            "expected_cells_after_growth": s.expected_cells_after_growth,
        }
        for key in gold.expected:
            if key in smap and smap[key] is not None:
                out[key] = smap[key]
    if plan.pk is not None:
        pk_map = {
            "extraction_ratio": plan.pk.extraction_ratio,
            "clearance_uLmin": plan.pk.clearance_uLmin,
            "half_life_h": plan.pk.half_life_h,
            "accumulation_ratio": plan.pk.accumulation_ratio,
            "mass_cleared_ug_h": plan.pk.mass_cleared_ug_h,
            "inlet_concentration_uM": plan.pk.inlet_concentration_uM,
            "outlet_concentration_uM": plan.pk.outlet_concentration_uM,
            "flow_rate_uLmin": plan.pk.flow_rate_uLmin,
            "system_volume_uL": plan.pk.system_volume_uL,
            "dose_interval_h": plan.pk.dose_interval_h,
            "molecular_weight_g_mol": plan.pk.molecular_weight_g_mol,
        }
        for key in gold.expected:
            if key in pk_map and pk_map[key] is not None:
                out[key] = pk_map[key]
    for key, value in _new_domain_claims(plan).items():
        if key in gold.expected and value is not None:
            out[key] = value
    return out


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
    "spheroid.spheroid_volume_ul", "spheroid.expected_diameter_um",
    "spheroid.cells_total", "spheroid.medium_volume_per_spheroid_ul",
    "spheroid.total_medium_ml", "spheroid.expected_cells_after_growth",
    "pk.extraction_ratio", "pk.clearance_uLmin", "pk.half_life_h",
    "pk.accumulation_ratio", "pk.mass_cleared_ug_h",
    # post-v1 domains
    "barrier.teer_ohm_cm2", "barrier.papp_cm_s", "barrier.clearance_mL_min",
    "oxygen.dissolved_o2_mM", "oxygen.penetration_depth_um",
    "oxygen.necrotic_fraction", "oxygen.demand_umol_min",
    "pumpless.hydrostatic_head_pa", "pumpless.driven_flow_rate_uLmin",
    "pumpless.peak_wall_shear_pa", "pumpless.volume_per_half_cycle_ul",
    "pumpless.oscillatory_shear_index", "pumpless.cycles_per_hour",
    "pumpless.shear_ratio_to_target",
    "breathing.breaths_per_minute", "breathing.cyclic_displacement_um",
    "breathing.strain_rate_per_s", "breathing.total_cycles",
    "breathing.stretch_duty_fraction", "breathing.ali_liquid_film_um",
    "pulsatile.womersley_number", "pulsatile.oscillatory_shear_index",
    "pulsatile.peak_shear_pa", "pulsatile.pulsatility_index",
    "scaling.organ_flow_fraction", "scaling.organ_flow_rate_mlmin",
    "scaling.cells_in_organ", "scaling.allometric_scale",
    "scaling.transit_time_s", "scaling.residence_time_match_error_s",
    "gradient.steepness_um_per_mm", "gradient.midpoint_conc_um",
    "gradient.relaxation_time_s", "gradient.flux_mol_m2s",
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
    if plan.spheroid is None:
        for f in ("spheroid.spheroid_volume_ul", "spheroid.expected_diameter_um",
                  "spheroid.cells_total", "spheroid.medium_volume_per_spheroid_ul",
                  "spheroid.total_medium_ml", "spheroid.expected_cells_after_growth"):
            present.discard(f)
    if plan.pk is None:
        for f in ("pk.extraction_ratio", "pk.clearance_uLmin", "pk.half_life_h",
                  "pk.accumulation_ratio", "pk.mass_cleared_ug_h"):
            present.discard(f)
    # post-v1 blocks (discard a block's whole field set when the plan lacks it)
    _BLOCK_FIELDS = {
        "barrier": ("barrier.teer_ohm_cm2", "barrier.papp_cm_s", "barrier.clearance_mL_min"),
        "oxygen": ("oxygen.dissolved_o2_mM", "oxygen.penetration_depth_um",
                   "oxygen.necrotic_fraction", "oxygen.demand_umol_min"),
        "pumpless": ("pumpless.hydrostatic_head_pa", "pumpless.driven_flow_rate_uLmin",
                     "pumpless.peak_wall_shear_pa", "pumpless.volume_per_half_cycle_ul",
                     "pumpless.oscillatory_shear_index", "pumpless.cycles_per_hour",
                     "pumpless.shear_ratio_to_target"),
        "breathing": ("breathing.breaths_per_minute", "breathing.cyclic_displacement_um",
                      "breathing.strain_rate_per_s", "breathing.total_cycles",
                      "breathing.stretch_duty_fraction", "breathing.ali_liquid_film_um"),
        "pulsatile": ("pulsatile.womersley_number", "pulsatile.oscillatory_shear_index",
                      "pulsatile.peak_shear_pa", "pulsatile.pulsatility_index"),
        "scaling": ("scaling.organ_flow_fraction", "scaling.organ_flow_rate_mlmin",
                    "scaling.cells_in_organ", "scaling.allometric_scale",
                    "scaling.transit_time_s", "scaling.residence_time_match_error_s"),
        "gradient": ("gradient.steepness_um_per_mm", "gradient.midpoint_conc_um",
                     "gradient.relaxation_time_s", "gradient.flux_mol_m2s"),
    }
    for attr, fields in _BLOCK_FIELDS.items():
        if getattr(plan, attr, None) is None:
            present.difference_update(fields)
    return len(errored & present) / max(len(present), 1)


# ---------------------------------------------------------------------------
# Bare-LLM path (lenient number extraction)
# ---------------------------------------------------------------------------


def _is_culture_gold(gold: GoldExperiment) -> bool:
    """True when the gold's expected keys are plate-culture derived numbers."""
    return bool(set(gold.expected) & set(_CULTURE_DERIVED_KEYS))


def _is_spheroid_gold(gold: GoldExperiment) -> bool:
    """True when the gold's expected keys are spheroid numbers.

    Includes the raw design targets (``cells_per_spheroid``, ``spheroid_count``)
    because for 3D culture the seeding decision is itself a target the model
    must get right, and it is only checkable when the model also reports the
    raws that produce the derived sizes.
    """
    return bool(set(gold.expected) & (set(_SPHEROID_DERIVED_KEYS) | set(_SPHEROID_CONSISTENCY_KEYS)))


def _is_pk_gold(gold: GoldExperiment) -> bool:
    """True when the gold's expected keys are PK numbers.

    Includes the raw design targets (``inlet_concentration_uM``,
    ``outlet_concentration_uM``, ``flow_rate_uLmin``) because for a perfused
    clearance study the flow/concentration decision is itself a target the
    model must get right, and it is only checkable when the model also reports
    the raws that produce the derived clearance numbers.
    """
    return bool(set(gold.expected) & (set(_PK_DERIVED_KEYS) | set(_PK_CONSISTENCY_KEYS)))


def _prompt_keys_for(gold: GoldExperiment) -> list[str]:
    """Key set a bare model must report, chosen per gold domain.

    Flow goals need geometry+flow raws (``_CONSISTENCY_KEYS``); plate-culture
    goals need plate_format + seeding density + wells; spheroid goals need the
    vessel format + count + cells-per-spheroid + cell diameter; PK goals need
    inlet/outlet concentration + flow. Everything else is the goal's own
    targets.
    """
    if _is_culture_gold(gold):
        return sorted(set(gold.expected) | set(_CULTURE_CONSISTENCY_KEYS))
    if _is_spheroid_gold(gold):
        return sorted(set(gold.expected) | set(_SPHEROID_CONSISTENCY_KEYS))
    if _is_pk_gold(gold):
        return sorted(set(gold.expected) | set(_PK_CONSISTENCY_KEYS))
    block = _new_domain_block(gold)
    if block is not None:
        return sorted(set(gold.expected) | set(BLOCKS[block].consistency_keys))
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
    elif _is_spheroid_gold(gold):
        check = (
            "BEFORE you finalize: re-derive every derived 3D-culture number "
            f"({', '.join(_SPHEROID_DERIVED_KEYS)}) from your own spheroid_format/"
            "spheroid_count/cells_per_spheroid/cell_diameter_um using the standard "
            "spheroid geometry (solid-sphere packing) and standard vessel volumes, "
            "and correct any value that does not match."
        )
    elif _is_pk_gold(gold):
        check = (
            "BEFORE you finalize: re-derive every derived PK number "
            f"({', '.join(_PK_DERIVED_KEYS)}) from your own "
            "inlet_concentration_uM/outlet_concentration_uM/flow_rate_uLmin (and "
            "system_volume_uL / dose_interval_h / molecular_weight_g_mol when you "
            "reported them) using E = 1 − C_out/C_in, Cl = E·Q, t½ = ln2·V/Cl, "
            "R = 1/(1 − e^(−ln2·τ/t½)) and M = Cl·C_in·MW·6e-5, and correct any "
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


def _self_verify_domain(gold: GoldExperiment) -> tuple[list[str], list[str], str]:
    """(consistency keys, derived keys, formula instruction) for a gold's domain.

    The verifier pass must hand back the raw inputs of the *right* domain: a
    plate-culture gold is checked against plate raws and plate derived numbers,
    a spheroid gold against spheroid raws and spheroid derived numbers, not the
    flow sets a generic prompt would assume.
    """
    if _is_culture_gold(gold):
        return (
            _CULTURE_CONSISTENCY_KEYS, _CULTURE_DERIVED_KEYS,
            "standard multi-well plate dimensions (well surface area and working "
            "volume per format) and the hemocytometer/viability formulas",
        )
    if _is_spheroid_gold(gold):
        return (
            _SPHEROID_CONSISTENCY_KEYS, _SPHEROID_DERIVED_KEYS,
            "standard spheroid geometry (solid-sphere packing, volume-to-diameter) "
            "and the standard ULA-plate / hanging-drop vessel volumes",
        )
    if _is_pk_gold(gold):
        return (
            _PK_CONSISTENCY_KEYS, _PK_DERIVED_KEYS,
            "first-pass clearance (E = 1 − C_out/C_in, Cl = E·Q) plus t½ = ln2·V/Cl, "
            "R = 1/(1 − e^(−ln2·τ/t½)) and M = Cl·C_in·MW·6e-5 when the extra "
            "inputs are present",
        )
    block = _new_domain_block(gold)
    if block is not None:
        return (
            list(BLOCKS[block].consistency_keys),
            list(BLOCKS[block].derived_keys),
            _NEW_DOMAIN_FORMULAS[block],
        )
    return (
        _CONSISTENCY_KEYS, _DERIVED_KEYS,
        "standard rectangular-channel microfluidic formulas",
    )


def self_verify_prompt_for(raw: dict[str, float | str | None], derived_keys: list[str],
                           formulas: str, consistency_keys: list[str]) -> str:
    """Stage-2 prompt: hand the model its own raw inputs and ask it to recompute.

    The second LLM pass plays the role of verifier. It sees only the raw inputs
    the first pass reported — never the deterministic answers — so any agreement
    is the model's own arithmetic, not a leak.
    """
    present = {k: raw[k] for k in consistency_keys if raw.get(k) is not None}
    rendered = ", ".join(f"{k}={present[k]}" for k in consistency_keys if k in present)
    return (
        "A design proposed these raw inputs: " + rendered + ".\n"
        "Using " + formulas + ", recompute EXACTLY these derived values yourself ("
        + ", ".join(derived_keys) + ") from those inputs, and return a single flat "
        "JSON object with ONLY those keys (use exactly these names; do the "
        "arithmetic):\n"
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


#: Gold keys whose value is a format identifier, not a number. ``_find_key`` is
#: float-only, so a bare model that reports ``spheroid_format: "96-ula"`` or
#: ``plate_format: "96-well"`` (the canonical forms) would be scored as if it
#: reported nothing at all — spheroid golds were unverifiable for bare / soft
#: gate / self-verify unconditionally. These are extracted by string instead and
#: normalised by the calculators' own format tables.
_STRING_KEYS = {"plate_format", "spheroid_format"}


def _find_str_key(data: Any, key: str, depth: int = 0) -> str | None:
    """Recursively find a string field by exact name anywhere in a JSON tree."""
    if depth > 12:
        return None
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    # "96" written as a bare number is still a plate format.
                    return str(v)
            found = _find_str_key(v, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_str_key(item, key, depth + 1)
            if found is not None:
                return found
    return None


def _extract_key(data: Any, key: str) -> float | str | None:
    """Extract one key: as a number, or as a string for the format identifiers."""
    if key in _STRING_KEYS:
        return _find_str_key(data, key)
    return _find_key(data, key)


def run_bare_llm(gold: GoldExperiment, chat: Callable, attempts: int = 3) -> dict[str, float | str | None]:
    """Ask a raw LLM for the design numbers; return a lenient extraction.

    Retries on empty/unparseable responses (the models intermittently spend the
    whole token budget on hidden reasoning and emit nothing — a transient
    budget artifact, not a competence verdict). Returns a dict mapping every
    key to a float (or a string for ``plate_format`` / ``spheroid_format``), or
    ``None`` when the model never reported it.
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
        extracted = {k: _extract_key(data, k) for k in keys}
        if any(v is not None for v in extracted.values()):
            return extracted
    return empty


def run_soft_gate(gold: GoldExperiment, chat: Callable, attempts: int = 3) -> dict[str, float | str | None]:
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
        extracted = {k: _extract_key(data, k) for k in keys}
        if any(v is not None for v in extracted.values()):
            return extracted
    return empty


def run_self_verify(gold: GoldExperiment, chat: Callable, attempts: int = 2) -> dict[str, float | str | None]:
    """Two-stage 'LLM as its own verifier': propose, then recompute.

    Stage 1 is exactly the bare prompt (propose numbers from memory). Stage 2
    hands the model its own reported raw inputs back and asks it to recompute the
    derived numbers of the goal's own domain (flow, plate-culture or spheroid)
    *itself*. The final answer is the stage-2 numbers where returned; if the
    verifier pass returns nothing checkable, the proposal stands unverified
    (scored exactly like a bare answer). This is the naive alternative to
    Labwright's deterministic verifier: can a second LLM pass correct the first
    LLM's arithmetic? The benchmark shows it cannot reliably.
    """
    extracted = run_bare_llm(gold, chat, attempts=attempts)
    consistency, derived, formulas = _self_verify_domain(gold)
    raw = {k: extracted.get(k) for k in consistency}
    if None in raw.values():
        return extracted  # missing raws → nothing for a verifier to check
    prompt = self_verify_prompt_for(raw, derived, formulas, consistency)
    for _ in range(attempts):
        text = chat(prompt) or ""
        if not text.strip():
            continue
        try:
            data = _extract_json(text)
        except Exception:
            continue
        stage2 = {k: _extract_key(data, k) for k in derived}
        if any(v is not None for v in stage2.values()):
            merged = dict(extracted)
            merged.update(stage2)
            return merged
    return extracted  # verifier returned nothing checkable → proposal stands


def bare_recovery(extracted: dict[str, float | str | None], gold: GoldExperiment) -> dict[str, float]:
    """Relative error of the *reported* numbers vs the gold standard."""
    return {key: relative_error(extracted.get(key), expected) for key, expected in gold.expected.items()}


def bare_checkable(extracted: dict[str, float | str | None]) -> bool:
    """Whether the bare answer reported enough to cross-check any derived number.

    Flow-verifiable = geometry + flow *and* at least one derived flow metric.
    Culture-verifiable = plate_format + seeding density (+ wells) *and* at
    least one derived culture number. PK-verifiable = inlet + outlet + flow
    *and* at least one derived PK number. A bare answer that only states a
    headline number (e.g. ``seed_count``) without the raw inputs that produce
    it cannot be cross-checked.
    """
    chip = (extracted.get("width_um"), extracted.get("height_um"), extracted.get("length_mm"))
    flow_rate = extracted.get("flow_rate_uLmin")
    if None not in chip and flow_rate is not None:
        return any(extracted.get(k) is not None for k in _DERIVED_KEYS)
    if extracted.get("plate_format") and extracted.get("seeding_density_cells_cm2") is not None:
        return any(extracted.get(k) is not None for k in _CULTURE_DERIVED_KEYS)
    if (
        extracted.get("cells_per_spheroid") is not None
        and extracted.get("cell_diameter_um") is not None
    ):
        # geometry (diameter / volume) is cross-checkable from cells × cell size
        # alone; the vessel format is only needed for the medium fields.
        return any(extracted.get(k) is not None for k in _SPHEROID_DERIVED_KEYS)
    if (
        extracted.get("inlet_concentration_uM") is not None
        and extracted.get("outlet_concentration_uM") is not None
        and flow_rate is not None
    ):
        # clearance is cross-checkable from inlet/outlet × flow alone; the
        # volume / interval / MW inputs are only needed for the extra fields.
        return any(extracted.get(k) is not None for k in _PK_DERIVED_KEYS)
    block = _new_domain_block_of_extracted(extracted)
    if block is not None:
        return any(extracted.get(k) is not None for k in BLOCKS[block].derived_keys)
    return False


def _flow_hallucination(extracted: dict[str, float | str | None]) -> float | None:
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


def _culture_hallucination(extracted: dict[str, float | str | None]) -> float | None:
    """Cross-check reported plate-culture numbers against the model's own raws.

    Recomputes seed_per_well / total_seed_count / medium_volume_per_well_ml /
    total_medium_ml from the reported plate_format + seeding density (+ wells)
    with the culture calculators. ``expected_confluence_pct`` is only
    cross-checked when the model also reported the growth inputs that produce it
    (confluent density + doubling time + duration); a confluence number typed
    without those is neither counted right nor wrong (it cannot be re-derived) —
    and a model over-reporting it must never crash the run. Returns the error
    fraction, or ``None`` when the answer is not culture-verifiable.
    """
    from labwright.calc import cell as calc_cell
    from labwright.calc import culture as calc_culture

    plate = extracted.get("plate_format")
    density = extracted.get("seeding_density_cells_cm2")
    if not plate or density is None:
        return None
    try:
        wells = extracted.get("wells") if extracted.get("wells") is not None else 1
        per_well = calc_culture.cells_per_well(density, plate)
        med = calc_culture.medium_volume_per_well(plate)
        computed: dict[str, float] = {
            "seed_per_well": per_well,
            "total_seed_count": per_well * wells,
            "medium_volume_per_well_ml": med,
            "total_medium_ml": med * wells,
        }
    except (ValueError, TypeError, ArithmeticError):
        return None
    if (extracted.get("confluent_density_cells_cm2") is not None
            and extracted.get("doubling_time_h") is not None
            and extracted.get("culture_duration_h") is not None):
        try:
            area = calc_culture.well_surface_area_cm2(plate)
            final = calc_cell.cell_count_after_time(
                per_well, extracted["doubling_time_h"], extracted["culture_duration_h"]
            )
            computed["expected_confluence_pct"] = calc_culture.cell_count_to_confluence(
                final, extracted["confluent_density_cells_cm2"], area
            )
        except (ValueError, TypeError, ArithmeticError):
            pass
    if not all(math.isfinite(v) for v in computed.values()):
        return None
    claimed = {k: extracted.get(k) for k in computed}
    present = [k for k in computed if claimed[k] is not None]
    if not present:
        return None  # no derived culture number reported → nothing to check
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


def _spheroid_hallucination(extracted: dict[str, float | str | None]) -> float | None:
    """Cross-check reported 3D-spheroid numbers against the model's own raws.

    Each derived number is recomputed from exactly the raws it needs, so a
    geometry-only answer (diameter / volume from cells_per_spheroid ×
    cell_diameter_um) is checkable even when the model never names a vessel
    format, while vessel numbers (medium volume / total medium) additionally
    need a parseable ``spheroid_format``. A reported vessel number with no
    usable format is *not* counted as wrong (it cannot be re-derived), but it is
    also not counted as verified — if that was the only thing reported, the
    answer is unverifiable. Returns the error fraction, or ``None`` when the
    answer is not spheroid-verifiable.
    """
    from labwright.calc import cell as calc_cell
    from labwright.calc import spheroid as calc_spheroid

    per_sph = extracted.get("cells_per_spheroid")
    cell_d = extracted.get("cell_diameter_um")
    if per_sph is None or cell_d is None:
        return None  # geometry (and everything else) needs these two raws
    computed: dict[str, float] = {}
    try:
        computed["expected_diameter_um"] = calc_spheroid.spheroid_diameter_from_cells(per_sph, cell_d)
        computed["spheroid_volume_ul"] = calc_spheroid.spheroid_volume_from_cells(per_sph, cell_d)
    except (ValueError, TypeError, ArithmeticError):
        return None
    count = extracted.get("spheroid_count")
    if count is not None:
        try:
            computed["cells_total"] = calc_spheroid.cells_needed_for_spheroids(count, per_sph)
        except (ValueError, TypeError, ArithmeticError):
            return None
    fmt = extracted.get("spheroid_format")
    if fmt:
        try:
            med = calc_spheroid.medium_volume_per_spheroid(fmt)
            computed["medium_volume_per_spheroid_ul"] = med
            if count is not None:
                computed["total_medium_ml"] = calc_spheroid.total_medium_volume(count, med)
        except (ValueError, TypeError, ArithmeticError):
            # Unparseable vessel (e.g. "solid_sphere", "single_96_well_plate"):
            # the vessel fields cannot be re-derived, but the geometry can be.
            pass
    dt = extracted.get("doubling_time_h")
    dur = extracted.get("culture_duration_h")
    if dt is not None and dur is not None:
        try:
            computed["expected_cells_after_growth"] = calc_cell.cell_count_after_time(per_sph, dt, dur)
        except (ValueError, TypeError, ArithmeticError):
            pass
    if not all(math.isfinite(v) for v in computed.values()):
        return None
    claimed = {k: extracted.get(k) for k in computed}
    present = [k for k in computed if claimed[k] is not None]
    if not present:
        return None  # no derived spheroid number reported → nothing to check
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


def _pk_hallucination(extracted: dict[str, float | str | None]) -> float | None:
    """Cross-check reported PK numbers against the model's own raws.

    Extraction ratio and clearance are recomputed from the reported inlet /
    outlet concentrations × flow. The extra fields (half-life, accumulation
    ratio, mass cleared) are cross-checked only when the model also reported
    the inputs that produce them (system volume, dose interval, molecular
    weight); a value typed without its input is neither counted right nor
    wrong (it cannot be re-derived). Returns the error fraction, or ``None``
    when the answer is not PK-verifiable (no inlet/outlet/flow, or
    inlet/outlet/flow but no derived PK number at all).
    """
    from labwright.calc import pk as calc_pk

    c_in = extracted.get("inlet_concentration_uM")
    c_out = extracted.get("outlet_concentration_uM")
    flow = extracted.get("flow_rate_uLmin")
    if c_in is None or c_out is None or flow is None:
        return None
    computed: dict[str, float] = {}
    try:
        computed["extraction_ratio"] = calc_pk.extraction_ratio(c_in, c_out)
        computed["clearance_uLmin"] = calc_pk.clearance_uLmin(c_in, c_out, flow)
    except (ValueError, TypeError, ArithmeticError):
        return None
    vol = extracted.get("system_volume_uL")
    if vol is not None:
        try:
            computed["half_life_h"] = calc_pk.half_life_h(vol, computed["clearance_uLmin"])
        except (ValueError, TypeError, ArithmeticError):
            return None
    interval = extracted.get("dose_interval_h")
    if computed.get("half_life_h") is not None and interval is not None:
        try:
            computed["accumulation_ratio"] = calc_pk.accumulation_ratio(
                computed["half_life_h"], interval
            )
        except (ValueError, TypeError, ArithmeticError):
            return None
    mw = extracted.get("molecular_weight_g_mol")
    if mw is not None:
        try:
            computed["mass_cleared_ug_h"] = calc_pk.mass_cleared_ug_h(
                computed["clearance_uLmin"], c_in, mw
            )
        except (ValueError, TypeError, ArithmeticError):
            return None
    if not all(math.isfinite(v) for v in computed.values()):
        return None
    claimed = {k: extracted.get(k) for k in computed}
    present = [k for k in computed if claimed[k] is not None]
    if not present:
        return None  # no derived PK number reported → nothing to check
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


#: Post-v1 design domains, mirroring the seven Blocks registered in
#: ``labwright.blocks``. These did not exist when the bare/soft-gate paths were
#: first written; without explicit branches a bare run on ``gold_new_domains``
#: would be prompted for *flow* keys and score 1.0 unconditionally.
_NEW_DOMAIN_BLOCKS = ("barrier", "oxygen", "pumpless", "breathing",
                     "pulsatile", "scaling", "gradient")

_NEW_DOMAIN_FORMULAS: dict[str, str] = {
    "barrier": "Transwell QC: TEER = (R_total − R_blank) × A and Papp = "
               "flux/(60·A·C0), clearance = Papp·A·60",
    "oxygen": "Henry's law for dissolved O2 from target pO2, the Krogh "
               "penetration depth from the cell-type OCR, and the necrotic-core "
               "fraction from penetration vs spheroid diameter",
    "pumpless": "gravity rocking: hydrostatic head = ρg·L·sinθ, Hagen-Poiseuille "
                 "flow from the head, peak wall shear and OSI from the rocking "
                 "waveform",
    "breathing": "lung stretch: breaths/min = f·60, strain rate = (ε/100)·f, "
                  "cyclic displacement = ε·L, duty fraction and ALI film as defined",
    "pulsatile": "cardiac waveform: Womersley α from h/2·√(ωρ/μ), OSI from the "
                  "sinusoid, peak shear = mean + amplitude",
    "scaling": "body-on-chip allometry: organ flow fraction from the physiology "
                "table, organ flow = fraction × cardiac output, cells scaled by "
                "organ mass fraction",
    "gradient": "chemotaxis: steepness = (C_src − C_sink)/d, midpoint from the "
                 "steady-state profile, relaxation time = d²/(2D)",
}


def _new_domain_block(gold: GoldExperiment) -> str | None:
    """Block name for a post-v1-domain gold, else ``None``.

    ``oscillatory_shear_index`` is shared by pumpless and pulsatile, so a
    pulsatile gold would match pumpless by any-overlap; pick the block with the
    *largest* overlap with the gold's expected keys instead (tie-break by
    ``_NEW_DOMAIN_BLOCKS`` order).
    """
    exp = set(gold.expected)
    best: tuple[int, int, str] | None = None
    for i, name in enumerate(_NEW_DOMAIN_BLOCKS):
        overlap = len(exp & set(BLOCKS[name].derived_keys))
        if overlap and (best is None or (overlap, -i) > (best[0], -best[1])):
            best = (overlap, i, name)
    return best[2] if best is not None else None


def _new_domain_block_of_extracted(extracted: dict[str, float | str | None]) -> str | None:
    """Which post-v1 block the answer reports raws for, else ``None``."""
    for name in _NEW_DOMAIN_BLOCKS:
        ckeys = BLOCKS[name].consistency_keys
        if all(extracted.get(k) is not None for k in ckeys):
            return name
    return None


def _new_domain_computed(extracted: dict[str, float | str | None], block: str) -> dict[str, float]:
    """Recompute the block's derived values from the reported raws.

    Mirrors the ``derive_*`` chains in :mod:`labwright.design` but only for keys
    computable from the raws actually reported, so an answer that states just the
    essential inputs is still cross-checked on the numbers those inputs
    determine. Keys needing a field the model did not report are skipped —
    neither counted right nor wrong, exactly like the optional-PK-inputs rule in
    :func:`_pk_hallucination`.
    """
    from labwright.calc import barrier as cb_
    from labwright.calc import o2 as co2_
    from labwright.calc import pumpless as cp_
    from labwright.calc import breathing as cbr_
    from labwright.calc import pulsatile as cps_
    from labwright.calc import scaling as csc_
    from labwright.calc import gradient as cg_
    out: dict[str, float] = {}
    try:
        if block == "barrier":
            A = extracted.get("insert_area_cm2")
            Rt = extracted.get("resistance_total_ohm")
            Rb = extracted.get("resistance_blank_ohm")
            if None not in (A, Rt, Rb):
                out["teer_ohm_cm2"] = cb_.teer_ohm_cm2(Rt, Rb, A)
                if extracted.get("flux_nmol_min") is not None and extracted.get("donor_conc_um") is not None:
                    papp = cb_.papp_cm_s(extracted["flux_nmol_min"], A, extracted["donor_conc_um"])
                    out["papp_cm_s"] = papp
                    out["clearance_mL_min"] = cb_.clearance_mL_min(papp, A)
        elif block == "oxygen":
            po2 = extracted.get("target_po2_mmhg")
            if po2 is not None:
                out["dissolved_o2_mM"] = co2_.o2_conc_mm_from_po2(po2)
                prof = None
                if extracted.get("cell_type"):
                    try:
                        from labwright.physiology import lookup_cell
                        prof = lookup_cell(extracted["cell_type"])
                    except Exception:
                        prof = None
                if extracted.get("cell_density_cells_ml") is not None and prof is not None:
                    ocr = getattr(prof, "o2_consumption_nmol_min_1e6", None)
                    if ocr:
                        fmol = co2_.nmol_min_per_1e6_to_fmol_s((ocr[0] + ocr[1]) / 2.0)
                        q = co2_.volumetric_o2_consumption(fmol, extracted["cell_density_cells_ml"])
                        pen = co2_.o2_penetration_depth_um(q)
                        out["penetration_depth_um"] = pen
                        if extracted.get("spheroid_diameter_um") is not None:
                            out["necrotic_fraction"] = co2_.spheroid_necrotic_fraction(
                                extracted["spheroid_diameter_um"], pen)
        elif block == "pumpless":
            t = extracted.get("tilt_angle_deg"); L = extracted.get("channel_length_mm")
            w = extracted.get("width_um"); h = extracted.get("height_um")
            rp = extracted.get("rocking_half_period_s")
            if None not in (t, L, w, h, rp):
                rho = extracted.get("density_kgm3") or 1000.0
                visc = extracted.get("viscosity_pas") or 1e-3
                head = cp_.hydrostatic_pressure_pa(rho, t, L)
                out["hydrostatic_head_pa"] = head
                flow = cp_.flow_rate_from_pressure_head(head, w, h, L, visc)
                out["driven_flow_rate_uLmin"] = flow
                peak = cp_.peak_wall_shear_from_head(head, w, h, L)
                out["peak_wall_shear_pa"] = peak
                out["volume_per_half_cycle_ul"] = cp_.rocking_volume_per_half_cycle_ul(flow, rp)
                bsf = extracted.get("backward_shear_fraction")
                out["oscillatory_shear_index"] = cp_.oscillatory_shear_index(
                    peak, peak * (bsf if bsf is not None else 1.0))
                out["cycles_per_hour"] = cp_.cycles_per_hour(rp)
        elif block == "breathing":
            f = extracted.get("frequency_hz"); st = extracted.get("strain_pct")
            if None not in (f, st):
                out["breaths_per_minute"] = cbr_.breaths_per_minute(f)
                out["strain_rate_per_s"] = cbr_.strain_rate_per_s(st, f)
                span = extracted.get("membrane_span_um") or 250.0
                out["cyclic_displacement_um"] = cbr_.cyclic_displacement_um(st, span)
                if extracted.get("culture_duration_h") is not None:
                    out["total_cycles"] = cbr_.total_cycles(extracted["culture_duration_h"], f)
                if extracted.get("stretch_seconds") is not None and extracted.get("cycle_seconds") is not None:
                    out["stretch_duty_fraction"] = cbr_.stretch_duty_fraction(
                        extracted["stretch_seconds"], extracted["cycle_seconds"])
                if extracted.get("apical_volume_ul") is not None and extracted.get("surface_area_cm2") is not None:
                    out["ali_liquid_film_um"] = cbr_.ali_liquid_film_um(
                        extracted["apical_volume_ul"], extracted["surface_area_cm2"])
        elif block == "pulsatile":
            f = extracted.get("frequency_hz"); h = extracted.get("channel_height_um")
            sm = extracted.get("shear_mean_pa"); sa = extracted.get("shear_amplitude_pa")
            if None not in (f, h, sm, sa):
                visc = extracted.get("viscosity_pas") or 1e-3
                rho = extracted.get("density_kgm3") or 1000.0
                out["womersley_number"] = cps_.womersley_number(f, h, visc, rho)
                out["oscillatory_shear_index"] = cps_.oscillatory_shear_index_from_sinusoid(sm, sa)
                out["peak_shear_pa"] = cps_.peak_shear_of_sinusoid(sm, sa)
                if (extracted.get("peak_flow_uLmin") is not None
                        and extracted.get("minimum_flow_uLmin") is not None
                        and extracted.get("mean_flow_uLmin") is not None):
                    out["pulsatility_index"] = cps_.pulsatility_index(
                        extracted["peak_flow_uLmin"], extracted["minimum_flow_uLmin"],
                        extracted["mean_flow_uLmin"])
        elif block == "scaling":
            organ = extracted.get("organ")
            if organ is not None and extracted.get("total_cells_chip") is not None:
                out["organ_flow_fraction"] = csc_.organ_flow_fraction(organ)
                out["organ_flow_rate_mlmin"] = csc_.organ_flow_rate_mlmin(
                    organ, extracted.get("cardiac_output_mlmin") or csc_.CARDIAC_OUTPUT_MLMIN)
                om = csc_.ORGAN_MASS_G.get(organ)
                if om is not None:
                    bm = extracted.get("body_mass_g") or csc_.BODY_MASS_G
                    out["cells_in_organ"] = csc_.scale_cell_number(om, bm, extracted["total_cells_chip"])
                    out["allometric_scale"] = csc_.allometric_metabolic_scale(om, bm)
                if extracted.get("chip_volume_ul") is not None and extracted.get("flow_rate_uLmin") is not None:
                    out["transit_time_s"] = csc_.transit_time_s(
                        extracted["chip_volume_ul"], extracted["flow_rate_uLmin"])
        elif block == "gradient":
            src = extracted.get("source_conc_um")
            snk = extracted.get("sink_conc_um")
            dist = extracted.get("distance_um")
            if None not in (src, snk, dist):
                out["steepness_um_per_mm"] = cg_.linear_gradient_steepness_um_per_mm(src, snk, dist)
                out["midpoint_conc_um"] = cg_.steady_state_profile_conc_um(src, snk, dist, dist / 2.0)
                diff = extracted.get("diffusivity_m2s") or cg_.SMALL_MOLECULE_DIFFUSIVITY_M2S
                out["relaxation_time_s"] = cg_.diffusive_relaxation_time_s(dist, diff)
                out["flux_mol_m2s"] = cg_.diffusive_flux_mol_m2s(src, snk, dist, diff)
    except (ValueError, TypeError, ArithmeticError, KeyError):
        return {}
    return out


def _new_domain_hallucination(extracted: dict[str, float | str | None], block: str) -> float | None:
    """Cross-check reported post-v1 numbers against the model's own raws.

    Returns the error fraction, or ``None`` when nothing is re-derivable (the
    "unverifiable" convention maps to 1.0 upstream, matching
    :func:`bare_hallucination`'s other domains).
    """
    computed = _new_domain_computed(extracted, block)
    if not computed or not all(math.isfinite(v) for v in computed.values()):
        return None
    claimed = {k: extracted.get(k) for k in computed}
    present = [k for k in computed if claimed[k] is not None]
    if not present:
        return None
    wrong = sum(
        1 for k in present
        if abs(claimed[k] - computed[k]) > BARE_CONSISTENCY_TOL * max(abs(computed[k]), 1e-12)
    )
    return wrong / len(present)


def bare_hallucination(extracted: dict[str, float | str | None]) -> float:
    """Fraction of reported derived numbers inconsistent with the model's own raw inputs.

    Checks whichever domain the answer is verifiable in (flow, then culture,
    then spheroid, then pk). An answer that is not verifiable in any — no
    geometry+flow, or geometry+flow but **no derived flow numbers at all**, or
    no plate+density and no culture numbers, or no spheroid raws and no
    spheroid numbers, or no inlet/outlet/flow and no PK numbers — is scored
    1.0. The second case matters: a design whose every number is typed
    from memory and cannot be re-derived from the model's own inputs is exactly
    the case Labwright refuses to trust ("numbers you type are not trusted").
    This mirrors the Labwright convention where a run that never submits a plan
    is scored hallucination 1.0.
    """
    rate = _flow_hallucination(extracted)
    if rate is not None:
        return rate
    rate = _culture_hallucination(extracted)
    if rate is not None:
        return rate
    rate = _spheroid_hallucination(extracted)
    if rate is not None:
        return rate
    rate = _pk_hallucination(extracted)
    if rate is not None:
        return rate
    block = _new_domain_block_of_extracted(extracted)
    if block is not None:
        rate = _new_domain_hallucination(extracted, block)
        if rate is not None:
            return rate
    return 1.0


def run_labwright(goal: str, agent_factory: Callable) -> tuple[DesignPlan | None, str | None, Any]:
    """Run the real Labwright pipeline.

    Returns ``(design, error, result)``. When the agent produced no design (``plan:
    false``), ``error`` carries the agent's own failure reason so a silent
    refusal is auditable rather than an unexplained blank. The third value is the
    agent's full :class:`~labwright.agent.AgentResult` (tool-call trace).
    """
    result = agent_factory().run(goal)
    return result.design, result.error, result


def run_tools_no_gate(goal: str, agent_factory_nogate: Callable) -> tuple[DesignPlan | None, str | None, Any]:
    """Run the *no-gate ablation*: the same tool loop, verifier switched off.

    ``agent_factory_nogate`` builds a :class:`~labwright.agent.DesignAgent`
    with ``verify_gate=False`` — same calculators, same loop, but
    ``submit_design`` never verifies and always accepts, and the system prompt
    drops the verification discipline. The plans are scored post-hoc by the
    identical :func:`_score_design`, so the only difference from ``labwright``
    is the verification layer. That isolates what the verifier adds beyond the
    calculators themselves (the response to the "circular verification"
    criticism).

    The third return value is the agent's full :class:`~labwright.agent.AgentResult`
    (tool-call trace) so the record can report how many calculators the no-gate
    agent actually called.
    """
    result = agent_factory_nogate().run(goal)
    return result.design, result.error, result


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


def _score_reported(reported: dict[str, float | str | None], gold: GoldExperiment) -> dict[str, Any]:
    """Score a flat 'reported numbers' answer with the bare-LLM convention.

    Shared by bare, soft-gate and self-verify so the three competitors are
    judged by identical extraction, tolerance and verifiability rules — only the
    prompt/stage structure differs.
    """
    rec = bare_recovery(reported, gold)
    hall = bare_hallucination(reported)
    primary = _primary_key(gold)
    record = {
        "reported": {k: v for k, v in reported.items() if v is not None},
        "verifiable": bare_checkable(reported),
        "recovery": {k: round(v, 6) for k, v in rec.items()},
        "hallucination_rate": round(hall, 6),
        "valid": hall == 0.0 and all(err <= 0.05 for err in rec.values()),
    }
    record["unit_misread"] = unit_misreads(reported, gold)
    record["failure"] = classify_failure(record, gold)
    record["target_selected"] = (
        bool(primary) and primary in rec and rec[primary] <= 0.05
    )
    return record


def _score_design(plan: DesignPlan | None, error: str | None, gold: GoldExperiment) -> dict[str, Any]:
    """Score a verified design path — Labwright or the fine-tuned fast path.

    Shared by the agent-built Labwright pipeline and the deterministic
    extractor fast path, so both are judged by identical recovery, tolerance
    and verifiability rules. The only difference is *how the plan was built*,
    which the record's ``plan``/``error`` fields make auditable.
    """
    lw_rec: dict[str, float] = {}
    lw_hall = 1.0
    claimed: dict[str, float | str | None] = {}
    if plan is not None:
        lw_rec = parameter_recovery(gold, plan)
        lw_hall = hallucination_rate(plan)
        claimed = _design_claimed(plan, gold)
    record = {
        "plan": plan is not None,
        "error": error,
        "recovery": {k: round(v, 6) for k, v in lw_rec.items()},
        "hallucination_rate": round(lw_hall, 6),
        # usable: a plan that verifies AND recovers every gold target.
        "valid": (
            plan is not None
            and lw_hall == 0.0
            and bool(lw_rec)
            and all(err <= 0.05 for err in lw_rec.values())
        ),
    }
    record["unit_misread"] = unit_misreads(claimed, gold)
    record["failure"] = classify_failure(record, gold)
    primary = _primary_key(gold)
    record["target_selected"] = (
        plan is not None and bool(primary) and primary in lw_rec and lw_rec[primary] <= 0.05
    )
    return record


def run_finetuned(gold: GoldExperiment, extractor: Callable) -> tuple[DesignPlan | None, str | None]:
    """Run the fine-tuned extractor fast path: goal → raw → derive → verify.

    Returns ``(design, error)`` in the same contract as :func:`run_labwright`.
    ``extractor`` carries :meth:`~labwright.extract.pipeline.Extractor.extract_plan`
    (goal → plan, issues, error). The extractor never writes a derived number:
    its raw crosses the same gate as the agent's ``submit_design``, so this is
    Labwright's deterministic fast path — no agent loop, no API cost — scored by
    the exact same usable/hallucination rules.

    Fairness note: the extractor was fine-tuned on synthetic flow/culture
    instances whose shear targets are reused from the benchmark gold sets (see
    ``labwright/extract/synthetic.py``), so its numbers on those domains are
    *in-distribution* and must be labelled as such at report time; on domains it
    never trained on (spheroid, PK) it is a clean out-of-distribution test.
    """
    try:
        plan, _issues, error = extractor.extract_plan(gold.goal)
    except Exception as exc:  # noqa: BLE001 - an extractor failure is a scored outcome
        return None, f"extractor_error: {exc}"
    if plan is None:
        return None, error or "no_plan"
    return plan, None


def _run_system(
    name: str,
    gold: GoldExperiment,
    chat: Callable,
    agent_factory: Callable,
    extractor: Callable | None = None,
    agent_factory_nogate: Callable | None = None,
    agent_factory_iter: Callable | None = None,
) -> dict[str, Any]:
    """Run one named system on one gold entry and return its scored record."""
    if name == "labwright":
        lw, lw_error, lw_result = run_labwright(gold.goal, agent_factory)
        rec = _score_design(lw, lw_error, gold)
        # Same tool-use trace as tool_no_gate, for the ablation comparison.
        rec["tool_calls"] = sum(1 for s in lw_result.steps if isinstance(s, dict) and s.get("tool"))
        rec["prose_refusals"] = sum(
            1 for s in lw_result.steps if isinstance(s, dict) and s.get("type") == "prose-refused"
        )
        rec["no_plan"] = lw is None
        return rec
    if name == "labwright_iter":
        if agent_factory_iter is None:
            raise ValueError("the 'labwright_iter' system requires an iterating agent factory")
        lw, lw_error, lw_result = run_labwright(gold.goal, agent_factory_iter)
        rec = _score_design(lw, lw_error, gold)
        rec["tool_calls"] = sum(1 for s in lw_result.steps if isinstance(s, dict) and s.get("tool"))
        rec["prose_refusals"] = sum(
            1 for s in lw_result.steps if isinstance(s, dict) and s.get("type") == "prose-refused"
        )
        rec["fix_rounds"] = sum(
            1 for s in lw_result.steps
            if isinstance(s, dict) and s.get("type") == "review_required"
        )
        rec["no_plan"] = lw is None
        return rec
    if name == "tool_no_gate":
        if agent_factory_nogate is None:
            raise ValueError("the 'tool_no_gate' system requires a no-gate agent factory (verify_gate=False)")
        plan, error, result = run_tools_no_gate(gold.goal, agent_factory_nogate)
        rec = _score_design(plan, error, gold)
        # Diagnostic trace: did removing verification change tool usage / refusals?
        rec["tool_calls"] = sum(1 for s in result.steps if isinstance(s, dict) and s.get("tool"))
        rec["prose_refusals"] = sum(
            1 for s in result.steps if isinstance(s, dict) and s.get("type") == "prose-refused"
        )
        rec["no_plan"] = plan is None
        return rec
    if name == "finetuned":
        if extractor is None:
            raise ValueError("the 'finetuned' system requires an extractor with extract_plan()")
        plan, error = run_finetuned(gold, extractor)
        return _score_design(plan, error, gold)
    return _score_reported(_SYSTEM_RUNNERS[name](gold, chat, agent_factory), gold)


def evaluate(
    gold: list[GoldExperiment],
    agent_factory: Callable,
    chat: Callable,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    systems: tuple[str, ...] = ("bare", "labwright"),
    extractor: Callable | None = None,
    agent_factory_nogate: Callable | None = None,
    agent_factory_iter: Callable | None = None,
) -> dict[str, Any]:
    """Run the requested systems on every gold experiment and aggregate metrics.

    ``systems`` names which systems to run (bare / soft_gate / self_verify /
    labwright / tool_no_gate / labwright_iter / finetuned, any subset). The
    default keeps the historical bare-vs-Labwright comparison; the competitor
    baselines are extra systems scored by the same rules.

    ``extractor`` supplies :meth:`extract_plan` for the ``finetuned`` system —
    the fine-tuned raw-input extractor run as Labwright's deterministic fast
    path (no API cost). Its in-distribution / out-of-distribution split across
    gold domains is documented on :func:`run_finetuned`.

    ``agent_factory_nogate`` supplies the ``tool_no_gate`` ablation system — a
    :class:`~labwright.agent.DesignAgent` built with ``verify_gate=False``. It
    is required only when ``tool_no_gate`` is named in ``systems``.

    ``agent_factory_iter`` supplies the ``labwright_iter`` system — the same
    agent with ``max_submission_attempts > 1``, so a ``review_required`` verdict
    feeds back into the loop and the agent fixes and resubmits. Required only
    when ``labwright_iter`` is named in ``systems``.
    """
    summary: dict[str, Any] = {"n_gold": len(gold), "per_entry": []}
    for name in systems:
        summary[name] = {"recovery": {}, "hallucination_rate": []}

    for g in gold:
        entry: dict[str, Any] = {
            "id": g.id,
            "gold": {
                "id": g.id,
                "blind_strength": g.blind_strength,
                "scenario": g.scenario,
            },
        }
        for name in systems:
            if progress:
                progress(f"[{g.id}] {name} ...")
            rec = _run_system(
                name, g, chat, agent_factory,
                extractor=extractor, agent_factory_nogate=agent_factory_nogate,
                agent_factory_iter=agent_factory_iter,
            )
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
    "run_finetuned",
    "run_tools_no_gate",
    "bare_prompt_for",
    "soft_gate_prompt_for",
    "self_verify_prompt_for",
    "BARE_CONSISTENCY_TOL",
]
