"""Design construction — the boundary between the LLM and the math.

The agent proposes **raw inputs** (geometry, flow, cells, dosing, statistics
assumptions). It never writes a derived number. :func:`build_design` derives
every computed quantity through the calculators, then :func:`submit_design`
runs the verifier.

This split is what makes Labwright's numbers trustworthy: an LLM that asserts
``shear_pa=0.25`` can be wrong; a design whose ``shear_pa`` was produced by
:mod:`labwright.calc` cannot.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from labwright.blocks import ALL_DERIVED_KEYS
from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import microfluidics as mf
from labwright.calc import pk as calc_pk
from labwright.calc import spheroid as calc_spheroid
from labwright.schema.design import (
    CellPlan,
    ChipGeometry,
    CulturePlan,
    DerivedFlowMetrics,
    DesignPlan,
    DosePlan,
    FlowParams,
    PkPlan,
    SpheroidPlan,
    StatsPlan,
)
from labwright.verify.checker import format_issues, verify_design

#: Field names the calculators own. ``submit_design`` refuses any of these from
#: the model — a derived number is computed, never accepted. Kept as a plain set
#: so a smuggled field is rejected with a clear message instead of being
#: silently overwritten (the old pydantic default was to drop it). The set is
#: the union of every design domain's derived keys, declared once per domain in
#: :mod:`labwright.blocks` — adding a domain's derived fields means editing that
#: domain's ``Block``, not this list.
_DERIVED_FIELD_NAMES: frozenset[str] = ALL_DERIVED_KEYS


# ---------------------------------------------------------------------------
# Raw input the LLM is allowed to propose
# ---------------------------------------------------------------------------


class DesignInput(BaseModel):
    """Everything the agent is allowed to choose. No derived numbers here."""

    # A key the extractor/LLM invents (e.g. ``cell_count``, ``flow_rate_ml_min``)
    # is a schema error, not something to silently drop — the whole point of the
    # gate is that a hallucinated field must surface loudly, not pass unnoticed.
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(description="Experimental goal in one sentence")
    rationale: str = Field(description="Why this design; assumptions and references")
    chip: ChipGeometry | None = Field(
        default=None, description="Channel geometry (omit for plate-only culture designs)"
    )
    flow: FlowParams | None = Field(
        default=None, description="Perfusion inputs (omit for plate-only culture designs)"
    )
    cells: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, seeding_density_cells_cm2, culture_area_cm2, "
        "doubling_time_h, culture_duration_h (no seed_count)",
    )
    dosing: dict[str, Any] | None = Field(
        default=None,
        description="compound, molecular_weight_g_mol, stock_mM, working_mM, "
        "vehicle_control, exposure_h (no dmso_fraction_vv)",
    )
    stats: dict[str, Any] | None = Field(
        default=None,
        description="effect_size, std_dev, alpha, power (no n_per_group; it is computed)",
    )
    culture: dict[str, Any] | None = Field(
        default=None,
        description="plate_format, wells, cell_type, seeding_density_cells_cm2, "
        "viability_pct, confluent_density_cells_cm2, doubling_time_h, "
        "culture_duration_h (no seed_per_well / total_seed_count / "
        "medium_volume_per_well_ml / total_medium_ml / expected_confluence_pct; "
        "they are computed)",
    )
    spheroid: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, spheroid_format (96-ula / 384-ula / hanging-drop), "
        "spheroid_count, cells_per_spheroid, cell_diameter_um, doubling_time_h, "
        "culture_duration_h (no spheroid_volume_ul / expected_diameter_um / "
        "cells_total / medium_volume_per_spheroid_ul / total_medium_ml / "
        "expected_cells_after_growth; they are computed)",
    )
    pk: dict[str, Any] | None = Field(
        default=None,
        description="compound, molecular_weight_g_mol, inlet_concentration_uM, "
        "outlet_concentration_uM, flow_rate_uLmin, system_volume_uL, "
        "dose_interval_h (no extraction_ratio / clearance_uLmin / half_life_h / "
        "accumulation_ratio / mass_cleared_ug_h; they are computed)",
    )
    caveats: list[str] = Field(default_factory=list, description="What must be checked in the lab")


# ---------------------------------------------------------------------------
# Derivation (the math)
# ---------------------------------------------------------------------------


def derive_culture(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived CulturePlan field from the raw plate-culture inputs.

    The LLM never writes a derived culture number: seed per well, total seed
    count, medium volumes and predicted confluence all come from
    :mod:`labwright.calc.culture` (with growth math from
    :mod:`labwright.calc.cell`).

    Parameters
    ----------
    raw : dict
        plate_format, wells, cell_type, seeding_density_cells_cm2, and any of
        viability_pct, confluent_density_cells_cm2, doubling_time_h,
        culture_duration_h.

    Returns
    -------
    dict
        ``raw`` plus ``seed_per_well``, ``total_seed_count``,
        ``medium_volume_per_well_ml``, ``total_medium_ml`` and — when the growth
        inputs are present — ``expected_confluence_pct``.
    """
    area = calc_culture.well_surface_area_cm2(raw["plate_format"])
    per_well = calc_culture.cells_per_well(raw["seeding_density_cells_cm2"], raw["plate_format"])
    med = calc_culture.medium_volume_per_well(raw["plate_format"])
    wells = int(raw.get("wells", 1))
    out = dict(raw)
    out["seed_per_well"] = per_well
    out["total_seed_count"] = per_well * wells
    out["medium_volume_per_well_ml"] = med
    out["total_medium_ml"] = med * wells
    if (
        raw.get("doubling_time_h") is not None
        and raw.get("confluent_density_cells_cm2") is not None
        and raw.get("culture_duration_h") is not None
    ):
        final_cells = calc_cell.cell_count_after_time(
            per_well, raw["doubling_time_h"], raw["culture_duration_h"]
        )
        out["expected_confluence_pct"] = calc_culture.cell_count_to_confluence(
            final_cells, raw["confluent_density_cells_cm2"], area
        )
    else:
        out["expected_confluence_pct"] = None
    return out


def derive_spheroid(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived SpheroidPlan field from the raw 3D-culture inputs.

    The LLM never writes a derived spheroid number: volume, expected diameter,
    total cells, medium volume and post-growth cell count all come from
    :mod:`labwright.calc.spheroid` (with growth math from
    :mod:`labwright.calc.cell`).

    Parameters
    ----------
    raw : dict
        cell_type, spheroid_format, spheroid_count, cells_per_spheroid,
        cell_diameter_um, and any of doubling_time_h, culture_duration_h.

    Returns
    -------
    dict
        ``raw`` plus ``spheroid_volume_ul``, ``expected_diameter_um``,
        ``cells_total``, ``medium_volume_per_spheroid_ul``, ``total_medium_ml``
        and — when the growth inputs are present — ``expected_cells_after_growth``.
    """
    n = int(raw.get("spheroid_count", 1))
    per_sph = raw["cells_per_spheroid"]
    cell_d = raw["cell_diameter_um"]
    per_med = calc_spheroid.medium_volume_per_spheroid(raw["spheroid_format"])
    out = dict(raw)
    out["spheroid_count"] = n
    out["spheroid_volume_ul"] = calc_spheroid.spheroid_volume_from_cells(per_sph, cell_d)
    out["expected_diameter_um"] = calc_spheroid.spheroid_diameter_from_cells(per_sph, cell_d)
    out["cells_total"] = calc_spheroid.cells_needed_for_spheroids(n, per_sph)
    out["medium_volume_per_spheroid_ul"] = per_med
    out["total_medium_ml"] = calc_spheroid.total_medium_volume(n, per_med)
    if (
        raw.get("doubling_time_h") is not None
        and raw.get("culture_duration_h") is not None
    ):
        out["expected_cells_after_growth"] = calc_cell.cell_count_after_time(
            per_sph, raw["doubling_time_h"], raw["culture_duration_h"]
        )
    else:
        out["expected_cells_after_growth"] = None
    return out


def derive_pk(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived PkPlan field from the perfused-system raw inputs.

    The LLM never writes a derived PK number: extraction ratio and clearance
    always come from :mod:`labwright.calc.pk`; half-life, accumulation ratio and
    mass cleared are added when their input fields are present (system volume,
    dose interval, molecular weight).

    Parameters
    ----------
    raw : dict
        compound, molecular_weight_g_mol, inlet_concentration_uM,
        outlet_concentration_uM, flow_rate_uLmin, and any of system_volume_uL,
        dose_interval_h.

    Returns
    -------
    dict
        ``raw`` plus ``extraction_ratio``, ``clearance_uLmin`` and — when the
        extra inputs are present — ``half_life_h``, ``accumulation_ratio``,
        ``mass_cleared_ug_h``.
    """
    out = dict(raw)
    out["extraction_ratio"] = calc_pk.extraction_ratio(
        raw["inlet_concentration_uM"], raw["outlet_concentration_uM"]
    )
    out["clearance_uLmin"] = calc_pk.clearance_uLmin(
        raw["inlet_concentration_uM"], raw["outlet_concentration_uM"], raw["flow_rate_uLmin"]
    )
    out["half_life_h"] = None
    if raw.get("system_volume_uL") is not None:
        out["half_life_h"] = calc_pk.half_life_h(raw["system_volume_uL"], out["clearance_uLmin"])
    out["accumulation_ratio"] = None
    if out["half_life_h"] is not None and raw.get("dose_interval_h") is not None:
        out["accumulation_ratio"] = calc_pk.accumulation_ratio(
            out["half_life_h"], raw["dose_interval_h"]
        )
    out["mass_cleared_ug_h"] = None
    if raw.get("molecular_weight_g_mol") is not None:
        out["mass_cleared_ug_h"] = calc_pk.mass_cleared_ug_h(
            out["clearance_uLmin"], raw["inlet_concentration_uM"], raw["molecular_weight_g_mol"]
        )
    return out


def build_design(inp: DesignInput) -> DesignPlan:
    """Derive every computed field from the agent's raw inputs.

    A design carries the flow/cell block when a chip + flow + cell inputs are
    given, and/or the culture block for plate-based culture. Blocks the agent
    did not propose are left ``None`` — a plate-only design is never forced to
    invent a chip.
    """
    # Flow metrics — all recomputed here, never accepted from the LLM.
    derived = None
    if inp.chip is not None and inp.flow is not None:
        derived = DerivedFlowMetrics(
            shear_pa=mf.wall_shear_stress(
                inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um, inp.flow.viscosity_pas
            ),
            reynolds=mf.reynolds_number(
                inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um,
                inp.flow.viscosity_pas, inp.flow.density_kgm3,
            ),
            pressure_drop_pa=mf.pressure_drop(
                inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um,
                inp.chip.length_mm, inp.flow.viscosity_pas,
            ),
            residence_time_s=mf.residence_time(
                inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um, inp.chip.length_mm
            ),
            channel_volume_ul=mf.channel_volume(
                inp.chip.width_um, inp.chip.height_um, inp.chip.length_mm
            ),
            mean_velocity_mms=mf.mean_velocity(
                inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um
            ),
        )

    cells = None
    if inp.cells is not None:
        density = inp.cells.get("seeding_density_cells_cm2")
        area = inp.cells.get("culture_area_cm2")
        if density is None or area is None:
            raise ValueError(
                "cells block requires seeding_density_cells_cm2 and culture_area_cm2"
            )
        cells = CellPlan(
            **inp.cells,
            seed_count=calc_cell.seeding_cell_count(density, area),
        )

    dosing = None
    if inp.dosing is not None:
        d = dict(inp.dosing)
        # DMSO fraction is derived from stock/working, never asserted.
        from labwright.calc import dosing as calc_dosing

        d["dmso_fraction_vv"] = calc_dosing.dmso_fraction(d["stock_mM"], d["working_mM"])
        dosing = DosePlan(**d)

    stats = None
    if inp.stats is not None:
        from labwright.calc import stats as calc_stats

        s = dict(inp.stats)
        s["n_per_group"] = calc_stats.sample_size_per_group(s["effect_size"], s["std_dev"], s["alpha"], s["power"])
        stats = StatsPlan(**s)

    culture = None
    if inp.culture is not None:
        culture = CulturePlan(**derive_culture(inp.culture))

    spheroid = None
    if inp.spheroid is not None:
        spheroid = SpheroidPlan(**derive_spheroid(inp.spheroid))

    pk = None
    if inp.pk is not None:
        pk = PkPlan(**derive_pk(inp.pk))

    plan = DesignPlan(
        goal=inp.goal,
        rationale=inp.rationale,
        chip=inp.chip,
        flow=inp.flow,
        derived=derived,
        cells=cells,
        culture=culture,
        spheroid=spheroid,
        dosing=dosing,
        stats=stats,
        pk=pk,
        caveats=inp.caveats,
    )
    return plan


def _reject_derived_fields(input_dict: dict[str, Any]) -> None:
    """Refuse any derived field the model tried to write.

    ``submit_design`` is the agent's only gate into a verified design. If the
    model smuggles a derived number — a top-level flow metric, or a derived key
    inside a raw block (``culture.seed_per_well``, ``spheroid.expected_diameter_um``,
    ...) — reject the submission explicitly rather than silently overwriting it.
    The model sees a ``validation_error`` tool result and resubmits raw-only, and
    the rejection is a testable property of the gate.
    """
    found: list[str] = []
    for key, value in input_dict.items():
        if key == "derived" or key in _DERIVED_FIELD_NAMES:
            found.append(key)
        elif isinstance(value, dict):
            for nested in value:
                if nested in _DERIVED_FIELD_NAMES:
                    found.append(f"{key}.{nested}")
    if found:
        raise ValueError(
            "derived field(s) are computed by Labwright, not accepted from the model: "
            + ", ".join(sorted(set(found)))
            + ". Remove them and submit raw inputs only."
        )


def submit_design(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate raw input, derive everything, verify, and report.

    This is the final tool the agent calls. The result carries the complete
    design plus the verifier's report, so a design with unresolved errors is
    visible to the agent and can be corrected in a follow-up turn.
    """
    _reject_derived_fields(input_dict)
    inp = DesignInput(**input_dict)
    plan = build_design(inp)
    issues = verify_design(plan)
    from labwright.sop.provenance import provenance_for

    return {
        "design": plan.model_dump(mode="json"),
        "verification": [i.__dict__ for i in issues],
        "verification_summary": format_issues(issues),
        # Full computation path: every derived number's formula, inputs, units,
        # code version and verification status (exportable to an ELN/LIMS).
        "provenance": provenance_for(plan, issues),
        "status": "ok" if not issues else "review_required",
    }


__all__ = ["DesignInput", "build_design", "submit_design"]
