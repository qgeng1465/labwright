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
from labwright.calc import barrier as calc_barrier
from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import microfluidics as mf
from labwright.calc import o2 as calc_o2
from labwright.calc import pk as calc_pk
from labwright.calc import spheroid as calc_spheroid
from labwright.schema.design import (
    BarrierPlan,
    BreathingPlan,
    CellPlan,
    ChipGeometry,
    CulturePlan,
    DerivedFlowMetrics,
    DesignPlan,
    DosePlan,
    FlowParams,
    GradientPlan,
    OxygenPlan,
    PkPlan,
    PulsatilePlan,
    PumplessPlan,
    ScalingPlan,
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
    barrier: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, insert_area_cm2, resistance_total_ohm, "
        "resistance_blank_ohm, probe, donor_conc_um, flux_nmol_min (no "
        "teer_ohm_cm2 / papp_cm_s / clearance_mL_min; they are computed)",
    )
    oxygen: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, target_po2_mmhg, cell_density_cells_ml, "
        "spheroid_diameter_um (no dissolved_o2_mM / penetration_depth_um / "
        "necrotic_fraction / demand_umol_min; they are computed)",
    )
    pumpless: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, tilt_angle_deg, channel_length_mm, width_um, "
        "height_um, rocking_half_period_s, viscosity_pas, density_kgm3, "
        "backward_shear_fraction (no hydrostatic_head_pa / driven_flow_rate_uLmin / "
        "peak_wall_shear_pa / volume_per_half_cycle_ul / oscillatory_shear_index / "
        "cycles_per_hour / shear_ratio_to_target; they are computed)",
    )
    breathing: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, frequency_hz, strain_pct, membrane_span_um, "
        "apical_volume_ul, surface_area_cm2, culture_duration_h, "
        "stretch_seconds, cycle_seconds (no breaths_per_minute / "
        "cyclic_displacement_um / strain_rate_per_s / total_cycles / "
        "stretch_duty_fraction / ali_liquid_film_um; they are computed)",
    )
    pulsatile: dict[str, Any] | None = Field(
        default=None,
        description="cell_type, frequency_hz, channel_height_um, viscosity_pas, "
        "density_kgm3, shear_mean_pa, shear_amplitude_pa, peak_flow_uLmin, "
        "minimum_flow_uLmin, mean_flow_uLmin (no womersley_number / "
        "oscillatory_shear_index / peak_shear_pa / pulsatility_index; they are "
        "computed)",
    )
    scaling: dict[str, Any] | None = Field(
        default=None,
        description="organ, total_cells_chip, cardiac_output_mlmin, body_mass_g, "
        "chip_volume_ul, flow_rate_uLmin, target_transit_s (no "
        "organ_flow_fraction / organ_flow_rate_mlmin / cells_in_organ / "
        "allometric_scale / transit_time_s / residence_time_match_error_s; they "
        "are computed)",
    )
    gradient: dict[str, Any] | None = Field(
        default=None,
        description="chemoattractant, source_conc_um, sink_conc_um, distance_um, "
        "experiment_hours, diffusivity_m2s (no steepness_um_per_mm / "
        "midpoint_conc_um / relaxation_time_s / flux_mol_m2s; they are computed)",
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
    # A model-reported molecular weight must agree with the named compound when
    # the compound is one we have a pinned value for (see calc.pk.COMPOUND_MW).
    # This closes the raw/derived gate hole where "warfarin, MW 464" could pass.
    if raw.get("compound") is not None and raw.get("molecular_weight_g_mol") is not None:
        calc_pk.check_compound_mw(raw["compound"], raw["molecular_weight_g_mol"])
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


def derive_barrier(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived BarrierPlan field from the measured resistances/flux.

    The LLM never writes a derived barrier number: TEER always comes from
    :mod:`labwright.calc.barrier`; Papp and the permeability-surface-area
    product (clearance) are added when the probe flux and donor concentration
    are present.
    """
    out = dict(raw)
    out["teer_ohm_cm2"] = calc_barrier.teer_ohm_cm2(
        raw["resistance_total_ohm"], raw["resistance_blank_ohm"], raw["insert_area_cm2"]
    )
    out["papp_cm_s"] = None
    out["clearance_mL_min"] = None
    if raw.get("flux_nmol_min") is not None and raw.get("donor_conc_um") is not None:
        out["papp_cm_s"] = calc_barrier.papp_cm_s(
            raw["flux_nmol_min"], raw["insert_area_cm2"], raw["donor_conc_um"]
        )
        out["clearance_mL_min"] = calc_barrier.clearance_mL_min(
            out["papp_cm_s"], raw["insert_area_cm2"]
        )
    return out


def derive_oxygen(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived OxygenPlan field from the target pO2 and consumption.

    The LLM never writes a derived oxygen number: dissolved concentration comes
    from Henry's law, and — when the cell density is given — the Krogh
    penetration depth and (for spheroids) necrotic-core fraction come from
    :mod:`labwright.calc.o2` using the cell-type OCR from
    :mod:`labwright.physiology` (never proposed by the LLM).
    """
    from labwright.physiology import lookup_cell

    out = dict(raw)
    out["dissolved_o2_mM"] = calc_o2.o2_conc_mm_from_po2(raw["target_po2_mmhg"])

    prof = lookup_cell(raw.get("cell_type"))
    ocr = prof.o2_consumption_nmol_min_1e6 if prof else None

    pen = None
    if raw.get("cell_density_cells_ml") is not None and ocr is not None:
        ocr_mid = (ocr[0] + ocr[1]) / 2.0
        fmol_s = calc_o2.nmol_min_per_1e6_to_fmol_s(ocr_mid)
        q = calc_o2.volumetric_o2_consumption(fmol_s, raw["cell_density_cells_ml"])
        pen = calc_o2.o2_penetration_depth_um(q)
    out["penetration_depth_um"] = pen

    out["necrotic_fraction"] = None
    if pen is not None and raw.get("spheroid_diameter_um") is not None:
        out["necrotic_fraction"] = calc_o2.spheroid_necrotic_fraction(
            raw["spheroid_diameter_um"], pen
        )

    out["demand_umol_min"] = None
    if ocr is not None:
        out["demand_umol_min"] = calc_o2.o2_demand_umol_min(
            1e6, calc_o2.nmol_min_per_1e6_to_fmol_s((ocr[0] + ocr[1]) / 2.0)
        )
    return out


def derive_pumpless(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived PumplessPlan field from the rocking-platform inputs.

    The LLM never writes a derived pumpless number: hydrostatic head, driven
    flow rate, peak wall shear, half-cycle volume, OSI and cycles-per-hour all
    come from :mod:`labwright.calc.pumpless`. The physiological shear target is
    the cell-type registry's ``shear_range_pa`` (falling back to the liver
    sinusoidal range cited for gravity-driven chips), never invented here.
    """
    from labwright.calc import pumpless as cp
    from labwright.physiology import lookup_cell

    out = dict(raw)
    rho = raw.get("density_kgm3", cp.CULTURE_MEDIUM_DENSITY_KGM3)
    mu = raw.get("viscosity_pas", cp.CULTURE_MEDIUM_VISCOSITY_PAS)
    head = cp.hydrostatic_pressure_pa(
        rho, raw["tilt_angle_deg"], raw["channel_length_mm"]
    )
    out["hydrostatic_head_pa"] = head
    q = cp.flow_rate_from_pressure_head(
        head, raw["width_um"], raw["height_um"], raw["channel_length_mm"], mu
    )
    out["driven_flow_rate_uLmin"] = q
    tau = cp.peak_wall_shear_from_head(
        head, raw["width_um"], raw["height_um"], raw["channel_length_mm"]
    )
    out["peak_wall_shear_pa"] = tau
    out["volume_per_half_cycle_ul"] = cp.rocking_volume_per_half_cycle_ul(
        q, raw["rocking_half_period_s"]
    )
    bwd_frac = raw.get("backward_shear_fraction", 1.0)
    out["oscillatory_shear_index"] = cp.oscillatory_shear_index(
        tau, tau * bwd_frac
    )
    out["cycles_per_hour"] = cp.cycles_per_hour(raw["rocking_half_period_s"])

    prof = lookup_cell(raw.get("cell_type"))
    if prof is not None and prof.shear_range_pa is not None:
        lo, hi = prof.shear_range_pa
    else:
        lo, hi = cp.LIVER_SINUSOID_WSS_MIN_PA, cp.LIVER_SINUSOID_WSS_MAX_PA
    target = (lo + hi) / 2.0
    out["shear_ratio_to_target"] = round(tau / target, 4)
    return out


def derive_breathing(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived BreathingPlan field from the lung-chip inputs.

    The LLM never writes a derived breathing number: breaths/min, membrane
    stroke, strain rate and the conditional total cycles / duty fraction / ALI
    film all come from :mod:`labwright.calc.breathing`.
    """
    from labwright.calc import breathing as cb

    out = dict(raw)
    out["breaths_per_minute"] = cb.breaths_per_minute(raw["frequency_hz"])
    out["cyclic_displacement_um"] = cb.cyclic_displacement_um(
        raw["strain_pct"], raw.get("membrane_span_um") or cb.DEFAULT_MEMBRANE_SPAN_UM
    )
    out["strain_rate_per_s"] = cb.strain_rate_per_s(raw["strain_pct"], raw["frequency_hz"])
    out["total_cycles"] = None
    if raw.get("culture_duration_h") is not None:
        out["total_cycles"] = cb.total_cycles(raw["culture_duration_h"], raw["frequency_hz"])
    out["stretch_duty_fraction"] = None
    if raw.get("stretch_seconds") is not None and raw.get("cycle_seconds") is not None:
        out["stretch_duty_fraction"] = cb.stretch_duty_fraction(
            raw["stretch_seconds"], raw["cycle_seconds"]
        )
    out["ali_liquid_film_um"] = None
    if raw.get("apical_volume_ul") is not None and raw.get("surface_area_cm2") is not None:
        out["ali_liquid_film_um"] = cb.ali_liquid_film_um(
            raw["apical_volume_ul"], raw["surface_area_cm2"]
        )
    return out


def derive_pulsatile(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived PulsatilePlan field from the cardiac-waveform inputs.

    The LLM never writes a derived pulsatile number: Womersley number, OSI and
    peak shear always come from :mod:`labwright.calc.pulsatile`; the Gosling
    pulsatility index is added when the flow-waveform inputs are present.
    """
    from labwright.calc import pulsatile as cp

    out = dict(raw)
    out["womersley_number"] = cp.womersley_number(
        raw["frequency_hz"], raw["channel_height_um"],
        raw.get("viscosity_pas", cp.MEDIUM_VISCOSITY_PAS),
        raw.get("density_kgm3", cp.MEDIUM_DENSITY_KGM3),
    )
    out["oscillatory_shear_index"] = cp.oscillatory_shear_index_from_sinusoid(
        raw["shear_mean_pa"], raw["shear_amplitude_pa"]
    )
    out["peak_shear_pa"] = cp.peak_shear_of_sinusoid(raw["shear_mean_pa"], raw["shear_amplitude_pa"])
    out["pulsatility_index"] = None
    if (
        raw.get("peak_flow_uLmin") is not None
        and raw.get("minimum_flow_uLmin") is not None
        and raw.get("mean_flow_uLmin") is not None
    ):
        out["pulsatility_index"] = cp.pulsatility_index(
            raw["peak_flow_uLmin"], raw["minimum_flow_uLmin"], raw["mean_flow_uLmin"]
        )
    return out


def derive_scaling(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived ScalingPlan field from the body-on-chip inputs.

    The LLM never writes a derived scaling number: organ flow fraction, organ
    perfusion flow, mass-proportional cell number and the Kleiber allometric
    factor all come from :mod:`labwright.calc.scaling` using the physiology
    tables (organ mass / flow fractions are pinned, never proposed). Transit
    and residence-match numbers are added when the compartment volume/flow and
    target transit are present.
    """
    from labwright.calc import scaling as cs

    out = dict(raw)
    organ = raw["organ"]
    out["organ_flow_fraction"] = cs.organ_flow_fraction(organ)
    out["organ_flow_rate_mlmin"] = cs.organ_flow_rate_mlmin(
        organ, raw.get("cardiac_output_mlmin", cs.CARDIAC_OUTPUT_MLMIN)
    )
    organ_mass = cs.ORGAN_MASS_G[organ]
    body_mass = raw.get("body_mass_g", cs.BODY_MASS_G)
    out["cells_in_organ"] = cs.scale_cell_number(
        organ_mass, body_mass, raw["total_cells_chip"]
    )
    out["allometric_scale"] = cs.allometric_metabolic_scale(organ_mass, body_mass)
    out["transit_time_s"] = None
    out["residence_time_match_error_s"] = None
    if raw.get("chip_volume_ul") is not None and raw.get("flow_rate_uLmin") is not None:
        out["transit_time_s"] = cs.transit_time_s(raw["chip_volume_ul"], raw["flow_rate_uLmin"])
        if raw.get("target_transit_s") is not None:
            out["residence_time_match_error_s"] = cs.residence_time_match_error_s(
                raw["chip_volume_ul"], raw["flow_rate_uLmin"], raw["target_transit_s"]
            )
    return out


def derive_gradient(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill every derived GradientPlan field from the source-sink inputs.

    The LLM never writes a derived gradient number: steepness, mid-gap
    concentration, relaxation time and steady-state flux all come from
    :mod:`labwright.calc.gradient`.
    """
    from labwright.calc import gradient as cg

    out = dict(raw)
    out["steepness_um_per_mm"] = cg.linear_gradient_steepness_um_per_mm(
        raw["source_conc_um"], raw["sink_conc_um"], raw["distance_um"]
    )
    out["midpoint_conc_um"] = cg.steady_state_profile_conc_um(
        raw["source_conc_um"], raw["sink_conc_um"], raw["distance_um"], raw["distance_um"] / 2.0
    )
    diff = raw.get("diffusivity_m2s") or cg.SMALL_MOLECULE_DIFFUSIVITY_M2S
    out["relaxation_time_s"] = cg.diffusive_relaxation_time_s(raw["distance_um"], diff)
    out["flux_mol_m2s"] = cg.diffusive_flux_mol_m2s(
        raw["source_conc_um"], raw["sink_conc_um"], raw["distance_um"], diff
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

    barrier = None
    if inp.barrier is not None:
        barrier = BarrierPlan(**derive_barrier(inp.barrier))

    oxygen = None
    if inp.oxygen is not None:
        oxygen = OxygenPlan(**derive_oxygen(inp.oxygen))

    pumpless = None
    if inp.pumpless is not None:
        pumpless = PumplessPlan(**derive_pumpless(inp.pumpless))

    breathing = None
    if inp.breathing is not None:
        breathing = BreathingPlan(**derive_breathing(inp.breathing))

    pulsatile = None
    if inp.pulsatile is not None:
        pulsatile = PulsatilePlan(**derive_pulsatile(inp.pulsatile))

    scaling = None
    if inp.scaling is not None:
        scaling = ScalingPlan(**derive_scaling(inp.scaling))

    gradient = None
    if inp.gradient is not None:
        gradient = GradientPlan(**derive_gradient(inp.gradient))

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
        barrier=barrier,
        oxygen=oxygen,
        pumpless=pumpless,
        breathing=breathing,
        pulsatile=pulsatile,
        scaling=scaling,
        gradient=gradient,
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


def submit_design(input_dict: dict[str, Any], verify: bool = True) -> dict[str, Any]:
    """Validate raw input, derive everything, and (optionally) verify.

    This is the final tool the agent calls. With ``verify=True`` (the default
    and the only mode the shipped agent uses), the result carries the complete
    design plus the verifier's report, so a design with unresolved errors is
    visible to the agent and can be corrected in a follow-up turn.

    ``verify=False`` is the *no-gate ablation* used only by the benchmark's
    ``tool_no_gate`` system: the calculators still derive every number, but the
    verifier never runs and the submission is always accepted. Post-hoc the
    benchmark runs the identical ``_score_design`` on the resulting plan, so a
    design the verifier would have rejected still reads as a failed submission.
    """
    _reject_derived_fields(input_dict)
    inp = DesignInput(**input_dict)
    plan = build_design(inp)
    from labwright.sop.provenance import provenance_for

    if verify:
        issues = verify_design(plan)
        return {
            "design": plan.model_dump(mode="json"),
            "verification": [i.__dict__ for i in issues],
            "verification_summary": format_issues(issues),
            # Full computation path: every derived number's formula, inputs,
            # units, code version and verification status (ELN/LIMS-exportable).
            "provenance": provenance_for(plan, issues),
            "status": "ok" if not issues else "review_required",
        }
    return {
        "design": plan.model_dump(mode="json"),
        "verification": [],
        "verification_summary": "",
        "provenance": provenance_for(plan, []),
        "status": "ok",  # no-gate ablation: the verifier is switched off
    }


__all__ = ["DesignInput", "build_design", "submit_design"]
