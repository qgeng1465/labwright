"""Physiological range checks — the second verification layer.

Arithmetic cross-checks (:mod:`labwright.verify.checker`) prove a number
*follows from the inputs*; range checks prove the number is *physiologically
and physically plausible*. A design that computes cleanly but proposes 500 Pa
of wall shear, a Reynolds number of 5000, or a seeding density of 1e9 cells/cm²
is exactly as unusable as one with an arithmetic error — it just fails
differently.

Two bands per field:

- **soft band** — the physiological range (plus a safety margin). Falling
  outside it is a *warning*: the design is plausible only under an unusual
  interpretation, and the human must confirm it. Never silent.
- **hard band** — the physical boundary (negative, non-finite, or
  order-of-magnitude absurd). Falling outside it is an *error*: the design
  cannot be built or cultured as stated.

Bands are deliberately wide. The point is to catch order-of-magnitude and
unit-scale mistakes, not to second-guess biology the model is better placed to
judge. Unit-scale mistakes (dyn/cm² vs Pa, etc.) are the job of
:mod:`labwright.verify.units`.
"""

from __future__ import annotations

from labwright.blocks import ALL_SANITY_BANDS, Band
from labwright.schema.design import DesignPlan
from labwright.verify.checker import Issue

#: Keyed by the verifier's issue field names (same keys as the checker uses).
#: The bands themselves are declared once in :mod:`labwright.blocks` — one
#: ``Block`` per design domain carries its own bands — and merged here. Adding
#: a domain's range check means editing that domain's ``Block``, not this table.
SANITY_BANDS: dict[str, Band] = ALL_SANITY_BANDS


def check_sanity(plan: DesignPlan, issues: list[Issue]) -> None:
    """Append a range issue for every out-of-band raw/derived value.

    Soft violations are *warnings* (plausible only under an unusual
    interpretation — the human must confirm); hard violations are *errors*
    (the design cannot be built as stated). Neither is ever silently passed.
    """
    values: dict[str, float] = {}
    if plan.derived is not None:
        values.update({
            "derived.shear_pa": plan.derived.shear_pa,
            "derived.reynolds": plan.derived.reynolds,
            "derived.pressure_drop_pa": plan.derived.pressure_drop_pa,
            "derived.residence_time_s": plan.derived.residence_time_s,
            "derived.channel_volume_ul": plan.derived.channel_volume_ul,
            "derived.mean_velocity_mms": plan.derived.mean_velocity_mms,
        })
    if plan.cells is not None:
        values.update({
            "cells.seed_count": plan.cells.seed_count,
            "cells.seeding_density_cells_cm2": plan.cells.seeding_density_cells_cm2,
            "cells.culture_area_cm2": plan.cells.culture_area_cm2,
        })
    if plan.culture is not None:
        c = plan.culture
        values.update({
            "culture.seed_per_well": c.seed_per_well,
            "culture.total_seed_count": c.total_seed_count,
            "culture.seeding_density_cells_cm2": c.seeding_density_cells_cm2,
            "culture.medium_volume_per_well_ml": c.medium_volume_per_well_ml,
            "culture.total_medium_ml": c.total_medium_ml,
            "culture.doubling_time_h": c.doubling_time_h,
            "culture.culture_duration_h": c.culture_duration_h,
        })
        if c.expected_confluence_pct is not None:
            values["culture.expected_confluence_pct"] = c.expected_confluence_pct
    if plan.spheroid is not None:
        s = plan.spheroid
        values.update({
            "spheroid.cells_per_spheroid": s.cells_per_spheroid,
            "spheroid.expected_diameter_um": s.expected_diameter_um,
            "spheroid.spheroid_volume_ul": s.spheroid_volume_ul,
            "spheroid.cell_diameter_um": s.cell_diameter_um,
            "spheroid.medium_volume_per_spheroid_ul": s.medium_volume_per_spheroid_ul,
            "spheroid.total_medium_ml": s.total_medium_ml,
            "spheroid.cells_total": s.cells_total,
            "spheroid.spheroid_count": float(s.spheroid_count),
            "spheroid.doubling_time_h": s.doubling_time_h,
            "spheroid.culture_duration_h": s.culture_duration_h,
        })
        if s.expected_cells_after_growth is not None:
            values["spheroid.expected_cells_after_growth"] = s.expected_cells_after_growth
    if plan.dosing is not None:
        values.update({
            "dosing.stock_mM": plan.dosing.stock_mM,
            "dosing.working_mM": plan.dosing.working_mM,
            "dosing.dmso_fraction_vv": plan.dosing.dmso_fraction_vv,
        })
    if plan.stats is not None:
        values["stats.n_per_group"] = float(plan.stats.n_per_group)
    if plan.pk is not None:
        p = plan.pk
        values.update({
            "pk.extraction_ratio": p.extraction_ratio,
            "pk.clearance_uLmin": p.clearance_uLmin,
            "pk.inlet_concentration_uM": p.inlet_concentration_uM,
            "pk.outlet_concentration_uM": p.outlet_concentration_uM,
            "pk.flow_rate_uLmin": p.flow_rate_uLmin,
            "pk.molecular_weight_g_mol": p.molecular_weight_g_mol,
        })
        if p.half_life_h is not None:
            values["pk.half_life_h"] = p.half_life_h
        if p.accumulation_ratio is not None:
            values["pk.accumulation_ratio"] = p.accumulation_ratio
        if p.mass_cleared_ug_h is not None:
            values["pk.mass_cleared_ug_h"] = p.mass_cleared_ug_h
        if p.system_volume_uL is not None:
            values["pk.system_volume_uL"] = p.system_volume_uL
        if p.dose_interval_h is not None:
            values["pk.dose_interval_h"] = p.dose_interval_h
    if plan.barrier is not None:
        b = plan.barrier
        values.update({
            "barrier.teer_ohm_cm2": b.teer_ohm_cm2,
            "barrier.insert_area_cm2": b.insert_area_cm2,
            "barrier.resistance_total_ohm": b.resistance_total_ohm,
            "barrier.resistance_blank_ohm": b.resistance_blank_ohm,
            "barrier.donor_conc_um": b.donor_conc_um,
        })
        if b.papp_cm_s is not None:
            values["barrier.papp_cm_s"] = b.papp_cm_s
        if b.clearance_mL_min is not None:
            values["barrier.clearance_mL_min"] = b.clearance_mL_min
    if plan.oxygen is not None:
        o = plan.oxygen
        values.update({
            "oxygen.target_po2_mmhg": o.target_po2_mmhg,
            "oxygen.dissolved_o2_mM": o.dissolved_o2_mM,
        })
        if o.penetration_depth_um is not None:
            values["oxygen.penetration_depth_um"] = o.penetration_depth_um
        if o.necrotic_fraction is not None:
            values["oxygen.necrotic_fraction"] = o.necrotic_fraction
        if o.demand_umol_min is not None:
            values["oxygen.demand_umol_min"] = o.demand_umol_min
    if plan.pumpless is not None:
        p = plan.pumpless
        values.update({
            "pumpless.tilt_angle_deg": p.tilt_angle_deg,
            "pumpless.channel_length_mm": p.channel_length_mm,
            "pumpless.rocking_half_period_s": p.rocking_half_period_s,
            "pumpless.hydrostatic_head_pa": p.hydrostatic_head_pa,
            "pumpless.driven_flow_rate_uLmin": p.driven_flow_rate_uLmin,
            "pumpless.peak_wall_shear_pa": p.peak_wall_shear_pa,
            "pumpless.volume_per_half_cycle_ul": p.volume_per_half_cycle_ul,
            "pumpless.oscillatory_shear_index": p.oscillatory_shear_index,
            "pumpless.cycles_per_hour": p.cycles_per_hour,
        })
        if p.shear_ratio_to_target is not None:
            values["pumpless.shear_ratio_to_target"] = p.shear_ratio_to_target
    if plan.breathing is not None:
        b = plan.breathing
        values.update({
            "breathing.frequency_hz": b.frequency_hz,
            "breathing.strain_pct": b.strain_pct,
            "breathing.membrane_span_um": b.membrane_span_um,
            "breathing.breaths_per_minute": b.breaths_per_minute,
            "breathing.cyclic_displacement_um": b.cyclic_displacement_um,
            "breathing.strain_rate_per_s": b.strain_rate_per_s,
        })
        if b.total_cycles is not None:
            values["breathing.total_cycles"] = b.total_cycles
        if b.stretch_duty_fraction is not None:
            values["breathing.stretch_duty_fraction"] = b.stretch_duty_fraction
        if b.ali_liquid_film_um is not None:
            values["breathing.ali_liquid_film_um"] = b.ali_liquid_film_um
    if plan.pulsatile is not None:
        p = plan.pulsatile
        values.update({
            "pulsatile.frequency_hz": p.frequency_hz,
            "pulsatile.channel_height_um": p.channel_height_um,
            "pulsatile.womersley_number": p.womersley_number,
            "pulsatile.oscillatory_shear_index": p.oscillatory_shear_index,
            "pulsatile.peak_shear_pa": p.peak_shear_pa,
            "pulsatile.shear_mean_pa": p.shear_mean_pa,
            "pulsatile.shear_amplitude_pa": p.shear_amplitude_pa,
        })
        if p.pulsatility_index is not None:
            values["pulsatile.pulsatility_index"] = p.pulsatility_index
    if plan.scaling is not None:
        s = plan.scaling
        values.update({
            "scaling.organ_flow_fraction": s.organ_flow_fraction,
            "scaling.organ_flow_rate_mlmin": s.organ_flow_rate_mlmin,
            "scaling.cells_in_organ": s.cells_in_organ,
            "scaling.allometric_scale": s.allometric_scale,
            "scaling.total_cells_chip": s.total_cells_chip,
            "scaling.cardiac_output_mlmin": s.cardiac_output_mlmin,
        })
        if s.transit_time_s is not None:
            values["scaling.transit_time_s"] = s.transit_time_s
        if s.residence_time_match_error_s is not None:
            values["scaling.residence_time_match_error_s"] = s.residence_time_match_error_s
    if plan.gradient is not None:
        g = plan.gradient
        values.update({
            "gradient.source_conc_um": g.source_conc_um,
            "gradient.sink_conc_um": g.sink_conc_um,
            "gradient.distance_um": g.distance_um,
            "gradient.steepness_um_per_mm": g.steepness_um_per_mm,
            "gradient.midpoint_conc_um": g.midpoint_conc_um,
            "gradient.relaxation_time_s": g.relaxation_time_s,
            "gradient.flux_mol_m2s": g.flux_mol_m2s,
            "gradient.experiment_hours": g.experiment_hours,
        })

    for field, value in values.items():
        if value is None:
            continue
        band = SANITY_BANDS.get(field)
        if band is None:
            continue
        _check_value(issues, field, value, band)

    # Relational checks the range bands cannot express (aspect ratio, units).
    check_channel_geometry(plan, issues)


def _check_value(issues: list[Issue], field: str, value: float, band: Band) -> None:
    soft_lo = band.soft_min is not None and value < band.soft_min
    soft_hi = band.soft_max is not None and value > band.soft_max
    hard_lo = band.hard_min is not None and value < band.hard_min
    hard_hi = band.hard_max is not None and value > band.hard_max
    if hard_lo or hard_hi:
        issues.append(Issue(
            level="error",
            field=field,
            message=(
                f"{field} = {value:.6g} {band.units} is outside any physically "
                f"plausible {band.description} (hard bound "
                f"{_fmt_lo(band.hard_min)}–{_fmt_hi(band.hard_max)} {band.units})"
            ),
        ))
    elif soft_lo or soft_hi:
        issues.append(Issue(
            level="warning",
            field=field,
            message=(
                f"{field} = {value:.6g} {band.units} is outside the physiological "
                f"{band.description} range "
                f"{_fmt_lo(band.soft_min)}–{_fmt_hi(band.soft_max)} {band.units} "
                f"— confirm this is intentional"
            ),
        ))


def _fmt_lo(x: float | None) -> str:
    return "−∞" if x is None else f"{x:.6g}"


def _fmt_hi(x: float | None) -> str:
    return "∞" if x is None else f"{x:.6g}"


def check_channel_geometry(plan: DesignPlan, issues: list[Issue]) -> None:
    """Relational geometry checks the range bands cannot express.

    Two failure modes a per-field band would let through:

    1. **Aspect ratio** — the parallel-plate wall-shear formula ``6μQ/(wh²)``
       is valid only when *height is the narrow dimension*. A chip reported
       with height ≥ width silently produces wrong shear/pressure numbers; an
       aspect ratio close to square pushes the approximation error past ~10 %.
    2. **Mixed length units** — ``length_mm`` is in millimetres while
       ``width_um``/``height_um`` are in micrometres. A length that is shorter
       than the channel's own cross-section means it was entered in µm (or the
       geometry is not a channel at all) — the error message states the
       conversion instead of letting a 1000× volume slip through as a value.
    """
    chip = plan.chip
    if chip is None:
        return
    w, h = chip.width_um, chip.height_um
    if h >= w:
        issues.append(Issue(
            level="error",
            field="chip.height_um",
            message=(
                f"channel height {h:g} µm is not the narrow dimension "
                f"(width {w:g} µm) — the parallel-plate wall-shear formula "
                "requires height < width; swap the two or the derived shear/"
                "pressure numbers are wrong"
            ),
        ))
    elif w / h < 2:
        issues.append(Issue(
            level="warning",
            field="chip.height_um",
            message=(
                f"channel aspect ratio w/h = {w / h:.2f} < 2 — the parallel-plate "
                "wall-shear approximation error grows past ~10 % as the "
                "cross-section approaches square; confirm the wide-channel "
                "solution is appropriate"
            ),
        ))
    length_um = chip.length_mm * 1000.0
    if length_um < w or length_um < h:
        issues.append(Issue(
            level="error",
            field="chip.length_mm",
            message=(
                f"channel length {chip.length_mm:g} mm (= {length_um:g} µm) is "
                "shorter than the channel's own width/height — length_mm is in "
                "millimetres (a 20 mm channel is 20,000 µm); if you meant µm, "
                "divide by 1000"
            ),
        ))


__all__ = ["Band", "SANITY_BANDS", "check_sanity", "check_channel_geometry"]
