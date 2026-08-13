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
