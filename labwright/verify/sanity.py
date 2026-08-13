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

from dataclasses import dataclass

from labwright.schema.design import DesignPlan
from labwright.verify.checker import Issue


@dataclass(frozen=True)
class Band:
    """A physiological range with a hard physical boundary.

    ``None`` bounds are unbounded on that side. ``soft_min/soft_max`` are the
    warning band; ``hard_min/hard_max`` the error band. hard bounds are always
    wider than (or equal to) the soft bounds.
    """

    soft_min: float | None
    soft_max: float | None
    hard_min: float | None
    hard_max: float | None
    description: str
    units: str


#: Keyed by the verifier's issue field names (same keys as the checker uses).
SANITY_BANDS: dict[str, Band] = {
    "derived.shear_pa": Band(0.001, 10, 1e-4, 50,
        "wall shear stress in organ-on-chip culture", "Pa"),
    "derived.reynolds": Band(0.001, 200, 0.0, 2300,
        "Reynolds number (flow must be laminar)", "dimensionless"),
    "derived.pressure_drop_pa": Band(1.0, 1e5, 0.0, 1e6,
        "laminar pressure drop along a microchannel", "Pa"),
    "derived.residence_time_s": Band(0.1, 1e4, 1e-4, 1e5,
        "fluid residence time in a channel", "s"),
    "derived.channel_volume_ul": Band(0.01, 100, 1e-6, 1e4,
        "per-channel culture volume", "uL"),
    "derived.mean_velocity_mms": Band(0.01, 1e4, 1e-4, 1e5,
        "mean flow velocity in a channel", "mm/s"),
    "cells.seed_count": Band(1e2, 1e9, 1.0, 1e10,
        "cells seeded onto a culture area", "cells"),
    "cells.seeding_density_cells_cm2": Band(1e3, 1e7, 1.0, 1e9,
        "cell seeding density", "cells/cm^2"),
    "cells.culture_area_cm2": Band(1e-4, 100, 1e-6, 1e3,
        "cell culture area", "cm^2"),
    "culture.seed_per_well": Band(1e2, 1e9, 1.0, 1e10,
        "cells seeded per well", "cells"),
    "culture.total_seed_count": Band(1e2, 1e10, 1.0, 1e11,
        "total cells seeded across wells", "cells"),
    "culture.seeding_density_cells_cm2": Band(1e3, 1e7, 1.0, 1e9,
        "cell seeding density", "cells/cm^2"),
    "culture.medium_volume_per_well_ml": Band(0.01, 5.0, 1e-4, 100,
        "working medium volume per well", "mL"),
    "culture.total_medium_ml": Band(0.01, 1e3, 1e-4, 1e4,
        "total medium volume across wells", "mL"),
    "culture.expected_confluence_pct": Band(0.0, 100, 0.0, 1000,
        "predicted confluence at harvest (may exceed 100 % for over-confluent cultures)", "%"),
    "culture.doubling_time_h": Band(10, 200, 0.1, 1000,
        "population doubling time", "h"),
    "culture.culture_duration_h": Band(0.0, 2000, 0.0, 1e5,
        "culture duration", "h"),
    "dosing.stock_mM": Band(0.1, 1e4, 1e-4, 1e6,
        "compound stock concentration", "mM"),
    "dosing.working_mM": Band(1e-3, 100, 1e-6, 1e4,
        "compound working concentration", "mM"),
    "dosing.dmso_fraction_vv": Band(0.0, 0.005, 0.0, 0.14,
        "DMSO volume fraction in medium", "v/v"),
    "stats.n_per_group": Band(3, 1000, 2, 1e6,
        "biological replicates per group", "n"),
}


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


__all__ = ["Band", "SANITY_BANDS", "check_sanity"]
