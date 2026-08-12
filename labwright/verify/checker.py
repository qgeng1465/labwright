"""Deterministic cross-check of a :class:`~labwright.schema.design.DesignPlan`.

The agent is free to propose geometries, flows, doses and cell plans, but it
is *not* free to invent derived numbers. The checker re-runs the governing
equations on the agent's own inputs and reports every mismatch. Labwright's
public interface refuses to display a design with unresolved errors, which is
the practical difference between "an LLM that guesses numbers" and "an LLM
that quotes the calculators".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import dosing as calc_dosing
from labwright.calc import microfluidics as mf
from labwright.schema.design import DesignPlan

# Relative tolerance for float re-comparison (below experimental precision)
_TOL = 1e-6


@dataclass
class Issue:
    """A single verification finding."""

    level: str  # "error" | "warning"
    field: str
    message: str
    expected: float | None = None
    found: float | None = None


def _close(a: float, b: float, tol: float = _TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def check_flow(plan: DesignPlan, issues: list[Issue]) -> None:
    """Recompute every derived flow metric from the raw inputs."""
    w, h, L = plan.chip.width_um, plan.chip.height_um, plan.chip.length_mm
    q, mu = plan.flow.flow_rate_uLmin, plan.flow.viscosity_pas
    d = plan.derived

    pairs = [
        ("shear_pa", mf.wall_shear_stress(q, w, h, mu), d.shear_pa),
        ("reynolds", mf.reynolds_number(q, w, h, mu, plan.flow.density_kgm3), d.reynolds),
        ("pressure_drop_pa", mf.pressure_drop(q, w, h, L, mu), d.pressure_drop_pa),
        ("residence_time_s", mf.residence_time(q, w, h, L), d.residence_time_s),
        ("channel_volume_ul", mf.channel_volume(w, h, L), d.channel_volume_ul),
        ("mean_velocity_mms", mf.mean_velocity(q, w, h), d.mean_velocity_mms),
    ]
    for field, expected, found in pairs:
        if not _close(expected, found):
            issues.append(
                Issue(
                    level="error",
                    field=f"derived.{field}",
                    message=f"{field} does not match the calculators (recomputed from inputs)",
                    expected=expected,
                    found=found,
                )
            )


def check_seeding(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check cell seeding against the culture area and density."""
    expected = calc_cell.seeding_cell_count(
        plan.cells.seeding_density_cells_cm2, plan.cells.culture_area_cm2
    )
    if not _close(expected, plan.cells.seed_count):
        issues.append(
            Issue(
                level="error",
                field="cells.seed_count",
                message="seed_count does not equal density × culture area",
                expected=expected,
                found=plan.cells.seed_count,
            )
        )
    # culture area should match the chip geometry
    area_from_geom = calc_cell.culture_area(plan.chip.width_um, plan.chip.length_mm)
    if not _close(area_from_geom, plan.cells.culture_area_cm2):
        issues.append(
            Issue(
                level="warning",
                field="cells.culture_area_cm2",
                message="culture_area_cm2 disagrees with width × length of the chip",
                expected=area_from_geom,
                found=plan.cells.culture_area_cm2,
            )
        )


def check_dosing(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the dose plan when present."""
    if plan.dosing is None:
        return
    # DMSO fraction must match stock/working concentrations
    expected_dmso = calc_dosing.dmso_fraction(plan.dosing.stock_mM, plan.dosing.working_mM)
    if not _close(expected_dmso, plan.dosing.dmso_fraction_vv):
        issues.append(
            Issue(
                level="error",
                field="dosing.dmso_fraction_vv",
                message="DMSO fraction does not equal working_mM / stock_mM",
                expected=expected_dmso,
                found=plan.dosing.dmso_fraction_vv,
            )
        )
    # solvent-toxicity warning
    if plan.dosing.dmso_fraction_vv > 0.005:
        issues.append(
            Issue(
                level="warning",
                field="dosing.dmso_fraction_vv",
                message=f"DMSO {plan.dosing.dmso_fraction_vv*100:.2f}% v/v exceeds the usual "
                "0.1-0.5% safe window — consider a more concentrated stock or lower dose",
            )
        )


def check_culture(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the plate-culture plan when present.

    Re-runs :mod:`labwright.calc.culture` on the raw inputs and verifies every
    derived field; warns on over-confluence, low viability and missing growth
    inputs that make the confluence prediction impossible.
    """
    c = plan.culture
    if c is None:
        return
    area = calc_culture.well_surface_area_cm2(c.plate_format)

    expected_seed = calc_culture.cells_per_well(c.seeding_density_cells_cm2, c.plate_format)
    if not _close(expected_seed, c.seed_per_well):
        issues.append(
            Issue(
                level="error",
                field="culture.seed_per_well",
                message="seed_per_well does not equal density × well surface area",
                expected=expected_seed,
                found=c.seed_per_well,
            )
        )
    expected_total = c.seed_per_well * c.wells
    if not _close(expected_total, c.total_seed_count):
        issues.append(
            Issue(
                level="error",
                field="culture.total_seed_count",
                message="total_seed_count does not equal seed_per_well × wells",
                expected=expected_total,
                found=c.total_seed_count,
            )
        )
    expected_med = calc_culture.medium_volume_per_well(c.plate_format)
    if not _close(expected_med, c.medium_volume_per_well_ml):
        issues.append(
            Issue(
                level="error",
                field="culture.medium_volume_per_well_ml",
                message="medium_volume_per_well_ml does not equal the standard working volume",
                expected=expected_med,
                found=c.medium_volume_per_well_ml,
            )
        )
    expected_total_med = c.medium_volume_per_well_ml * c.wells
    if not _close(expected_total_med, c.total_medium_ml):
        issues.append(
            Issue(
                level="error",
                field="culture.total_medium_ml",
                message="total_medium_ml does not equal per-well volume × wells",
                expected=expected_total_med,
                found=c.total_medium_ml,
            )
        )

    can_predict = (
        c.doubling_time_h is not None
        and c.confluent_density_cells_cm2 is not None
        and c.culture_duration_h is not None
    )
    if c.expected_confluence_pct is not None:
        if not can_predict:
            issues.append(
                Issue(
                    level="warning",
                    field="culture.expected_confluence_pct",
                    message="expected_confluence_pct present but growth inputs "
                    "(doubling_time_h / confluent_density_cells_cm2 / culture_duration_h) "
                    "are missing — it cannot be re-derived",
                )
            )
        else:
            final_cells = calc_cell.cell_count_after_time(
                c.seed_per_well, c.doubling_time_h, c.culture_duration_h
            )
            expected_conf = calc_culture.cell_count_to_confluence(
                final_cells, c.confluent_density_cells_cm2, area
            )
            if not _close(expected_conf, c.expected_confluence_pct):
                issues.append(
                    Issue(
                        level="error",
                        field="culture.expected_confluence_pct",
                        message="expected_confluence_pct does not match the growth prediction",
                        expected=expected_conf,
                        found=c.expected_confluence_pct,
                    )
                )
            if c.expected_confluence_pct > 100:
                issues.append(
                    Issue(
                        level="warning",
                        field="culture.expected_confluence_pct",
                        message=f"predicted confluence {c.expected_confluence_pct:.1f}% exceeds "
                        "100% — over-confluent at harvest",
                    )
                )
    else:
        if can_predict:
            issues.append(
                Issue(
                    level="warning",
                    field="culture.expected_confluence_pct",
                    message="growth inputs are present but expected_confluence_pct is not predicted",
                )
            )

    if c.viability_pct is not None and c.viability_pct < 70:
        issues.append(
            Issue(
                level="warning",
                field="culture.viability_pct",
                message=f"viability {c.viability_pct:.1f}% is below the ~70% pass threshold "
                "commonly used for primary/sensitive cells",
            )
        )


def check_stats(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the replicate count against the stated effect/power."""
    if plan.stats is None:
        return
    n = plan.stats.n_per_group
    # Does the design actually achieve at least the stated power?
    power_achieved = _power_from_n(plan.stats.effect_size, plan.stats.std_dev, n, plan.stats.alpha)
    if power_achieved < plan.stats.power - 1e-9:
        issues.append(
            Issue(
                level="warning",
                field="stats.n_per_group",
                message=f"n={n} per group reaches power {power_achieved:.2f}, below the stated "
                f"target {plan.stats.power:.2f}",
                expected=plan.stats.power,
                found=power_achieved,
            )
        )


def _power_from_n(effect_size: float, std_dev: float, n: int, alpha: float) -> float:
    from labwright.calc import stats as calc_stats

    return calc_stats.power_for_sample_size(n, effect_size, std_dev, alpha)


def verify_design(plan: DesignPlan) -> list[Issue]:
    """Run every cross-check on a design plan. Errors must be resolved before use."""
    issues: list[Issue] = []
    check_flow(plan, issues)
    check_seeding(plan, issues)
    check_culture(plan, issues)
    check_dosing(plan, issues)
    check_stats(plan, issues)
    return issues


def has_errors(issues: list[Issue]) -> bool:
    """True if any issue is at error level (design cannot be trusted)."""
    return any(i.level == "error" for i in issues)


def format_issues(issues: list[Issue]) -> str:
    """Human-readable rendering of verification findings."""
    if not issues:
        return "✓ all derived numbers verified against the calculators"
    lines = []
    for i in issues:
        detail = ""
        if i.expected is not None and i.found is not None:
            detail = f" (expected {i.expected:.6g}, found {i.found:.6g})"
        lines.append(f"[{i.level.upper()}] {i.field}: {i.message}{detail}")
    return "\n".join(lines)


__all__ = ["Issue", "verify_design", "has_errors", "format_issues"]
