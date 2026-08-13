"""Deterministic cross-check of a :class:`~labwright.schema.design.DesignPlan`.

The agent is free to propose geometries, flows, doses and cell plans, but it
is *not* free to invent derived numbers. The checker re-runs the governing
equations on the agent's own inputs and reports every mismatch. Labwright's
public interface refuses to display a design with unresolved errors, which is
the practical difference between "an LLM that guesses numbers" and "an LLM
that quotes the calculators".

Verification is layered on top of the arithmetic cross-checks here:

- :mod:`labwright.verify.units` — every field's canonical unit and the unit
  aliases (dyn/cm² vs Pa, mL/min vs µL/min, ...) that cause real misreads;
- :mod:`labwright.verify.sanity` — physiological/physical range bands;
- :mod:`labwright.verify.safety` — chemical dose limits, biosafety hints and
  the institution's configurable safety boundary.

A design must pass every layer before it can be shown as "verified".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import dosing as calc_dosing
from labwright.calc import microfluidics as mf
from labwright.calc import pk as calc_pk
from labwright.calc import spheroid as calc_spheroid
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
    if plan.chip is None or plan.flow is None or plan.derived is None:
        return
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
    if plan.cells is None or plan.chip is None:
        return
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
    # DMSO fraction must match stock/working concentrations. The solvent-toxicity
    # warning lives in the safety layer (:mod:`labwright.verify.safety`) so the
    # threshold is the institution's configurable safety boundary, not a literal.
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


def check_spheroid(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the 3D spheroid/organoid plan when present.

    Re-runs :mod:`labwright.calc.spheroid` on the raw inputs and verifies every
    derived field; warns on necrotic-core sizes and on missing growth inputs
    that make the harvest cell count unpredictable.
    """
    s = plan.spheroid
    if s is None:
        return

    expected_d = calc_spheroid.spheroid_diameter_from_cells(
        s.cells_per_spheroid, s.cell_diameter_um
    )
    if not _close(expected_d, s.expected_diameter_um):
        issues.append(
            Issue(
                level="error",
                field="spheroid.expected_diameter_um",
                message="expected_diameter_um does not match cells_per_spheroid × "
                "single-cell volume (solid-sphere packing)",
                expected=expected_d,
                found=s.expected_diameter_um,
            )
        )
    expected_v = calc_spheroid.spheroid_volume_from_cells(
        s.cells_per_spheroid, s.cell_diameter_um
    )
    if not _close(expected_v, s.spheroid_volume_ul):
        issues.append(
            Issue(
                level="error",
                field="spheroid.spheroid_volume_ul",
                message="spheroid_volume_ul does not match cells_per_spheroid × single-cell volume",
                expected=expected_v,
                found=s.spheroid_volume_ul,
            )
        )
    expected_total = s.spheroid_count * s.cells_per_spheroid
    if not _close(expected_total, s.cells_total):
        issues.append(
            Issue(
                level="error",
                field="spheroid.cells_total",
                message="cells_total does not equal spheroid_count × cells_per_spheroid",
                expected=expected_total,
                found=s.cells_total,
            )
        )
    expected_med = calc_spheroid.medium_volume_per_spheroid(s.spheroid_format)
    if not _close(expected_med, s.medium_volume_per_spheroid_ul):
        issues.append(
            Issue(
                level="error",
                field="spheroid.medium_volume_per_spheroid_ul",
                message="medium_volume_per_spheroid_ul does not equal the standard working volume",
                expected=expected_med,
                found=s.medium_volume_per_spheroid_ul,
            )
        )
    expected_total_med = s.spheroid_count * s.medium_volume_per_spheroid_ul / 1000.0
    if not _close(expected_total_med, s.total_medium_ml):
        issues.append(
            Issue(
                level="error",
                field="spheroid.total_medium_ml",
                message="total_medium_ml does not equal per-spheroid volume × spheroid_count",
                expected=expected_total_med,
                found=s.total_medium_ml,
            )
        )

    if s.expected_diameter_um > 400:
        issues.append(
            Issue(
                level="warning",
                field="spheroid.expected_diameter_um",
                message=f"expected spheroid diameter {s.expected_diameter_um:.1f} µm exceeds "
                "~400 µm — oxygen diffuses only ~200 µm, so a necrotic core is likely",
            )
        )

    can_predict = (
        s.doubling_time_h is not None and s.culture_duration_h is not None
    )
    if s.expected_cells_after_growth is not None:
        if not can_predict:
            issues.append(
                Issue(
                    level="warning",
                    field="spheroid.expected_cells_after_growth",
                    message="expected_cells_after_growth present but growth inputs "
                    "(doubling_time_h / culture_duration_h) are missing — it cannot be re-derived",
                )
            )
        else:
            expected_g = calc_cell.cell_count_after_time(
                s.cells_per_spheroid, s.doubling_time_h, s.culture_duration_h
            )
            if not _close(expected_g, s.expected_cells_after_growth):
                issues.append(
                    Issue(
                        level="error",
                        field="spheroid.expected_cells_after_growth",
                        message="expected_cells_after_growth does not match the growth prediction",
                        expected=expected_g,
                        found=s.expected_cells_after_growth,
                    )
                )
    else:
        if can_predict:
            issues.append(
                Issue(
                    level="warning",
                    field="spheroid.expected_cells_after_growth",
                    message="growth inputs are present but expected_cells_after_growth is not predicted",
                )
            )


def check_oxygen(plan: DesignPlan, issues: list[Issue]) -> None:
    """Warn when a perfused design's O2 supply is below cellular demand.

    Supply is perfused O2 at maximal extraction (:func:`~labwright.calc.microfluidics.o2_delivery_rate`,
    air-equilibrated inlet); demand uses the physiology registry's OCR
    (nmol/min per 10⁶ cells, mid-range) converted to fmol/s/cell. Fires only
    when the design is perfused *and* the cell type resolves to a registry OCR
    — otherwise demand cannot be quantified without inventing numbers, so the
    check stays silent.
    """
    from labwright.calc import o2 as calc_o2
    from labwright.physiology import lookup_cell

    if plan.flow is None:
        return  # not perfused — O2 supply is diffusive, not computable here
    if plan.cells is not None and plan.cells.seed_count is not None:
        total_cells, cell_type = plan.cells.seed_count, plan.cells.cell_type
    elif plan.spheroid is not None:
        total_cells, cell_type = plan.spheroid.cells_total, plan.spheroid.cell_type
    else:
        return  # no quantitative cell load to match against supply

    profile = lookup_cell(cell_type)
    if profile is None or profile.o2_consumption_nmol_min_1e6 is None:
        return  # no registry OCR for this cell type — do not guess
    lo, hi = profile.o2_consumption_nmol_min_1e6
    mid_ocr = (lo + hi) / 2.0
    per_cell_fmol_s = calc_o2.nmol_min_per_1e6_to_fmol_s(mid_ocr)

    supply = mf.o2_delivery_rate(plan.flow.flow_rate_uLmin, calc_o2.AIR_SATURATED_O2_MM * 1e-3)
    demand = calc_o2.o2_demand_umol_min(total_cells, per_cell_fmol_s)
    if supply < demand:
        issues.append(
            Issue(
                level="warning",
                field="flow.flow_rate_uLmin",
                message=f"perfused O2 supply ≈ {supply:.4g} µmol/min is below cellular demand ≈ "
                f"{demand:.4g} µmol/min (registry OCR {lo:g}-{hi:g} nmol/min per 10⁶ cells, mid-range) "
                "— the culture is likely hypoxic; raise flow, reduce cell load, or oxygenate the medium",
                expected=demand,
                found=supply,
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


def check_pk(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the perfused-system PK plan when present.

    Re-runs :mod:`labwright.calc.pk` on the raw inlet/outlet/flow inputs and
    verifies every derived field; warns when the outlet exceeds the inlet (a
    negative extraction ratio — active secretion or a measurement error) and
    when a half-life is far shorter than the pass-through time (the "cleared in
    one pass" regime where a single-compartment half-life is not meaningful).
    """
    p = plan.pk
    if p is None:
        return

    e = calc_pk.extraction_ratio(p.inlet_concentration_uM, p.outlet_concentration_uM)
    if not _close(e, p.extraction_ratio):
        issues.append(
            Issue(
                level="error",
                field="pk.extraction_ratio",
                message="extraction_ratio does not equal 1 − C_out/C_in",
                expected=e,
                found=p.extraction_ratio,
            )
        )
    cl = calc_pk.clearance_uLmin(
        p.inlet_concentration_uM, p.outlet_concentration_uM, p.flow_rate_uLmin
    )
    if not _close(cl, p.clearance_uLmin):
        issues.append(
            Issue(
                level="error",
                field="pk.clearance_uLmin",
                message="clearance_uLmin does not equal extraction_ratio × flow_rate_uLmin",
                expected=cl,
                found=p.clearance_uLmin,
            )
        )

    if p.system_volume_uL is not None:
        if p.half_life_h is None:
            issues.append(
                Issue(
                    level="error",
                    field="pk.half_life_h",
                    message="system_volume_uL present but half_life_h is not computed",
                )
            )
        else:
            t_half = calc_pk.half_life_h(p.system_volume_uL, p.clearance_uLmin)
            if not _close(t_half, p.half_life_h):
                issues.append(
                    Issue(
                        level="error",
                        field="pk.half_life_h",
                        message="half_life_h does not equal ln2·V/Cl",
                        expected=t_half,
                        found=p.half_life_h,
                    )
                )
    elif p.half_life_h is not None:
        issues.append(
            Issue(
                level="warning",
                field="pk.half_life_h",
                message="half_life_h present but system_volume_uL is missing — it cannot be re-derived",
            )
        )

    if p.half_life_h is not None and p.dose_interval_h is not None:
        r = calc_pk.accumulation_ratio(p.half_life_h, p.dose_interval_h)
        if p.accumulation_ratio is None or not _close(r, p.accumulation_ratio):
            issues.append(
                Issue(
                    level="error",
                    field="pk.accumulation_ratio",
                    message="accumulation_ratio does not equal 1/(1 − e^(−ln2·τ/t½))",
                    expected=r,
                    found=p.accumulation_ratio,
                )
            )
    elif p.accumulation_ratio is not None:
        issues.append(
            Issue(
                level="warning",
                field="pk.accumulation_ratio",
                message="accumulation_ratio present but half-life or dose interval is missing — it cannot be re-derived",
            )
        )

    if p.molecular_weight_g_mol is not None:
        if p.mass_cleared_ug_h is None:
            issues.append(
                Issue(
                    level="error",
                    field="pk.mass_cleared_ug_h",
                    message="molecular_weight_g_mol present but mass_cleared_ug_h is not computed",
                )
            )
        else:
            m = calc_pk.mass_cleared_ug_h(
                p.clearance_uLmin, p.inlet_concentration_uM, p.molecular_weight_g_mol
            )
            if not _close(m, p.mass_cleared_ug_h):
                issues.append(
                    Issue(
                        level="error",
                        field="pk.mass_cleared_ug_h",
                        message="mass_cleared_ug_h does not equal Cl·C_in·MW·6e-5",
                        expected=m,
                        found=p.mass_cleared_ug_h,
                    )
                )
    elif p.mass_cleared_ug_h is not None:
        issues.append(
            Issue(
                level="warning",
                field="pk.mass_cleared_ug_h",
                message="mass_cleared_ug_h present but molecular_weight_g_mol is missing — it cannot be re-derived",
            )
        )

    if p.extraction_ratio < 0:
        issues.append(
            Issue(
                level="warning",
                field="pk.extraction_ratio",
                message=f"extraction_ratio {p.extraction_ratio:.3f} is negative — outlet exceeds inlet: "
                "active secretion (influx transporters) or an inlet/outlet measurement error",
            )
        )
    if (
        p.clearance_uLmin > 0
        and p.system_volume_uL is not None
        and p.half_life_h is not None
        and p.half_life_h * 3600 < p.system_volume_uL / p.flow_rate_uLmin
    ):
        issues.append(
            Issue(
                level="warning",
                field="pk.half_life_h",
                message=f"half-life {p.half_life_h:.3g} h is shorter than a single pass-through "
                f"(~{p.system_volume_uL / p.flow_rate_uLmin:.0f} min) — the system is cleared in "
                "~one pass and the single-compartment t½ = ln2·V/Cl is not physiologically meaningful",
            )
        )


def verify_design(plan: DesignPlan) -> list[Issue]:
    """Run every cross-check on a design plan. Errors must be resolved before use.

    The layers run in order: arithmetic cross-checks, physiological range
    checks (:mod:`~labwright.verify.sanity`) and, when a safety profile is in
    force, chemical-dose / biosafety checks (:mod:`~labwright.verify.safety`).
    """
    issues: list[Issue] = []
    check_flow(plan, issues)
    check_seeding(plan, issues)
    check_culture(plan, issues)
    check_spheroid(plan, issues)
    check_oxygen(plan, issues)
    check_dosing(plan, issues)
    check_stats(plan, issues)
    check_pk(plan, issues)
    from labwright.verify.sanity import check_sanity
    from labwright.verify.safety import check_safety
    from labwright.verify.prose import check_prose_numbers

    check_sanity(plan, issues)
    check_safety(plan, issues)
    check_prose_numbers(plan, issues)
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
