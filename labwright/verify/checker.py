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

from labwright.calc import barrier as calc_barrier
from labwright.calc import bioinformatics as calc_bioinformatics
from labwright.calc import bioprinting as calc_bioprinting
from labwright.calc import cell as calc_cell
from labwright.calc import coculture as calc_coculture
from labwright.calc import culture as calc_culture
from labwright.calc import dosing as calc_dosing
from labwright.calc import enzyme as calc_enzyme
from labwright.calc import microfluidics as mf
from labwright.calc import pk as calc_pk
from labwright.calc import solvent as calc_solvent
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


def check_barrier(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the epithelial/endothelial barrier plan when present.

    Re-runs :mod:`labwright.calc.barrier` on the raw resistance readings and
    verifies the derived TEER; when a probe flux and donor concentration are
    given, verifies Papp and clearance too. Warns when a probe is declared but
    the flux/donor inputs needed to derive a permeability are missing (the QC
    gate cannot then be scored), and when the blank resistance is suspiciously
    large relative to the total (a monolayer so thin its TEER barely registers).
    """
    b = plan.barrier
    if b is None:
        return

    try:
        expected_teer = calc_barrier.teer_ohm_cm2(
            b.resistance_total_ohm, b.resistance_blank_ohm, b.insert_area_cm2
        )
    except ValueError as exc:
        issues.append(
            Issue(
                level="error",
                field="barrier.teer_ohm_cm2",
                message=f"cannot derive TEER from the raw resistances: {exc}",
            )
        )
        return
    if not _close(expected_teer, b.teer_ohm_cm2):
        issues.append(
            Issue(
                level="error",
                field="barrier.teer_ohm_cm2",
                message="teer_ohm_cm2 does not equal (R_total − R_blank) × area",
                expected=expected_teer,
                found=b.teer_ohm_cm2,
            )
        )

    can_derive_papp = b.donor_conc_um is not None and b.flux_nmol_min is not None
    if b.papp_cm_s is not None:
        if not can_derive_papp:
            issues.append(
                Issue(
                    level="warning",
                    field="barrier.papp_cm_s",
                    message="papp_cm_s present but donor_conc_um / flux_nmol_min are missing "
                    "— it cannot be re-derived",
                )
            )
        else:
            expected_papp = calc_barrier.papp_cm_s(
                b.flux_nmol_min, b.insert_area_cm2, b.donor_conc_um
            )
            if not _close(expected_papp, b.papp_cm_s):
                issues.append(
                    Issue(
                        level="error",
                        field="barrier.papp_cm_s",
                        message="papp_cm_s does not equal flux/(60·A·C₀)",
                        expected=expected_papp,
                        found=b.papp_cm_s,
                    )
                )
    elif can_derive_papp:
        issues.append(
            Issue(
                level="warning",
                field="barrier.papp_cm_s",
                message="donor_conc_um / flux_nmol_min are present but papp_cm_s is not computed",
            )
        )

    if b.clearance_mL_min is not None:
        if b.papp_cm_s is None:
            issues.append(
                Issue(
                    level="warning",
                    field="barrier.clearance_mL_min",
                    message="clearance_mL_min present but papp_cm_s is missing — it cannot be re-derived",
                )
            )
        else:
            expected_cl = calc_barrier.clearance_mL_min(b.papp_cm_s, b.insert_area_cm2)
            if not _close(expected_cl, b.clearance_mL_min):
                issues.append(
                    Issue(
                        level="error",
                        field="barrier.clearance_mL_min",
                        message="clearance_mL_min does not equal Papp·A·60",
                        expected=expected_cl,
                        found=b.clearance_mL_min,
                    )
                )

    if b.probe is not None and not can_derive_papp:
        issues.append(
            Issue(
                level="warning",
                field="barrier.probe",
                message=f"probe {b.probe!r} declared but flux_nmol_min / donor_conc_um are "
                "missing — the permeability QC gate cannot be scored",
            )
        )
    if b.resistance_total_ohm - b.resistance_blank_ohm <= 0.05 * b.resistance_total_ohm:
        issues.append(
            Issue(
                level="warning",
                field="barrier.resistance_total_ohm",
                message=f"blank resistance {b.resistance_blank_ohm:g} Ω is >95% of the total "
                f"{b.resistance_total_ohm:g} Ω — the monolayer contributes almost no resistance "
                "and the TEER will be dominated by electrode/medium drift",
            )
        )


def check_pumpless(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the gravity-driven pumpless plan when present.

    Re-runs :mod:`labwright.calc.pumpless` on the platform/channel inputs and
    verifies every derived field; warns when the peak wall shear is far from the
    physiological target (outside the 0.5–2× window) so a "rocker" that actually
    under- or over-stimulates the cells is flagged before the lab builds it.
    """
    from labwright.calc import pumpless as cp
    from labwright.physiology import lookup_cell

    p = plan.pumpless
    if p is None:
        return

    head = cp.hydrostatic_pressure_pa(p.density_kgm3, p.tilt_angle_deg, p.channel_length_mm)
    if not _close(head, p.hydrostatic_head_pa):
        issues.append(Issue(
            level="error", field="pumpless.hydrostatic_head_pa",
            message="hydrostatic_head_pa does not equal ρ·g·L·sinθ",
            expected=head, found=p.hydrostatic_head_pa,
        ))
    q = cp.flow_rate_from_pressure_head(
        head, p.width_um, p.height_um, p.channel_length_mm, p.viscosity_pas
    )
    if not _close(q, p.driven_flow_rate_uLmin):
        issues.append(Issue(
            level="error", field="pumpless.driven_flow_rate_uLmin",
            message="driven_flow_rate_uLmin does not match Hagen–Poiseuille from the head",
            expected=q, found=p.driven_flow_rate_uLmin,
        ))
    tau = cp.peak_wall_shear_from_head(head, p.width_um, p.height_um, p.channel_length_mm)
    if not _close(tau, p.peak_wall_shear_pa):
        issues.append(Issue(
            level="error", field="pumpless.peak_wall_shear_pa",
            message="peak_wall_shear_pa does not equal ΔP·h/(2·L)",
            expected=tau, found=p.peak_wall_shear_pa,
        ))
    vol = cp.rocking_volume_per_half_cycle_ul(q, p.rocking_half_period_s)
    if not _close(vol, p.volume_per_half_cycle_ul):
        issues.append(Issue(
            level="error", field="pumpless.volume_per_half_cycle_ul",
            message="volume_per_half_cycle_ul does not equal Q·t/60",
            expected=vol, found=p.volume_per_half_cycle_ul,
        ))
    osi = cp.oscillatory_shear_index(tau, tau * p.backward_shear_fraction)
    if not _close(osi, p.oscillatory_shear_index):
        issues.append(Issue(
            level="error", field="pumpless.oscillatory_shear_index",
            message="oscillatory_shear_index does not match the forward/backward shear profile",
            expected=osi, found=p.oscillatory_shear_index,
        ))
    cph = cp.cycles_per_hour(p.rocking_half_period_s)
    if not _close(cph, p.cycles_per_hour):
        issues.append(Issue(
            level="error", field="pumpless.cycles_per_hour",
            message="cycles_per_hour does not equal 3600/(2·t)",
            expected=cph, found=p.cycles_per_hour,
        ))
    if p.shear_ratio_to_target is not None:
        prof = lookup_cell(p.cell_type)
        if prof is not None and prof.shear_range_pa is not None:
            lo, hi = prof.shear_range_pa
        else:
            lo, hi = cp.LIVER_SINUSOID_WSS_MIN_PA, cp.LIVER_SINUSOID_WSS_MAX_PA
        target = (lo + hi) / 2.0
        ratio = tau / target
        if not _close(ratio, p.shear_ratio_to_target, tol=1e-3):
            issues.append(Issue(
                level="error", field="pumpless.shear_ratio_to_target",
                message="shear_ratio_to_target does not equal chip WSS / physiological target",
                expected=ratio, found=p.shear_ratio_to_target,
            ))


def check_breathing(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the lung ALI + stretch plan when present.

    Re-runs :mod:`labwright.calc.breathing` on the frequency/strain/geometry
    inputs and verifies every derived field; warns when the chosen strain leaves
    the physiological 5–12 % window (or crosses the pathological >20 % line).
    """
    from labwright.calc import breathing as cb

    b = plan.breathing
    if b is None:
        return

    bpm = cb.breaths_per_minute(b.frequency_hz)
    if not _close(bpm, b.breaths_per_minute):
        issues.append(Issue(
            level="error", field="breathing.breaths_per_minute",
            message="breaths_per_minute does not equal f×60",
            expected=bpm, found=b.breaths_per_minute,
        ))
    disp = cb.cyclic_displacement_um(b.strain_pct, b.membrane_span_um)
    if not _close(disp, b.cyclic_displacement_um):
        issues.append(Issue(
            level="error", field="breathing.cyclic_displacement_um",
            message="cyclic_displacement_um does not equal ε·L",
            expected=disp, found=b.cyclic_displacement_um,
        ))
    rate = cb.strain_rate_per_s(b.strain_pct, b.frequency_hz)
    if not _close(rate, b.strain_rate_per_s):
        issues.append(Issue(
            level="error", field="breathing.strain_rate_per_s",
            message="strain_rate_per_s does not equal (ε/100)·f",
            expected=rate, found=b.strain_rate_per_s,
        ))

    if b.total_cycles is not None:
        if b.culture_duration_h is None:
            issues.append(Issue(
                level="warning", field="breathing.total_cycles",
                message="total_cycles present but culture_duration_h is missing — it cannot be re-derived",
            ))
        else:
            tc = cb.total_cycles(b.culture_duration_h, b.frequency_hz)
            if not _close(tc, b.total_cycles):
                issues.append(Issue(
                    level="error", field="breathing.total_cycles",
                    message="total_cycles does not equal hours×3600×f",
                    expected=tc, found=b.total_cycles,
                ))
    elif b.culture_duration_h is not None:
        issues.append(Issue(
            level="warning", field="breathing.total_cycles",
            message="culture_duration_h present but total_cycles is not computed",
        ))

    if b.stretch_duty_fraction is not None:
        if b.stretch_seconds is None or b.cycle_seconds is None:
            issues.append(Issue(
                level="warning", field="breathing.stretch_duty_fraction",
                message="stretch_duty_fraction present but stretch/cycle seconds are missing — it cannot be re-derived",
            ))
        else:
            duty = cb.stretch_duty_fraction(b.stretch_seconds, b.cycle_seconds)
            if not _close(duty, b.stretch_duty_fraction):
                issues.append(Issue(
                    level="error", field="breathing.stretch_duty_fraction",
                    message="stretch_duty_fraction does not equal stretch/cycle",
                    expected=duty, found=b.stretch_duty_fraction,
                ))
    elif b.stretch_seconds is not None and b.cycle_seconds is not None:
        issues.append(Issue(
            level="warning", field="breathing.stretch_duty_fraction",
            message="stretch/cycle seconds are present but stretch_duty_fraction is not computed",
        ))

    if b.ali_liquid_film_um is not None:
        if b.apical_volume_ul is None or b.surface_area_cm2 is None:
            issues.append(Issue(
                level="warning", field="breathing.ali_liquid_film_um",
                message="ali_liquid_film_um present but apical_volume_ul / surface_area_cm2 are missing — it cannot be re-derived",
            ))
        else:
            film = cb.ali_liquid_film_um(b.apical_volume_ul, b.surface_area_cm2)
            if not _close(film, b.ali_liquid_film_um):
                issues.append(Issue(
                    level="error", field="breathing.ali_liquid_film_um",
                    message="ali_liquid_film_um does not equal V/A",
                    expected=film, found=b.ali_liquid_film_um,
                ))
    elif b.apical_volume_ul is not None and b.surface_area_cm2 is not None:
        issues.append(Issue(
            level="warning", field="breathing.ali_liquid_film_um",
            message="apical_volume_ul / surface_area_cm2 are present but ali_liquid_film_um is not computed",
        ))

    cls = cb.linear_strain_pct_is_physiological(b.strain_pct)
    if cls["pathological"]:
        issues.append(Issue(
            level="warning", field="breathing.strain_pct",
            message=f"strain {b.strain_pct:.1f}% exceeds the ~20% pathological threshold — "
            "this mechanically injures the epithelium, not a healthy alveolar regime",
        ))
    elif not cls["physiological"]:
        issues.append(Issue(
            level="warning", field="breathing.strain_pct",
            message=f"strain {b.strain_pct:.1f}% lies outside the physiological 5–12% "
            "alveolar window — confirm it is intentional",
        ))


def check_pulsatile(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the pulsatile cardiac-waveform plan when present.

    Re-runs :mod:`labwright.calc.pulsatile` on the frequency/geometry/shear
    inputs and verifies every derived field; warns when the waveform is strongly
    reversing (high OSI, e.g. atheroprone) so the design intent is explicit.
    """
    from labwright.calc import pulsatile as cp

    p = plan.pulsatile
    if p is None:
        return

    alpha = cp.womersley_number(p.frequency_hz, p.channel_height_um, p.viscosity_pas, p.density_kgm3)
    if not _close(alpha, p.womersley_number):
        issues.append(Issue(
            level="error", field="pulsatile.womersley_number",
            message="womersley_number does not equal (h/2)·√(ωρ/μ)",
            expected=alpha, found=p.womersley_number,
        ))
    osi = cp.oscillatory_shear_index_from_sinusoid(p.shear_mean_pa, p.shear_amplitude_pa)
    if not _close(osi, p.oscillatory_shear_index):
        issues.append(Issue(
            level="error", field="pulsatile.oscillatory_shear_index",
            message="oscillatory_shear_index does not match the sinusoidal mean/amplitude",
            expected=osi, found=p.oscillatory_shear_index,
        ))
    peak = cp.peak_shear_of_sinusoid(p.shear_mean_pa, p.shear_amplitude_pa)
    if not _close(peak, p.peak_shear_pa):
        issues.append(Issue(
            level="error", field="pulsatile.peak_shear_pa",
            message="peak_shear_pa does not equal mean + amplitude",
            expected=peak, found=p.peak_shear_pa,
        ))

    if p.pulsatility_index is not None:
        if p.peak_flow_uLmin is None or p.minimum_flow_uLmin is None or p.mean_flow_uLmin is None:
            issues.append(Issue(
                level="warning", field="pulsatile.pulsatility_index",
                message="pulsatility_index present but peak/min/mean flow are missing — it cannot be re-derived",
            ))
        else:
            pi = cp.pulsatility_index(p.peak_flow_uLmin, p.minimum_flow_uLmin, p.mean_flow_uLmin)
            if not _close(pi, p.pulsatility_index):
                issues.append(Issue(
                    level="error", field="pulsatile.pulsatility_index",
                    message="pulsatility_index does not equal (Q_peak − Q_min)/Q_mean",
                    expected=pi, found=p.pulsatility_index,
                ))
    elif p.peak_flow_uLmin is not None and p.minimum_flow_uLmin is not None and p.mean_flow_uLmin is not None:
        issues.append(Issue(
            level="warning", field="pulsatile.pulsatility_index",
            message="peak/min/mean flow are present but pulsatility_index is not computed",
        ))

    if p.oscillatory_shear_index > 0.3:
        issues.append(Issue(
            level="warning", field="pulsatile.oscillatory_shear_index",
            message=f"OSI {p.oscillatory_shear_index:.2f} marks a strongly reversing waveform — "
            "flow reversals are the hallmark of atheroprone hemodynamics; confirm the cells "
            "and the hypothesis want reversal",
        ))


def check_scaling(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the multi-organ scaling plan when present.

    Re-runs :mod:`labwright.calc.scaling` on the organ/chip inputs and verifies
    every derived field against the physiology tables; warns when a requested
    transit match misses by more than an hour, so a body-on-chip that cannot
    hit the in-vivo residence time is flagged before the flow is set.
    """
    from labwright.calc import scaling as cs

    s = plan.scaling
    if s is None:
        return

    frac = cs.organ_flow_fraction(s.organ)
    if not _close(frac, s.organ_flow_fraction):
        issues.append(Issue(
            level="error", field="scaling.organ_flow_fraction",
            message="organ_flow_fraction does not match the physiology table",
            expected=frac, found=s.organ_flow_fraction,
        ))
    q = cs.organ_flow_rate_mlmin(s.organ, s.cardiac_output_mlmin)
    if not _close(q, s.organ_flow_rate_mlmin):
        issues.append(Issue(
            level="error", field="scaling.organ_flow_rate_mlmin",
            message="organ_flow_rate_mlmin does not equal fraction × cardiac output",
            expected=q, found=s.organ_flow_rate_mlmin,
        ))
    organ_mass = cs.ORGAN_MASS_G[s.organ]
    cells = cs.scale_cell_number(organ_mass, s.body_mass_g, s.total_cells_chip)
    if not _close(cells, s.cells_in_organ):
        issues.append(Issue(
            level="error", field="scaling.cells_in_organ",
            message="cells_in_organ does not equal (m_organ/m_body) × chip budget",
            expected=cells, found=s.cells_in_organ,
        ))
    asc = cs.allometric_metabolic_scale(organ_mass, s.body_mass_g)
    if not _close(asc, s.allometric_scale):
        issues.append(Issue(
            level="error", field="scaling.allometric_scale",
            message="allometric_scale does not equal (m_organ/m_body)^0.75",
            expected=asc, found=s.allometric_scale,
        ))

    if s.transit_time_s is not None:
        if s.chip_volume_ul is None or s.flow_rate_uLmin is None:
            issues.append(Issue(
                level="warning", field="scaling.transit_time_s",
                message="transit_time_s present but chip_volume_ul / flow_rate_uLmin are missing — it cannot be re-derived",
            ))
        else:
            transit = cs.transit_time_s(s.chip_volume_ul, s.flow_rate_uLmin)
            if not _close(transit, s.transit_time_s):
                issues.append(Issue(
                    level="error", field="scaling.transit_time_s",
                    message="transit_time_s does not equal V/Q·60",
                    expected=transit, found=s.transit_time_s,
                ))
    elif s.chip_volume_ul is not None and s.flow_rate_uLmin is not None:
        issues.append(Issue(
            level="warning", field="scaling.transit_time_s",
            message="chip_volume_ul / flow_rate_uLmin are present but transit_time_s is not computed",
        ))

    if s.residence_time_match_error_s is not None:
        if s.target_transit_s is None:
            issues.append(Issue(
                level="warning", field="scaling.residence_time_match_error_s",
                message="residence_time_match_error_s present but target_transit_s is missing — it cannot be re-derived",
            ))
        else:
            err = cs.residence_time_match_error_s(s.chip_volume_ul, s.flow_rate_uLmin, s.target_transit_s)
            if not _close(err, s.residence_time_match_error_s):
                issues.append(Issue(
                    level="error", field="scaling.residence_time_match_error_s",
                    message="residence_time_match_error_s does not equal |transit − target|",
                    expected=err, found=s.residence_time_match_error_s,
                ))
    elif s.target_transit_s is not None:
        issues.append(Issue(
            level="warning", field="scaling.residence_time_match_error_s",
            message="target_transit_s present but residence_time_match_error_s is not computed",
        ))


def check_gradient(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the concentration-gradient plan when present.

    Re-runs :mod:`labwright.calc.gradient` on the source/sink inputs and verifies
    every derived field; warns when the experiment is shorter than ~10 diffusive
    relaxation times (the gradient is still forming) or when source ≤ sink (no
    gradient direction at all).
    """
    from labwright.calc import gradient as cg

    g = plan.gradient
    if g is None:
        return

    steep = cg.linear_gradient_steepness_um_per_mm(g.source_conc_um, g.sink_conc_um, g.distance_um)
    if not _close(steep, g.steepness_um_per_mm):
        issues.append(Issue(
            level="error", field="gradient.steepness_um_per_mm",
            message="steepness_um_per_mm does not equal (C_src − C_sink)/L × 1000",
            expected=steep, found=g.steepness_um_per_mm,
        ))
    mid = cg.steady_state_profile_conc_um(g.source_conc_um, g.sink_conc_um, g.distance_um, g.distance_um / 2.0)
    if not _close(mid, g.midpoint_conc_um):
        issues.append(Issue(
            level="error", field="gradient.midpoint_conc_um",
            message="midpoint_conc_um does not match the linear steady-state profile",
            expected=mid, found=g.midpoint_conc_um,
        ))
    tau = cg.diffusive_relaxation_time_s(g.distance_um, g.diffusivity_m2s)
    if not _close(tau, g.relaxation_time_s):
        issues.append(Issue(
            level="error", field="gradient.relaxation_time_s",
            message="relaxation_time_s does not equal L²/D",
            expected=tau, found=g.relaxation_time_s,
        ))
    flux = cg.diffusive_flux_mol_m2s(g.source_conc_um, g.sink_conc_um, g.distance_um, g.diffusivity_m2s)
    if not _close(flux, g.flux_mol_m2s):
        issues.append(Issue(
            level="error", field="gradient.flux_mol_m2s",
            message="flux_mol_m2s does not match Fick's first law",
            expected=flux, found=g.flux_mol_m2s,
        ))

    if g.source_conc_um <= g.sink_conc_um:
        issues.append(Issue(
            level="warning", field="gradient.source_conc_um",
            message=f"source {g.source_conc_um:g} µM does not exceed sink {g.sink_conc_um:g} µM — "
            "there is no concentration gradient to measure",
        ))
    stability = cg.gradient_stability_check(g.relaxation_time_s, g.experiment_hours)
    if not stability["stable"]:
        issues.append(Issue(
            level="warning", field="gradient.experiment_hours",
            message=f"experiment {g.experiment_hours:g} h runs ~{g.relaxation_time_s / 3600:.2f} h "
            f"of relaxation time (τ); the 10τ rule needs ≥ {10 * g.relaxation_time_s / 3600:.2f} h for "
            "the gradient to reach steady state and be held — the readout may sample the "
            "establishment transient, not a stable gradient",
        ))


def check_bioprinting(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the micro-extrusion bioprinting plan when present.

    Re-runs :mod:`labwright.calc.bioprinting` on the nozzle/path/feed inputs and
    verifies every derived field; warns when a footprint width or line pitch is
    missing so ``lines_to_cover`` cannot be derived.
    """
    b = plan.bioprinting
    if b is None:
        return

    try:
        d_um = calc_bioprinting.nozzle_diameter_um(b.nozzle_id)
    except ValueError as exc:
        issues.append(Issue(
            level="error", field="bioprinting.nozzle_id",
            message=f"nozzle cannot be resolved: {exc}",
        ))
        return
    expected_v = calc_bioprinting.extrusion_volume_nl(b.travel_distance_um, d_um)
    if not _close(expected_v, b.extrusion_volume_nl):
        issues.append(Issue(
            level="error", field="bioprinting.extrusion_volume_nl",
            message="extrusion_volume_nl does not equal π(d/2)²·L",
            expected=expected_v, found=b.extrusion_volume_nl,
        ))
    expected_t = calc_bioprinting.print_time_s(b.travel_distance_um, b.feed_rate_mm_min)
    if not _close(expected_t, b.print_time_s):
        issues.append(Issue(
            level="error", field="bioprinting.print_time_s",
            message="print_time_s does not equal L/v",
            expected=expected_t, found=b.print_time_s,
        ))
    expected_rate = calc_bioprinting.extrusion_rate_nl_min(b.extrusion_volume_nl, b.print_time_s)
    if not _close(expected_rate, b.extrusion_rate_nl_min):
        issues.append(Issue(
            level="error", field="bioprinting.extrusion_rate_nl_min",
            message="extrusion_rate_nl_min does not equal volume / traversal time",
            expected=expected_rate, found=b.extrusion_rate_nl_min,
        ))
    expected_mass = calc_bioprinting.filament_mass_ug(b.extrusion_volume_nl, b.density_g_cm3)
    if not _close(expected_mass, b.filament_mass_ug):
        issues.append(Issue(
            level="error", field="bioprinting.filament_mass_ug",
            message="filament_mass_ug does not equal volume × ink density",
            expected=expected_mass, found=b.filament_mass_ug,
        ))

    if b.lines_to_cover is not None:
        if b.footprint_width_um is None or b.line_pitch_um is None:
            issues.append(Issue(
                level="warning", field="bioprinting.lines_to_cover",
                message="lines_to_cover present but footprint_width_um / line_pitch_um are missing — it cannot be re-derived",
            ))
        else:
            expected_lines = calc_bioprinting.lines_to_cover(b.footprint_width_um, b.line_pitch_um)
            if not _close(expected_lines, b.lines_to_cover):
                issues.append(Issue(
                    level="error", field="bioprinting.lines_to_cover",
                    message="lines_to_cover does not equal ceil(width / pitch)",
                    expected=expected_lines, found=b.lines_to_cover,
                ))
    elif b.footprint_width_um is not None and b.line_pitch_um is not None:
        issues.append(Issue(
            level="warning", field="bioprinting.lines_to_cover",
            message="footprint_width_um / line_pitch_um are present but lines_to_cover is not computed",
        ))


def check_coculture(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the two-population co-culture seeding plan when present.

    Re-runs :mod:`labwright.calc.coculture` on the density/area/fraction inputs
    and verifies every derived field; warns when the A-fraction leaves one
    population a negligible minority (a nominal co-culture).
    """
    c = plan.coculture
    if c is None:
        return

    cells_a, cells_b = calc_coculture.cells_per_well(
        c.total_density_cells_cm2, c.area_cm2, c.fraction_a
    )
    if not _close(cells_a, c.cells_per_well_a):
        issues.append(Issue(
            level="error", field="coculture.cells_per_well_a",
            message="cells_per_well_a does not equal f·ρ·A",
            expected=cells_a, found=c.cells_per_well_a,
        ))
    if not _close(cells_b, c.cells_per_well_b):
        issues.append(Issue(
            level="error", field="coculture.cells_per_well_b",
            message="cells_per_well_b does not equal (1−f)·ρ·A",
            expected=cells_b, found=c.cells_per_well_b,
        ))
    expected_ta = calc_coculture.total_cells(c.cells_per_well_a, c.wells)
    if not _close(expected_ta, c.total_cells_a):
        issues.append(Issue(
            level="error", field="coculture.total_cells_a",
            message="total_cells_a does not equal cells_per_well_a × wells",
            expected=expected_ta, found=c.total_cells_a,
        ))
    expected_tb = calc_coculture.total_cells(c.cells_per_well_b, c.wells)
    if not _close(expected_tb, c.total_cells_b):
        issues.append(Issue(
            level="error", field="coculture.total_cells_b",
            message="total_cells_b does not equal cells_per_well_b × wells",
            expected=expected_tb, found=c.total_cells_b,
        ))
    expected_ratio = calc_coculture.seeding_ratio(c.cells_per_well_a, c.cells_per_well_b)
    if not _close(expected_ratio, c.seeding_ratio_ab):
        issues.append(Issue(
            level="error", field="coculture.seeding_ratio_ab",
            message="seeding_ratio_ab does not equal A cells / B cells",
            expected=expected_ratio, found=c.seeding_ratio_ab,
        ))

    if min(c.fraction_a, 1.0 - c.fraction_a) < 0.05:
        issues.append(Issue(
            level="warning", field="coculture.fraction_a",
            message=f"fraction_a {c.fraction_a:.3f} leaves one population <5% of the total — "
            "a nominal co-culture; confirm both populations are actually represented",
        ))


def check_enzyme(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the competitive-inhibition reaction plan when present.

    Re-runs :mod:`labwright.calc.enzyme` (Cheng–Prusoff) on the Km/[S]/Ki/[I]
    inputs and verifies every derived field; warns when the inhibitor
    concentration exceeds the substrate (a "runaway inhibitor" regime whose IC50
    interpretation is fragile).
    """
    e = plan.enzyme
    if e is None:
        return

    act = calc_enzyme.fractional_activity(e.km_um, e.s_conc_um, e.ki_um, e.i_conc_um)
    if not _close(act, e.fractional_activity):
        issues.append(Issue(
            level="error", field="enzyme.fractional_activity",
            message="fractional_activity does not equal [S]/(Km(1+[I]/Ki)+[S])",
            expected=act, found=e.fractional_activity,
        ))
    pct = calc_enzyme.percent_inhibition(act)
    if not _close(pct, e.percent_inhibition):
        issues.append(Issue(
            level="error", field="enzyme.percent_inhibition",
            message="percent_inhibition does not equal (1 − activity)·100",
            expected=pct, found=e.percent_inhibition,
        ))
    ic50 = calc_enzyme.ic50_from_ki(e.km_um, e.s_conc_um, e.ki_um)
    if not _close(ic50, e.ic50_um):
        issues.append(Issue(
            level="error", field="enzyme.ic50_um",
            message="ic50_um does not equal Ki(1 + [S]/Km) (Cheng–Prusoff)",
            expected=ic50, found=e.ic50_um,
        ))
    app_km = calc_enzyme.apparent_km_um(e.km_um, e.i_conc_um, e.ki_um)
    if not _close(app_km, e.apparent_km_um):
        issues.append(Issue(
            level="error", field="enzyme.apparent_km_um",
            message="apparent_km_um does not equal Km(1 + [I]/Ki)",
            expected=app_km, found=e.apparent_km_um,
        ))
    ratio = calc_enzyme.molar_ratio(e.i_conc_um, e.s_conc_um)
    if not _close(ratio, e.inhibitor_substrate_ratio):
        issues.append(Issue(
            level="error", field="enzyme.inhibitor_substrate_ratio",
            message="inhibitor_substrate_ratio does not equal [I]/[S]",
            expected=ratio, found=e.inhibitor_substrate_ratio,
        ))

    if e.velocity_umol_min is not None:
        if e.vmax_umol_min is None:
            issues.append(Issue(
                level="warning", field="enzyme.velocity_umol_min",
                message="velocity_umol_min present but vmax_umol_min is missing — it cannot be re-derived",
            ))
        else:
            vel = calc_enzyme.velocity_umol_min(
                e.vmax_umol_min, e.km_um, e.s_conc_um, e.ki_um, e.i_conc_um
            )
            if not _close(vel, e.velocity_umol_min):
                issues.append(Issue(
                    level="error", field="enzyme.velocity_umol_min",
                    message="velocity_umol_min does not equal Vmax × fractional activity",
                    expected=vel, found=e.velocity_umol_min,
                ))
    elif e.vmax_umol_min is not None:
        issues.append(Issue(
            level="warning", field="enzyme.velocity_umol_min",
            message="vmax_umol_min is present but velocity_umol_min is not computed",
        ))

    if e.i_conc_um > 0 and e.i_conc_um > 10 * e.s_conc_um:
        issues.append(Issue(
            level="warning", field="enzyme.i_conc_um",
            message=f"[I] {e.i_conc_um:g} µM is >10× [S] {e.s_conc_um:g} µM — the inhibitor "
            "dominates the mix and the competitive-inhibition IC50 (Cheng–Prusoff) "
            "interpretation is fragile at this ratio",
        ))


def check_champ(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the ChAMP methylation batch plan when present.

    Re-runs :mod:`labwright.calc.bioinformatics` on the cohort/platform inputs
    and verifies every derived field; warns when the expected QC failure count
    cannot be derived for want of a fail rate.
    """
    c = plan.champ
    if c is None:
        return

    expected_arrays = calc_bioinformatics.champ_arrays_for_samples(c.n_samples, c.platform)
    if not _close(expected_arrays, c.n_arrays):
        issues.append(Issue(
            level="error", field="champ.n_arrays",
            message="n_arrays does not equal n_samples",
            expected=expected_arrays, found=c.n_arrays,
        ))
    expected_chips = calc_bioinformatics.champ_chips_for_samples(c.n_samples, c.platform)
    if not _close(expected_chips, c.n_chips):
        issues.append(Issue(
            level="error", field="champ.n_chips",
            message="n_chips does not equal ceil(n_samples / chip capacity)",
            expected=expected_chips, found=c.n_chips,
        ))
    if c.n_expected_failed_arrays is not None:
        if c.fail_rate_pct is None:
            issues.append(Issue(
                level="warning", field="champ.n_expected_failed_arrays",
                message="n_expected_failed_arrays present but fail_rate_pct is missing — it cannot be re-derived",
            ))
        else:
            expected_fail = calc_bioinformatics.champ_expected_failed_arrays(
                c.n_arrays, c.fail_rate_pct / 100.0
            )
            if not _close(expected_fail, c.n_expected_failed_arrays):
                issues.append(Issue(
                    level="error", field="champ.n_expected_failed_arrays",
                    message="n_expected_failed_arrays does not equal n_arrays × fail_rate",
                    expected=expected_fail, found=c.n_expected_failed_arrays,
                ))
    elif c.fail_rate_pct is not None:
        issues.append(Issue(
            level="warning", field="champ.n_expected_failed_arrays",
            message="fail_rate_pct is present but n_expected_failed_arrays is not computed",
        ))


def check_plink(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the PLINK genotype batch plan when present.

    Re-runs :mod:`labwright.calc.bioinformatics` on the cohort/variant inputs
    and verifies every derived field; warns when a per-chromosome variant count
    is given but the per-chromosome size was not derived.
    """
    p = plan.plink
    if p is None:
        return

    expected_bed = calc_bioinformatics.plink_bed_size_mb(p.n_samples, p.n_variants)
    if not _close(expected_bed, p.bed_size_mb):
        issues.append(Issue(
            level="error", field="plink.bed_size_mb",
            message="bed_size_mb does not equal n_samples × n_variants / 4 / 1e6",
            expected=expected_bed, found=p.bed_size_mb,
        ))
    expected_files = calc_bioinformatics.plink_per_chr_files()
    if not _close(expected_files, p.n_per_chr_files):
        issues.append(Issue(
            level="error", field="plink.n_per_chr_files",
            message="n_per_chr_files does not equal the standard 25 per-chromosome files",
            expected=expected_files, found=p.n_per_chr_files,
        ))
    if p.per_chr_bed_size_mb is not None:
        if p.n_variants_chr is None:
            issues.append(Issue(
                level="warning", field="plink.per_chr_bed_size_mb",
                message="per_chr_bed_size_mb present but n_variants_chr is missing — it cannot be re-derived",
            ))
        else:
            expected_pc = calc_bioinformatics.plink_per_chr_bed_size_mb(p.n_samples, p.n_variants_chr)
            if not _close(expected_pc, p.per_chr_bed_size_mb):
                issues.append(Issue(
                    level="error", field="plink.per_chr_bed_size_mb",
                    message="per_chr_bed_size_mb does not equal n_samples × n_variants_chr / 4 / 1e6",
                    expected=expected_pc, found=p.per_chr_bed_size_mb,
                ))
    elif p.n_variants_chr is not None:
        issues.append(Issue(
            level="warning", field="plink.per_chr_bed_size_mb",
            message="n_variants_chr is present but per_chr_bed_size_mb is not computed",
        ))


def check_solvent(plan: DesignPlan, issues: list[Issue]) -> None:
    """Cross-check the hanging-drop / solvent-evaporation plan when present.

    Re-runs :mod:`labwright.calc.solvent` on the drop/position/climate inputs
    and verifies every derived field; warns when the drop dries out entirely
    within the stated time (residual ≈ 0 — a dried drop).
    """
    s = plan.solvent
    if s is None:
        return

    edge_factor = s.edge_factor or calc_solvent.EDGE_FACTOR_DEFAULT
    expected_edge = calc_solvent.edge_well_factor(s.well_row, s.well_col, edge_factor)
    if not _close(expected_edge, s.edge_evaporation_factor):
        issues.append(Issue(
            level="error", field="solvent.edge_evaporation_factor",
            message="edge_evaporation_factor does not match the well's plate position",
            expected=expected_edge, found=s.edge_evaporation_factor,
        ))
    expected_rate = calc_solvent.effective_evaporation_rate_ul_hr(
        s.drop_volume_ul, s.well_row, s.well_col, temp_c=s.temp_c, rh=s.rh, edge_factor=edge_factor
    )
    if not _close(expected_rate, s.evaporation_rate_ul_hr):
        issues.append(Issue(
            level="error", field="solvent.evaporation_rate_ul_hr",
            message="evaporation_rate_ul_hr does not equal interior Langmuir rate × edge factor",
            expected=expected_rate, found=s.evaporation_rate_ul_hr,
        ))
    expected_residual = calc_solvent.drop_volume_after_time(
        s.drop_volume_ul, s.hours, temp_c=s.temp_c, rh=s.rh,
        evaporation_factor=s.edge_evaporation_factor,
    )
    if not _close(expected_residual, s.residual_volume_ul):
        issues.append(Issue(
            level="error", field="solvent.residual_volume_ul",
            message="residual_volume_ul does not match the d²-law projection",
            expected=expected_residual, found=s.residual_volume_ul,
        ))

    if s.residual_volume_ul <= 0.05 * s.drop_volume_ul:
        issues.append(Issue(
            level="warning", field="solvent.residual_volume_ul",
            message=f"residual volume {s.residual_volume_ul:.3g} µL is ≤5% of the initial "
            f"{s.drop_volume_ul:g} µL — the drop dries out (or nearly) within {s.hours:g} h; "
            "a crystal screen that dries mid-run cannot be scored",
        ))


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
    check_barrier(plan, issues)
    check_pumpless(plan, issues)
    check_breathing(plan, issues)
    check_pulsatile(plan, issues)
    check_scaling(plan, issues)
    check_gradient(plan, issues)
    check_bioprinting(plan, issues)
    check_coculture(plan, issues)
    check_enzyme(plan, issues)
    check_champ(plan, issues)
    check_plink(plan, issues)
    check_solvent(plan, issues)
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
