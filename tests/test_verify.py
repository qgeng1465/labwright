"""Tests for the verification layer — Labwright's anti-hallucination core.

The key property under test: given a design plan whose derived numbers were
*invented* (as an LLM would), the checker must flag them; given a plan whose
derived numbers came from the calculators, it must pass.
"""

import pytest

from labwright.calc import microfluidics as mf
from labwright.design import derive_culture
from labwright.schema.design import CulturePlan, DesignPlan
from labwright.verify.checker import has_errors, verify_design


def _make_plan(*, corrupt: str | None = None) -> DesignPlan:
    """Build a fully-verified design plan, optionally corrupting one field."""
    chip = dict(width_um=400, height_um=100, length_mm=20, channel_count=1)
    flow = dict(flow_rate_uLmin=10, viscosity_pas=1e-3, density_kgm3=1000)
    derived = dict(
        shear_pa=mf.wall_shear_stress(10, 400, 100, 1e-3),
        reynolds=mf.reynolds_number(10, 400, 100, 1e-3),
        pressure_drop_pa=mf.pressure_drop(10, 400, 100, 20, 1e-3),
        residence_time_s=mf.residence_time(10, 400, 100, 20),
        channel_volume_ul=mf.channel_volume(400, 100, 20),
        mean_velocity_mms=mf.mean_velocity(10, 400, 100),
    )
    cells = dict(cell_type="HepG2", seeding_density_cells_cm2=1e5, culture_area_cm2=0.08, seed_count=8000)
    dosing = dict(
        compound="Acetaminophen",
        molecular_weight_g_mol=151.16,
        stock_mM=100,
        working_mM=0.1,
        dmso_fraction_vv=0.001,
    )
    stats = dict(effect_size=1.0, std_dev=1.0, alpha=0.05, power=0.80, n_per_group=16)
    # A valid plate-culture plan: derived fields produced by the calculators.
    culture = derive_culture(
        dict(
            plate_format="96",
            wells=4,
            cell_type="HepG2",
            seeding_density_cells_cm2=1e4,
            viability_pct=90,
            confluent_density_cells_cm2=1e6,
            doubling_time_h=30,
            culture_duration_h=72,
        )
    )

    if corrupt == "shear":
        derived["shear_pa"] *= 10  # a hallucinated shear stress
    if corrupt == "seeding":
        cells["seed_count"] = 12345  # does not equal density × area
    if corrupt == "dmso":
        dosing["dmso_fraction_vv"] = 0.05  # does not equal 0.1/100
    if corrupt == "culture_seed":
        culture["seed_per_well"] = 9999  # does not equal density × well area
    if corrupt == "culture_conf":
        culture["expected_confluence_pct"] = 90.0  # does not match growth prediction

    return DesignPlan(
        goal="Model drug-induced liver injury in a perfused liver chip",
        rationale="test plan",
        chip=chip,
        flow=flow,
        derived=derived,
        cells=cells,
        culture=CulturePlan(**culture),
        dosing=dosing,
        stats=stats,
    )


def test_valid_plan_passes():
    issues = verify_design(_make_plan())
    assert issues == []
    assert not has_errors(issues)


def test_hallucinated_shear_is_caught():
    issues = verify_design(_make_plan(corrupt="shear"))
    assert has_errors(issues)
    assert any("shear_pa" in i.field for i in issues)


def test_wrong_seeding_is_caught():
    issues = verify_design(_make_plan(corrupt="seeding"))
    assert has_errors(issues)
    assert any("seed_count" in i.field for i in issues)


def test_wrong_dmso_is_caught():
    issues = verify_design(_make_plan(corrupt="dmso"))
    assert has_errors(issues)
    assert any("dmso_fraction_vv" in i.field for i in issues)


def test_dmso_toxicity_warning():
    # 5% DMSO (working 5 mM from 100 mM stock) is far above the 0.5% threshold
    plan = _make_plan()
    plan.dosing.working_mM = 5
    plan.dosing.dmso_fraction_vv = 0.05
    issues = verify_design(plan)
    dmso_issues = [i for i in issues if i.field == "dosing.dmso_fraction_vv"]
    assert any(i.level == "warning" for i in dmso_issues)


def test_hallucinated_culture_seed_is_caught():
    issues = verify_design(_make_plan(corrupt="culture_seed"))
    assert has_errors(issues)
    assert any("culture.seed_per_well" in i.field for i in issues)


def test_wrong_predicted_confluence_is_caught():
    issues = verify_design(_make_plan(corrupt="culture_conf"))
    assert has_errors(issues)
    assert any("culture.expected_confluence_pct" in i.field for i in issues)


def test_culture_without_plan_is_skipped():
    # A chip-only design (no plate culture) must not trip the culture checker.
    plan = _make_plan()
    plan.culture = None
    issues = verify_design(plan)
    assert not any(i.field.startswith("culture.") for i in issues)


def test_over_confluent_harvest_warns():
    # 400 h at 30 h doubling from 3200 cells vastly overshoots 100% confluence.
    culture = derive_culture(
        dict(
            plate_format="96",
            wells=4,
            cell_type="HepG2",
            seeding_density_cells_cm2=1e4,
            confluent_density_cells_cm2=1e6,
            doubling_time_h=30,
            culture_duration_h=400,
        )
    )
    assert culture["expected_confluence_pct"] > 100
    plan = _make_plan()
    plan.culture = CulturePlan(**culture)
    issues = verify_design(plan)
    conf_issues = [i for i in issues if i.field == "culture.expected_confluence_pct"]
    assert any(i.level == "warning" for i in conf_issues)


def test_low_viability_warns():
    plan = _make_plan()
    plan.culture.viability_pct = 55
    issues = verify_design(plan)
    assert any(i.level == "warning" and "viability" in i.field for i in issues)


def test_format_issues_empty():
    assert "verified" in __import__("labwright.verify.checker", fromlist=["format_issues"]).format_issues([])
