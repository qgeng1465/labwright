"""Tests for the verification layer — Labwright's anti-hallucination core.

The key property under test: given a design plan whose derived numbers were
*invented* (as an LLM would), the checker must flag them; given a plan whose
derived numbers came from the calculators, it must pass.
"""

import pytest

from labwright.calc import microfluidics as mf
from labwright.schema.design import DesignPlan
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

    if corrupt == "shear":
        derived["shear_pa"] *= 10  # a hallucinated shear stress
    if corrupt == "seeding":
        cells["seed_count"] = 12345  # does not equal density × area
    if corrupt == "dmso":
        dosing["dmso_fraction_vv"] = 0.05  # does not equal 0.1/100

    return DesignPlan(
        goal="Model drug-induced liver injury in a perfused liver chip",
        rationale="test plan",
        chip=chip,
        flow=flow,
        derived=derived,
        cells=cells,
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


def test_format_issues_empty():
    assert "verified" in __import__("labwright.verify.checker", fromlist=["format_issues"]).format_issues([])
