"""Tests for the safety & compliance layer.

The layer's contract: an out-of-bound dose is flagged (warning) or rejected
with a reason (error), a missing vehicle control is caught, BSL-2 material is
hinted, and the whole boundary is configurable per institution — never a hard
literal. Checks are exercised through :func:`check_safety` directly (isolated
from the sanity layer) plus one end-to-end pass through :func:`verify_design`.
"""

import pytest

from labwright.design import DesignInput, build_design
from labwright.verify.checker import verify_design
from labwright.verify.safety import (
    SafetyConfig,
    biosafety_level_for,
    check_safety,
    get_safety_config,
    load_safety_config,
    reset_safety_config,
    set_safety_config,
)


def _dosing_plan(*, compound="Acetaminophen", stock_mM=2000.0, working_mM=0.001,
                 vehicle_control=True, cell_type="HepG2"):
    """A dosing plan built through the real pipeline (derives dmso_fraction)."""
    cells = dict(cell_type=cell_type, seeding_density_cells_cm2=1e5, culture_area_cm2=0.08)
    dosing = dict(
        compound=compound, molecular_weight_g_mol=151.16,
        stock_mM=stock_mM, working_mM=working_mM, vehicle_control=vehicle_control,
    )
    return build_design(DesignInput(goal="toxicity study", rationale="r", cells=cells, dosing=dosing))


def _fields_of(issues):
    return [i.field for i in issues]


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    reset_safety_config()


def test_clean_dose_passes_silently():
    plan = _dosing_plan(compound="Acetaminophen", working_mM=0.001)
    issues = []
    check_safety(plan, issues)
    assert issues == []


def test_dmso_over_boundary_warns():
    # working 1.0 / stock 100 = 1% v/v — over the 0.5% boundary.
    plan = _dosing_plan(working_mM=1.0, stock_mM=100.0)
    issues = []
    check_safety(plan, issues)
    assert [i.level for i in issues if i.field == "dosing.dmso_fraction_vv"] == ["warning"]
    assert any("solvent toxicity" in i.message for i in issues)


def test_dmso_at_or_below_boundary_silent():
    # 0.5% exactly (boundary) is allowed; 0.0025% obviously.
    plan = _dosing_plan(working_mM=0.5, stock_mM=100.0)  # 0.5 % v/v
    issues = []
    check_safety(plan, issues)
    assert "dosing.dmso_fraction_vv" not in _fields_of(issues)


def test_compound_over_reject_cap_errors_with_reason():
    # Doxorubicin at 5 mM is far beyond the hard cap (0.5 mM) → rejected.
    plan = _dosing_plan(compound="Doxorubicin", working_mM=5.0, stock_mM=2000.0)
    issues = []
    check_safety(plan, issues)
    assert [i.level for i in issues if i.field == "dosing.working_mM"] == ["error"]
    assert any("rejected" in i.message for i in issues)


def test_compound_over_guidance_warns():
    # Doxorubicin at 50 µM is above the 10 µM guidance cap → warning, not error.
    plan = _dosing_plan(compound="Doxorubicin", working_mM=0.05, stock_mM=2000.0)
    issues = []
    check_safety(plan, issues)
    assert [i.level for i in issues if i.field == "dosing.working_mM"] == ["warning"]


def test_alias_of_compound_is_recognised():
    # "paracetamol" is a documented alias of acetaminophen.
    plan = _dosing_plan(compound="paracetamol", working_mM=50.0, stock_mM=2000.0)
    issues = []
    check_safety(plan, issues)
    assert [i.level for i in issues if i.field == "dosing.working_mM"] == ["warning"]


def test_missing_vehicle_control_warns():
    plan = _dosing_plan(working_mM=0.001, vehicle_control=False)
    issues = []
    check_safety(plan, issues)
    assert "dosing.vehicle_control" in _fields_of(issues)


def test_vehicle_control_present_silent():
    plan = _dosing_plan(working_mM=0.001, vehicle_control=True)
    issues = []
    check_safety(plan, issues)
    assert "dosing.vehicle_control" not in _fields_of(issues)


def test_biosafety_hint_for_hela():
    assert biosafety_level_for("HeLa")[0] == 2
    assert biosafety_level_for("HepG2")[0] == 1
    assert biosafety_level_for("PHH")[0] == 2
    assert biosafety_level_for("primary mouse hepatocytes")[0] == 2


def test_bsl2_and_ethics_warn_on_cell_type():
    plan = _dosing_plan(cell_type="primary mouse hepatocytes")
    issues = []
    check_safety(plan, issues)
    fields = _fields_of(issues)
    assert "cells.cell_type" in fields
    # Both the containment hint and the animal-ethics reminder fire.
    joined = " | ".join(i.message for i in issues)
    assert "containment" in joined
    assert "animal-ethics" in joined


def test_biosafety_hint_can_be_disabled():
    plan = _dosing_plan(cell_type="HeLa")
    issues = []
    check_safety(plan, issues, config=SafetyConfig(biosafety_hints=False))
    assert "cells.cell_type" not in _fields_of(issues)


def test_configurable_dmso_boundary():
    # A lab that routinely runs 1% DMSO tightens/relaxes the boundary in config.
    set_safety_config(SafetyConfig(max_dmso_vv=0.02))
    plan = _dosing_plan(working_mM=1.0, stock_mM=100.0)  # 1% v/v
    issues = []
    check_safety(plan, issues)
    assert "dosing.dmso_fraction_vv" not in _fields_of(issues)


def test_load_safety_config_from_json(tmp_path):
    cfg_path = tmp_path / "safety.json"
    cfg_path.write_text('{"max_dmso_vv": 0.02, "institution": "C-301"}')
    loaded = load_safety_config(str(cfg_path))
    assert loaded.max_dmso_vv == pytest.approx(0.02)
    assert get_safety_config().institution == "C-301"


def test_institution_note_rides_on_findings():
    set_safety_config(SafetyConfig(institution="C-301"))
    plan = _dosing_plan(working_mM=1.0, stock_mM=100.0)  # over the 0.5% boundary
    issues = []
    check_safety(plan, issues)
    assert issues and all("[C-301]" in i.message for i in issues)


def test_wired_into_verify_design():
    # End-to-end: verify_design surfaces the safety error for a rejected dose.
    plan = _dosing_plan(compound="Doxorubicin", working_mM=5.0, stock_mM=2000.0)
    issues = verify_design(plan)
    assert any(i.field == "dosing.working_mM" and i.level == "error" for i in issues)
