"""Tests for SOP rendering — output must be deterministic and number-complete."""

import re

from labwright.design import build_design, DesignInput
from labwright.sop.render import design_to_sop
from labwright.verify.checker import Issue, verify_design


def _plan():
    inp = DesignInput(
        goal="Model drug-induced liver injury in a perfused liver chip",
        rationale="test",
        chip={"width_um": 400, "height_um": 100, "length_mm": 20, "channel_count": 1},
        flow={"flow_rate_uLmin": 2, "viscosity_pas": 1e-3, "density_kgm3": 1000},
        cells={
            "cell_type": "HepG2",
            "seeding_density_cells_cm2": 1e5,
            "culture_area_cm2": 0.08,
            "doubling_time_h": 35,
            "culture_duration_h": 72,
        },
        dosing={
            "compound": "Acetaminophen",
            "molecular_weight_g_mol": 151.16,
            "stock_mM": 100,
            "working_mM": 0.1,
            "vehicle_control": True,
            "exposure_h": 24,
        },
        stats={"effect_size": 1.0, "std_dev": 1.0, "alpha": 0.05, "power": 0.80},
        caveats=["confirm shear from literature"],
    )
    return build_design(inp)


def test_sop_contains_all_sections():
    md = design_to_sop(_plan())
    for heading in ["## 1. Device", "## 2. Perfusion", "## 3. Cell seeding", "## 4. Compound", "## 5. Statistical", "## 6. Caveats"]:
        assert heading in md


def test_sop_numbers_match_plan():
    plan = _plan()
    md = design_to_sop(plan)
    # Shear must appear exactly as computed
    assert f"{plan.derived.shear_pa:.3f}" in md
    assert f"{plan.derived.channel_volume_ul:.2f}" in md
    assert f"{plan.cells.seed_count:g}" in md
    assert f"{plan.stats.n_per_group}" in md


def test_sop_deterministic():
    assert design_to_sop(_plan()) == design_to_sop(_plan())


def test_sop_attributes_to_calculators():
    md = design_to_sop(_plan())
    assert "deterministic calculators" in md
    assert "language model proposed only the raw inputs" in md


def test_sop_refuses_design_with_verification_errors():
    """A design with unresolved errors is not a followable protocol."""
    plan = _plan()
    plan.derived.shear_pa *= 10.0  # a hand-edited derived number
    issues = verify_design(plan)
    md = design_to_sop(plan, issues)
    assert "Not verified" in md
    assert "derived.shear_pa" in md
    assert "Do not follow this SOP" in md
    # the protocol body must not ship when errors exist
    assert "## 2. Perfusion" not in md


def test_sop_surfaces_warnings_when_issues_passed():
    """Verifier warnings (incl. prose contradictions) reach the SOP, not just the CLI."""
    plan = _plan()
    issues = verify_design(plan)
    issues.append(Issue(level="warning", field="prose",
                        message="number 0.5 (Pa) in the design text matches no value in this design"))
    md = design_to_sop(plan, issues)
    assert "Verification warnings" in md
    assert "prose" in md


def test_sop_provenance_status_renders_real_verdict():
    """The audit trail shows the verifier's real per-field verdicts, not "ok"."""
    plan = _plan()
    issues = verify_design(plan)
    issues.append(Issue(level="warning", field="prose",
                        message="number 0.5 (Pa) matches no value in this design"))
    md = design_to_sop(plan, issues)
    # the verified shear still carries its true "ok" verdict — issues are threaded
    assert "verify: ok" in md
    assert "derived.shear_pa" in md


def test_sop_reynolds_note_derived_from_computed_re():
    """The laminar claim must follow the computed Reynolds number, not be asserted."""
    plan = _plan()
    md = design_to_sop(plan)
    re_val = plan.derived.reynolds
    assert re_val < 100, "test design should be deep-laminar"
    assert "(laminar, Re << 2300)" in md
