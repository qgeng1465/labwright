"""Tests for SOP rendering — output must be deterministic and number-complete."""

import re

from labwright.design import build_design, DesignInput
from labwright.sop.render import design_to_sop


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
