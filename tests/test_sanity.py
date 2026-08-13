"""Tests for the physiological-range layer.

The property under test: a value inside the physiological band passes silently,
a value in the soft band warns, and a value outside the hard band errors. The
layer exists so out-of-physiology numbers are never silently passed even when
the arithmetic is perfect.
"""

import pytest

from labwright.design import build_design, DesignInput
from labwright.verify.checker import verify_design
from labwright.verify.sanity import check_sanity, SANITY_BANDS


def _plan(**overrides):
    raw = dict(
        goal="test", rationale="r",
        chip=dict(width_um=400, height_um=100, length_mm=20),
        flow=dict(flow_rate_uLmin=10, viscosity_pas=1e-3),
        cells=dict(cell_type="HepG2", seeding_density_cells_cm2=1e5, culture_area_cm2=0.08),
    )
    raw.update(overrides)
    return build_design(DesignInput(**raw))


def _issues(plan):
    return verify_design(plan)


def _levels_for(field, issues):
    return [i.level for i in issues if i.field == field]


def test_physiological_shear_passes_silently():
    issues = _issues(_plan())
    assert "derived.shear_pa" not in {i.field for i in issues}


def test_hard_physics_violation_errors():
    # 500 Pa of shear is not buildable/culturable as stated.
    plan = _plan(flow=dict(flow_rate_uLmin=10000, viscosity_pas=1e-3))
    issues = _issues(plan)
    assert "error" in _levels_for("derived.shear_pa", issues)


def test_reynolds_over_laminar_errors():
    # A "microchannel" that goes turbulent (Re >> 2300) is a hard error.
    plan = _plan(flow=dict(flow_rate_uLmin=200000, viscosity_pas=1e-3))
    issues = _issues(plan)
    assert "error" in _levels_for("derived.reynolds", issues)


def test_soft_band_warns_but_does_not_error():
    # Pathological-but-buildable shear (20 Pa) is a warning, not an error.
    # At this geometry+μ the 0.25 Pa probe sits at 10 µL/min and shear ∝ Q.
    plan = _plan(flow=dict(flow_rate_uLmin=800, viscosity_pas=1e-3))  # → 20 Pa
    issues = _issues(plan)
    assert "warning" in _levels_for("derived.shear_pa", issues)
    assert "error" not in _levels_for("derived.shear_pa", issues)


def test_absurd_seeding_density_errors():
    # 1e10 cells/cm² is beyond the hard physical band (1e9).
    plan = _plan(cells=dict(cell_type="HepG2", seeding_density_cells_cm2=1e10, culture_area_cm2=0.08))
    issues = _issues(plan)
    assert "error" in _levels_for("cells.seeding_density_cells_cm2", issues)


def test_over_confluent_harvest_is_warning_not_error():
    # The schema explicitly allows confluence > 100 % (over-confluent harvest);
    # sanity must warn, never error, on it.
    plan = build_design(DesignInput(
        goal="test", rationale="r",
        culture=dict(
            plate_format="24", wells=1, cell_type="HepG2",
            seeding_density_cells_cm2=5e4, confluent_density_cells_cm2=1e5,
            doubling_time_h=30, culture_duration_h=120,
        ),
    ))
    issues = _issues(plan)
    assert "error" not in _levels_for("culture.expected_confluence_pct", issues)


def test_all_bands_declared():
    for field in (
        "derived.shear_pa", "derived.reynolds", "derived.pressure_drop_pa",
        "derived.residence_time_s", "derived.channel_volume_ul",
        "derived.mean_velocity_mms", "cells.seed_count",
        "culture.seed_per_well", "culture.total_seed_count",
        "culture.medium_volume_per_well_ml", "culture.total_medium_ml",
        "culture.expected_confluence_pct", "dosing.working_mM",
        "dosing.dmso_fraction_vv", "stats.n_per_group",
    ):
        assert field in SANITY_BANDS, field
