"""Tests for benchmark metrics (no LLM involved — metrics are pure)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labwright.calc import microfluidics as mf  # noqa: E402
from labwright.schema.design import DesignPlan  # noqa: E402
from eval.benchmark import GoldExperiment, hallucination_rate, parameter_recovery  # noqa: E402


def _verified_plan(shear: float | None = None) -> DesignPlan:
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
    if shear is not None:
        derived["shear_pa"] = shear  # hallucinated shear
    return DesignPlan(
        goal="test", rationale="r", chip=chip, flow=flow, derived=derived,
        cells={"cell_type": "X", "seeding_density_cells_cm2": 1e5, "culture_area_cm2": 0.08, "seed_count": 8000},
    )


def test_hallucination_rate_zero_for_verified():
    assert hallucination_rate(_verified_plan()) == 0.0


def test_hallucination_rate_high_for_invented_shear():
    rate = hallucination_rate(_verified_plan(shear=1.0))  # 4x off from the true 0.25 Pa
    assert rate > 0


def test_parameter_recovery_exact():
    gold = GoldExperiment(id="x", goal="g", expected={"shear_pa": 0.25}, source="s")
    errs = parameter_recovery(gold, _verified_plan(shear=0.25))
    assert errs["shear_pa"] == pytest.approx(0.0)


def test_parameter_recovery_relative_error():
    gold = GoldExperiment(id="x", goal="g", expected={"shear_pa": 0.25}, source="s")
    errs = parameter_recovery(gold, _verified_plan(shear=0.5))  # 2x high
    assert errs["shear_pa"] == pytest.approx(1.0)
