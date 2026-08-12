"""Tests for benchmark metrics (no LLM involved — metrics are pure)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labwright.calc import microfluidics as mf  # noqa: E402
from labwright.design import derive_culture  # noqa: E402
from labwright.schema.design import CulturePlan, DesignPlan  # noqa: E402
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


# --- competitor baselines (no LLM — prompt structure and scoring are pure) ---

from eval.benchmark import (  # noqa: E402
    bare_checkable,
    bare_hallucination,
    bare_prompt_for,
    evaluate,
    run_self_verify,
    run_soft_gate,
    soft_gate_prompt_for,
)

_GOLD = GoldExperiment(
    id="g1",
    goal="Perfuse a 400x100 um channel at 2 uL/min (water-like) and report shear.",
    expected={"shear_pa": 0.05},
    source="self-consistent",
)


def _derived_for(w=400, h=100, L=20, Q=2.0, mu=1e-3, rho=1000.0) -> dict:
    return {
        "shear_pa": mf.wall_shear_stress(Q, w, h, mu),
        "reynolds": mf.reynolds_number(Q, w, h, mu, rho),
        "pressure_drop_pa": mf.pressure_drop(Q, w, h, L, mu),
        "residence_time_s": mf.residence_time(Q, w, h, L),
        "channel_volume_ul": mf.channel_volume(w, h, L),
        "mean_velocity_mms": mf.mean_velocity(Q, w, h),
    }


def test_bare_hallucination_tolerates_zero_flow():
    """A reported flow of 0 (produced with thinking ON by pro on the blind
    retinal goal) must not crash the scorer — it is unverifiable → 1.0."""
    extracted = {
        "width_um": 400, "height_um": 100, "length_mm": 20, "flow_rate_uLmin": 0.0,
        "shear_pa": 0.05,  # claimed, but cannot follow from flow 0
    }
    assert bare_hallucination(extracted) == 1.0


def test_bare_hallucination_tolerates_non_finite_flow():
    extracted = {
        "width_um": 400, "height_um": 100, "length_mm": 20,
        "flow_rate_uLmin": float("inf"), "shear_pa": 0.05,
    }
    assert bare_hallucination(extracted) == 1.0


def test_soft_gate_prompt_demands_self_check():
    prompt = soft_gate_prompt_for(_GOLD)
    assert "re-derive every derived flow number" in prompt
    assert "shear_pa" in prompt


def test_soft_gate_prompt_differs_from_bare_only_by_instruction():
    bare = bare_prompt_for(_GOLD)
    soft = soft_gate_prompt_for(_GOLD)
    # the added instruction is the *only* difference — parsing/scoring stay shared
    assert soft.startswith(bare.split("\n")[0])


def test_run_soft_gate_uses_soft_prompt_and_extracts():
    ok = {
        "width_um": 400, "height_um": 100, "length_mm": 20, "flow_rate_uLmin": 2.0,
        "viscosity_pas": 1e-3, "density_kgm3": 1000.0,
        **_derived_for(), "shear_pa": 0.05,
    }
    seen = []

    def chat(prompt: str) -> str:
        seen.append(prompt)
        return "{" + ",".join(f'"{k}":{v}' for k, v in ok.items()) + "}"

    out = run_soft_gate(_GOLD, chat)
    assert soft_gate_prompt_for(_GOLD) in seen  # the soft prompt, not the bare one
    assert out["shear_pa"] == pytest.approx(0.05)


def test_run_self_verify_merges_stage2_over_stage1():
    stage1 = {
        "width_um": 400, "height_um": 100, "length_mm": 20, "flow_rate_uLmin": 2.0,
        "viscosity_pas": 1e-3, "density_kgm3": 1000.0, "shear_pa": 99.0,  # invented
    }
    stage2 = _derived_for()  # the "verifier" recomputes correctly

    def chat(prompt: str) -> str:
        if "Goal:" in prompt:
            data = stage1
        else:
            data = stage2  # verify pass
        return "{" + ",".join(f'"{k}":{v}' for k, v in data.items()) + "}"

    out = run_self_verify(_GOLD, chat)
    # stage-2 recompute replaces the invented stage-1 number
    assert out["shear_pa"] == pytest.approx(stage2["shear_pa"])
    assert out["shear_pa"] != pytest.approx(99.0)
    # raw inputs survive the merge unchanged
    assert out["width_um"] == pytest.approx(400.0)


def test_run_self_verify_proposal_stands_when_verifier_is_silent():
    def chat(prompt: str) -> str:
        return "I cannot compute this."  # both stages return prose

    out = run_self_verify(_GOLD, chat)
    assert all(v is None for v in out.values())


def test_evaluate_supports_competitor_systems():
    """evaluate() runs each requested system and scores it with bare metrics."""
    stage1 = {
        "width_um": 400, "height_um": 100, "length_mm": 20, "flow_rate_uLmin": 2.0,
        "viscosity_pas": 1e-3, "density_kgm3": 1000.0, "shear_pa": 0.05,
    }

    def chat(prompt: str) -> str:
        data = stage1 if "Goal:" in prompt else _derived_for()
        return "{" + ",".join(f'"{k}":{v}' for k, v in data.items()) + "}"

    def agent_factory():
        raise AssertionError("labwright should not run when excluded")

    summary = evaluate([_GOLD], agent_factory, chat,
                       systems=("bare", "soft_gate", "self_verify"))
    for sys in ("bare", "soft_gate", "self_verify"):
        assert sys in summary
        assert summary[sys]["usable_design_rate"] == 1.0
        assert summary[sys]["hallucination_rate"] == 0.0
    assert summary["per_entry"][0]["bare"]["valid"] is True
    assert "labwright" not in summary


# --- plate-culture domain: bare metrics are culture-aware, not flow-only ---

_CULTURE_GOLD = GoldExperiment(
    id="c1",
    goal="Seed a 96-well plate at 1e4 cells/cm^2; report cells per well and "
    "the standard medium volume per well.",
    expected={"seed_per_well": 3200.0, "medium_volume_per_well_ml": 0.17},
    source="Corning plate table",
)


def test_bare_prompt_for_culture_uses_plate_keys():
    prompt = bare_prompt_for(_CULTURE_GOLD)
    assert "plate_format" in prompt
    assert "seed_per_well" in prompt
    assert "width_um" not in prompt  # no flow keys for a culture goal


def test_bare_hallucination_culture_consistent():
    extracted = {
        "plate_format": "96", "seeding_density_cells_cm2": 1e4, "wells": 1,
        "seed_per_well": 3200.0, "medium_volume_per_well_ml": 0.17,
    }
    assert bare_hallucination(extracted) == 0.0
    assert bare_checkable(extracted) is True


def test_bare_hallucination_culture_wrong():
    extracted = {
        "plate_format": "96", "seeding_density_cells_cm2": 1e4, "wells": 1,
        "seed_per_well": 9999.0,  # does not equal 1e4 × 0.32
    }
    assert bare_hallucination(extracted) > 0


def test_bare_hallucination_culture_no_derived_is_unverifiable():
    # plate+density but no culture derived number → nothing checkable → 1.0
    extracted = {"plate_format": "96", "seeding_density_cells_cm2": 1e4}
    assert bare_hallucination(extracted) == 1.0
    assert bare_checkable(extracted) is False


def test_parameter_recovery_culture_exact():
    plan = _verified_plan()
    plan.culture = CulturePlan(**derive_culture(
        dict(plate_format="96", wells=1, cell_type="HepG2",
             seeding_density_cells_cm2=1e4)
    ))
    errs = parameter_recovery(_CULTURE_GOLD, plan)
    assert errs["seed_per_well"] == pytest.approx(0.0)
    assert errs["medium_volume_per_well_ml"] == pytest.approx(0.0)


def test_hallucination_rate_ignores_culture_when_absent():
    # A chip-only plan must not be scored against culture fields.
    plan = _verified_plan(shear=1.0)  # flow error present
    plan.culture = None
    rate = hallucination_rate(plan)
    assert 0 < rate < 1  # flow fields counted, culture fields not in denominator
