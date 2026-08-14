"""Tests for benchmark metrics (no LLM involved — metrics are pure)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labwright.calc import microfluidics as mf  # noqa: E402
from labwright.design import derive_culture  # noqa: E402
from labwright.schema.design import CulturePlan, DesignPlan  # noqa: E402
from eval.benchmark import (  # noqa: E402
    GoldExperiment,
    _design_claimed,
    hallucination_rate,
    parameter_recovery,
    relative_error,
)


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


# --- zero-valued targets (post-v1 domains: OSI 0, no anoxic core) ---


def test_relative_error_zero_target():
    # A target of exactly 0 (unidirectional flow, necrotic_fraction 0) has no
    # relative denominator; the absolute deviation is the error.
    assert relative_error(0.0, 0.0) == 0.0
    assert relative_error(0.001, 0.0) == pytest.approx(0.001)
    assert relative_error(None, 0.0) == float("inf")


def _new_domain_plan() -> DesignPlan:
    from labwright.design import submit_design

    payload = {"goal": "rocking chip", "rationale": "ok", "caveats": [],
               "pumpless": {
                   "cell_type": "HepG2", "tilt_angle_deg": 15.0,
                   "channel_length_mm": 30.0, "width_um": 700.0, "height_um": 150.0,
                   "rocking_half_period_s": 20.0, "backward_shear_fraction": 1.0}}
    return DesignPlan(**submit_design(payload)["design"])


def test_new_domain_recovery_and_claimed():
    plan = _new_domain_plan()
    gold = GoldExperiment(
        id="pumpless", goal="rocking chip", source="s",
        expected={
            "peak_wall_shear_pa": plan.pumpless.peak_wall_shear_pa,
            "oscillatory_shear_index": plan.pumpless.oscillatory_shear_index,
            "cycles_per_hour": plan.pumpless.cycles_per_hour,
        },
    )
    errs = parameter_recovery(gold, plan)
    assert all(v == pytest.approx(0.0) for v in errs.values())
    claimed = _design_claimed(plan, gold)
    assert set(claimed) == set(gold.expected)
    assert claimed["peak_wall_shear_pa"] == pytest.approx(plan.pumpless.peak_wall_shear_pa)


def test_new_domain_zero_target_scored():
    plan = _new_domain_plan()
    gold = GoldExperiment(
        id="pumpless", goal="rocking chip", source="s",
        expected={"oscillatory_shear_index": 0.0},  # a target that is exactly 0
    )
    # plan's OSI is 0.5 (symmetric rocking) -> error = 0.5 > 0.05 -> not usable
    errs = parameter_recovery(gold, plan)
    assert errs["oscillatory_shear_index"] == pytest.approx(0.5)


# --- competitor baselines (no LLM — prompt structure and scoring are pure) ---

from eval.benchmark import (  # noqa: E402
    _find_key,
    _find_str_key,
    _is_spheroid_gold,
    bare_checkable,
    bare_hallucination,
    bare_prompt_for,
    evaluate,
    run_bare_llm,
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


# --- fine-tuned extractor as a design-benchmark system (the fast path) ---


def _flow_plan_for(gold):
    """A verified DesignPlan for the flow _GOLD, as the extractor would build it."""
    from labwright.design import DesignInput, build_design
    from labwright.schema.design import ChipGeometry, FlowParams

    return build_design(DesignInput(
        goal=gold.goal,
        rationale="mock extractor",
        chip=ChipGeometry(width_um=400, height_um=100, length_mm=20),
        flow=FlowParams(flow_rate_uLmin=2.0, viscosity_pas=1e-3, density_kgm3=1000.0),
    ))


class _MockExtractor:
    """A fake Extractor whose extract_plan returns a fixed plan / failure."""

    def __init__(self, plan=None, error=None):
        self._plan = plan
        self._error = error

    def extract_plan(self, goal):
        if self._plan is not None:
            return self._plan, [], None
        return None, None, self._error or "unparseable_json"


def test_run_finetuned_fast_path_contract():
    """run_finetuned returns (plan, None) on a clean extraction and (None, reason)
    on a failure — the same contract as run_labwright, so a silent extractor
    refusal is scored as an auditable error, not an unexplained blank."""
    from eval.benchmark import run_finetuned

    good = _MockExtractor(plan=_flow_plan_for(_GOLD))
    plan, error = run_finetuned(_GOLD, good)
    assert plan is not None and error is None

    broken = _MockExtractor(plan=None, error="unparseable_json")
    plan, error = run_finetuned(_GOLD, broken)
    assert plan is None and error == "unparseable_json"


def test_evaluate_supports_finetuned_system():
    """evaluate() scores the fine-tuned extractor with the design-path rules —
    the same usable/hallucination convention as Labwright. A valid extraction is
    usable (hallucination 0, target recovered); a failed extraction is scored as
    unverifiable (hallucination 1.0, not usable)."""
    summary = evaluate([_GOLD], agent_factory=None, chat=None,
                       systems=("finetuned",),
                       extractor=_MockExtractor(plan=_flow_plan_for(_GOLD)))
    assert summary["finetuned"]["usable_design_rate"] == 1.0
    assert summary["finetuned"]["hallucination_rate"] == 0.0
    assert summary["per_entry"][0]["finetuned"]["valid"] is True

    broken = evaluate([_GOLD], agent_factory=None, chat=None,
                      systems=("finetuned",),
                      extractor=_MockExtractor(plan=None, error="unparseable_json"))
    assert broken["finetuned"]["usable_design_rate"] == 0.0
    assert broken["finetuned"]["hallucination_rate"] == 1.0
    assert broken["per_entry"][0]["finetuned"]["plan"] is False
    assert broken["per_entry"][0]["finetuned"]["error"] == "unparseable_json"


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


def test_bare_hallucination_culture_confluence_without_growth_inputs_no_crash():
    # A model over-reporting expected_confluence_pct without the growth inputs
    # that produce it must not crash the scorer (regression: KeyError).
    extracted = {
        "plate_format": "96", "seeding_density_cells_cm2": 1e4, "wells": 1,
        "seed_per_well": 3200.0, "expected_confluence_pct": 42.0,  # unsupported
    }
    assert bare_hallucination(extracted) == 0.0  # confluence not counted, seed is right
    # confluence WITH its inputs is cross-checked: at 0 h, 3200 cells on a
    # 96-well (0.32 cm²) at 1e5 cells/cm² confluence = 3200 / (1e5 × 0.32) = 10 %
    extracted["confluent_density_cells_cm2"] = 1e5
    extracted["doubling_time_h"] = 24.0
    extracted["culture_duration_h"] = 0.0
    extracted["expected_confluence_pct"] = 10.0
    assert bare_hallucination(extracted) == 0.0
    extracted["expected_confluence_pct"] = 99.0  # wrong → now counted wrong
    assert bare_hallucination(extracted) > 0


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


# --- 3D-spheroid domain: bare metrics are spheroid-aware, not flow/culture-only ---

_SPHEROID_GOLD = GoldExperiment(
    id="s1",
    goal="Form one HepG2 spheroid per well in a 96-well ULA plate at 1000 "
    "cells/spheroid (20 um cells); report the expected diameter and the total "
    "cells for a full plate.",
    expected={"expected_diameter_um": 200.0, "cells_total": 96000.0},
    source="self-consistent (solid-sphere packing); 96-ULA one spheroid/well",
)


def test_is_spheroid_gold_routing():
    from eval.benchmark import _is_spheroid_gold

    assert _is_spheroid_gold(_SPHEROID_GOLD)
    assert not _is_spheroid_gold(_CULTURE_GOLD)


def test_bare_prompt_for_spheroid_uses_spheroid_keys():
    from eval.benchmark import _prompt_keys_for

    prompt = bare_prompt_for(_SPHEROID_GOLD)
    assert "spheroid_format" in prompt
    assert "expected_diameter_um" in prompt
    assert "width_um" not in prompt  # no flow keys for a spheroid goal
    keys = _prompt_keys_for(_SPHEROID_GOLD)
    assert {"spheroid_format", "spheroid_count", "cells_per_spheroid",
            "cell_diameter_um"} <= set(keys)


def test_bare_hallucination_spheroid_consistent():
    extracted = {
        "spheroid_format": "96-ula", "spheroid_count": 96.0,
        "cells_per_spheroid": 1000.0, "cell_diameter_um": 20.0,
        "expected_diameter_um": 200.0, "spheroid_volume_ul": 4.18879e-3,
        "cells_total": 96000.0, "medium_volume_per_spheroid_ul": 100.0,
        "total_medium_ml": 9.6,
    }
    assert bare_hallucination(extracted) == 0.0
    assert bare_checkable(extracted) is True


def test_bare_hallucination_spheroid_wrong():
    extracted = {
        "spheroid_format": "96-ula", "spheroid_count": 96.0,
        "cells_per_spheroid": 1000.0, "cell_diameter_um": 20.0,
        "cells_total": 50000.0,  # does not equal 96 × 1000
    }
    assert bare_hallucination(extracted) > 0


def test_bare_hallucination_spheroid_no_derived_is_unverifiable():
    # format + cells + count but no spheroid derived number → nothing checkable
    extracted = {"spheroid_format": "96-ula", "spheroid_count": 96.0,
                 "cells_per_spheroid": 1000.0}
    assert bare_hallucination(extracted) == 1.0
    assert bare_checkable(extracted) is False


def test_bare_hallucination_spheroid_bad_format_is_unverifiable():
    # an unrecognised spheroid_format cannot be cross-checked → 1.0
    extracted = {"spheroid_format": "petri-dish", "spheroid_count": 96.0,
                 "cells_per_spheroid": 1000.0, "expected_diameter_um": 200.0}
    assert bare_hallucination(extracted) == 1.0


def test_bare_hallucination_spheroid_geometry_without_vessel_format():
    # A geometry-only answer (volume / diameter from cells × cell size) is
    # checkable even when the model never names a vessel format — a model that
    # writes "solid_sphere" instead of "96-ula" is not penalised for the volume
    # and diameter it derived correctly.
    extracted = {
        "cells_per_spheroid": 1000.0, "cell_diameter_um": 20.0,
        "expected_diameter_um": 200.0, "spheroid_volume_ul": 4.18879e-3,
    }
    assert bare_checkable(extracted) is True
    assert bare_hallucination(extracted) == 0.0


def test_bare_hallucination_spheroid_unparseable_vessel_number_not_counted():
    # A vessel number reported with an unparseable format cannot be re-derived
    # from the model's own raws, so it is neither counted right nor wrong; the
    # geometry fields are still cross-checked.
    extracted = {
        "cells_per_spheroid": 1000.0, "cell_diameter_um": 20.0,
        "expected_diameter_um": 200.0, "spheroid_format": "solid_sphere",
        "medium_volume_per_spheroid_ul": 9999.0,  # unsupported — not counted
    }
    assert bare_hallucination(extracted) == 0.0


def test_bare_hallucination_spheroid_vessel_only_with_bad_format_is_unverifiable():
    # The only reported derived number is a vessel volume, but the vessel format
    # is unparseable → nothing cross-checkable → 1.0.
    extracted = {
        "cells_per_spheroid": 1000.0, "cell_diameter_um": 20.0,
        "spheroid_format": "solid_sphere", "medium_volume_per_spheroid_ul": 100.0,
    }
    assert bare_hallucination(extracted) == 1.0


def test_parameter_recovery_spheroid_exact():
    from labwright.design import derive_spheroid
    from labwright.schema.design import SpheroidPlan

    plan = _verified_plan()
    plan.spheroid = SpheroidPlan(**derive_spheroid(
        dict(cell_type="HepG2", spheroid_format="96-ula", spheroid_count=96,
             cells_per_spheroid=1000.0, cell_diameter_um=20.0)
    ))
    errs = parameter_recovery(_SPHEROID_GOLD, plan)
    assert errs["expected_diameter_um"] == pytest.approx(0.0)
    assert errs["cells_total"] == pytest.approx(0.0)


def test_hallucination_rate_ignores_spheroid_when_absent():
    # A chip-only plan must not be scored against spheroid fields.
    plan = _verified_plan(shear=1.0)  # flow error present
    plan.spheroid = None
    rate = hallucination_rate(plan)
    assert 0 < rate < 1  # flow fields counted, spheroid fields not in denominator


# --- string-format fairness: a format written as text must be extractable ---
#
# ``spheroid_format`` (always "96-ula"/"hanging-drop") and ``plate_format``
# (when written "96-well") are strings. The numeric finder ignores them, so a
# bare model reporting the canonical form was scored unverifiable — spheroid
# golds got hallucination 1.0 for bare/soft-gate/self-verify unconditionally.
# The string finder recovers them and the calculators normalise to the tables.


def test_find_key_ignores_string_format_but_find_str_key_recovers_it():
    assert _find_key({"spheroid_format": "96-ula"}, "spheroid_format") is None
    assert _find_key({"plate_format": "96-well"}, "plate_format") is None
    assert _find_str_key({"spheroid_format": "96-ula"}, "spheroid_format") == "96-ula"
    assert _find_str_key({"plate_format": "96-well"}, "plate_format") == "96-well"
    # a numeric-looking format still extracts as a string for the calc tables
    assert _find_str_key({"plate_format": "96"}, "plate_format") == "96"
    assert _find_str_key({"spheroid": {"spheroid_format": "hanging drop"}},
                         "spheroid_format") == "hanging drop"


def test_run_bare_llm_extracts_string_format_and_becomes_checkable():
    def chat(prompt: str) -> str:
        return json.dumps({
            "spheroid_format": "96-ula",
            "spheroid_count": 96,
            "cells_per_spheroid": 1000.0,
            "cell_diameter_um": 20.0,
            "expected_diameter_um": 200.0,
            "cells_total": 96000.0,
        })

    out = run_bare_llm(_SPHEROID_GOLD, chat)
    assert out["spheroid_format"] == "96-ula"
    assert out["spheroid_count"] == pytest.approx(96.0)
    assert out["expected_diameter_um"] == pytest.approx(200.0)
    # previously this reported the canonical format yet scored unverifiable;
    # now the spheroid branch of bare_checkable / bare_hallucination fires
    assert bare_checkable(out) is True
    assert bare_hallucination(out) == 0.0


def test_bare_hallucination_culture_with_dashed_plate_format():
    extracted = {
        "plate_format": "96-well", "seeding_density_cells_cm2": 1e4, "wells": 1,
        "seed_per_well": 3200.0, "medium_volume_per_well_ml": 0.17,
    }
    assert bare_checkable(extracted) is True
    assert bare_hallucination(extracted) == 0.0


def test_soft_gate_prompt_for_spheroid_uses_spheroid_check():
    prompt = soft_gate_prompt_for(_SPHEROID_GOLD)
    assert "3D-culture" in prompt
    assert "spheroid_format" in prompt
    assert "medium_volume_per_spheroid_ul" in prompt
    assert "width_um" not in prompt  # no flow keys for a spheroid goal


def test_run_self_verify_culture_stage2_replaces_invented_number():
    stage1 = {
        "plate_format": "96", "seeding_density_cells_cm2": 1e4, "wells": 1,
        "seed_per_well": 9999.0,  # invented
        "medium_volume_per_well_ml": 0.5,
    }
    stage2 = {
        "seed_per_well": 3200.0, "total_seed_count": 3200.0,
        "medium_volume_per_well_ml": 0.17, "total_medium_ml": 0.17,
        "expected_confluence_pct": None,
    }

    def chat(prompt: str) -> str:
        data = stage1 if "Goal:" in prompt else stage2
        return "{" + ",".join(f'"{k}":{json.dumps(v)}' for k, v in data.items()) + "}"

    out = run_self_verify(_CULTURE_GOLD, chat)
    assert out["seed_per_well"] == pytest.approx(3200.0)
    assert out["seed_per_well"] != pytest.approx(9999.0)
    assert out["plate_format"] == "96"  # raw survives the merge


def test_run_self_verify_spheroid_stage2_replaces_invented_number():
    stage1 = {
        "spheroid_format": "96-ula", "spheroid_count": 96.0,
        "cells_per_spheroid": 1000.0, "cell_diameter_um": 20.0,
        "expected_diameter_um": 999.0,  # invented
        "cells_total": 50000.0,
    }
    stage2 = {
        "expected_diameter_um": 200.0, "spheroid_volume_ul": 4.18879e-3,
        "medium_volume_per_spheroid_ul": 100.0, "cells_total": 96000.0,
        "total_medium_ml": 9.6, "expected_cells_after_growth": None,
    }

    def chat(prompt: str) -> str:
        data = stage1 if "Goal:" in prompt else stage2
        return "{" + ",".join(f'"{k}":{json.dumps(v)}' for k, v in data.items()) + "}"

    out = run_self_verify(_SPHEROID_GOLD, chat)
    assert out["expected_diameter_um"] == pytest.approx(200.0)
    assert out["expected_diameter_um"] != pytest.approx(999.0)
    assert out["spheroid_format"] == "96-ula"
    assert bare_hallucination(out) == 0.0
