"""End-to-end integration tests for the seven post-v1 design domains.

Each new block (barrier, oxygen, pumpless, breathing, pulsatile, scaling,
gradient) must flow through the full pipeline exactly like the v1 blocks:
raw inputs in → calculator-derived numbers out, the hard gate rejecting any
derived field smuggled in, and the verifier flagging pathological designs.
"""

import pytest

from labwright.design import submit_design
from labwright.blocks import BLOCKS, ALL_DERIVED_KEYS

# A generic goal/rationale with no numbers, so prose checks stay quiet.
_BASE = {
    "goal": "Organ-chip culture model with physiological recapitulation",
    "rationale": "Physiological target values; cell-type registry references",
    "caveats": [],
}


def _submit(extra: dict) -> dict:
    return submit_design({**_BASE, **extra})


def _errors(result: dict) -> list[str]:
    return [i["message"] for i in result["verification"] if i["level"] == "error"]


def _warnings(result: dict) -> list[str]:
    return [i["message"] for i in result["verification"] if i["level"] == "warning"]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_new_domains_registered_in_blocks():
    for name in ("barrier", "oxygen", "pumpless", "breathing", "pulsatile",
                 "scaling", "gradient"):
        assert name in BLOCKS, f"block {name!r} missing from registry"


def test_new_derived_keys_in_flat_gate_set():
    # The gate set is a flat frozenset of bare derived key names (no block
    # prefix), because _reject_derived_fields walks nested dict keys.
    for key in ("teer_ohm_cm2", "penetration_depth_um",
                "driven_flow_rate_uLmin", "breaths_per_minute",
                "womersley_number", "organ_flow_fraction",
                "steepness_um_per_mm"):
        assert key in ALL_DERIVED_KEYS


# ---------------------------------------------------------------------------
# Pumpless (gravity-flow rocking platform)
# ---------------------------------------------------------------------------


def test_pumpless_design_derives_gravity_flow():
    result = _submit({"pumpless": {
        "cell_type": "hepg2", "tilt_angle_deg": 15, "channel_length_mm": 20,
        "width_um": 1000, "height_um": 100, "rocking_half_period_s": 30,
    }})
    assert result["status"] == "ok", result["verification_summary"]
    d = result["design"]["pumpless"]
    assert d["hydrostatic_head_pa"] == pytest.approx(50.78, rel=1e-2)
    assert d["driven_flow_rate_uLmin"] == pytest.approx(12.7, rel=1e-1)
    assert d["peak_wall_shear_pa"] == pytest.approx(0.127, rel=1e-2)
    assert d["cycles_per_hour"] == pytest.approx(60.0, rel=1e-9)
    # hepg2 target 0.05–0.15 Pa → midpoint 0.10; 0.127/0.10 ≈ 1.27 in-range.
    assert d["shear_ratio_to_target"] == pytest.approx(1.27, rel=1e-2)


def test_pumpless_gate_rejects_derived():
    with pytest.raises(ValueError, match="peak_wall_shear_pa"):
        _submit({"pumpless": {
            "cell_type": "hepg2", "tilt_angle_deg": 15, "channel_length_mm": 20,
            "width_um": 1000, "height_um": 100, "rocking_half_period_s": 30,
            "peak_wall_shear_pa": 0.127,
        }})


# ---------------------------------------------------------------------------
# Breathing (lung ALI + cyclic stretch)
# ---------------------------------------------------------------------------


def test_breathing_design_derives_lung_params():
    result = _submit({"breathing": {
        "cell_type": "alveolar", "frequency_hz": 0.2, "strain_pct": 10,
        "culture_duration_h": 24, "apical_volume_ul": 20,
        "surface_area_cm2": 0.33, "stretch_seconds": 0.3, "cycle_seconds": 1.0,
    }})
    assert result["status"] == "ok", result["verification_summary"]
    d = result["design"]["breathing"]
    assert d["breaths_per_minute"] == pytest.approx(12.0, rel=1e-9)
    assert d["cyclic_displacement_um"] == pytest.approx(25.0, rel=1e-9)
    assert d["total_cycles"] == pytest.approx(17280.0, rel=1e-9)
    assert d["stretch_duty_fraction"] == pytest.approx(0.3, rel=1e-9)
    assert d["ali_liquid_film_um"] == pytest.approx(606.06, rel=1e-2)


def test_breathing_warns_pathological_strain():
    result = _submit({"breathing": {
        "cell_type": "alveolar", "frequency_hz": 0.2, "strain_pct": 25,
    }})
    msgs = _warnings(result)
    assert any("pathological" in m for m in msgs)
    assert "breathing.strain_pct" in {i["field"] for i in result["verification"]}


def test_breathing_warns_outside_window():
    result = _submit({"breathing": {
        "cell_type": "alveolar", "frequency_hz": 0.2, "strain_pct": 15,
    }})
    msgs = _warnings(result)
    assert any("5–12%" in m for m in msgs)


def test_breathing_gate_rejects_derived():
    with pytest.raises(ValueError, match="breaths_per_minute"):
        _submit({"breathing": {
            "cell_type": "alveolar", "frequency_hz": 0.2, "strain_pct": 10,
            "breaths_per_minute": 12,
        }})


# ---------------------------------------------------------------------------
# Pulsatile (cardiac waveform)
# ---------------------------------------------------------------------------


def test_pulsatile_design_derives_waveform():
    result = _submit({"pulsatile": {
        "cell_type": "endothelial", "frequency_hz": 1.2, "channel_height_um": 100,
        "shear_mean_pa": 0.59, "shear_amplitude_pa": 0.3,
        "peak_flow_uLmin": 10, "minimum_flow_uLmin": 2, "mean_flow_uLmin": 6,
    }})
    assert result["status"] == "ok", result["verification_summary"]
    d = result["design"]["pulsatile"]
    assert d["womersley_number"] == pytest.approx(0.137, rel=1e-2)
    assert d["oscillatory_shear_index"] == pytest.approx(0.0, rel=1e-9)  # no reversal
    assert d["peak_shear_pa"] == pytest.approx(0.89, rel=1e-9)
    assert d["pulsatility_index"] == pytest.approx(8 / 6, rel=1e-9)


def test_pulsatile_warns_strongly_reversing():
    # mean 0.1, amp 1.0 → OSI ≈ 0.42 > 0.3 (atheroprone reversal).
    result = _submit({"pulsatile": {
        "cell_type": "endothelial", "frequency_hz": 1.2, "channel_height_um": 100,
        "shear_mean_pa": 0.1, "shear_amplitude_pa": 1.0,
    }})
    assert result["design"]["pulsatile"]["oscillatory_shear_index"] > 0.3
    msgs = _warnings(result)
    assert any("reversing" in m for m in msgs)


def test_pulsatile_gate_rejects_derived():
    with pytest.raises(ValueError, match="womersley_number"):
        _submit({"pulsatile": {
            "cell_type": "endothelial", "frequency_hz": 1.2, "channel_height_um": 100,
            "shear_mean_pa": 0.59, "shear_amplitude_pa": 0.3,
            "womersley_number": 0.14,
        }})


# ---------------------------------------------------------------------------
# Scaling (body-on-chip allometry)
# ---------------------------------------------------------------------------


def test_scaling_design_derives_body_on_chip():
    result = _submit({"scaling": {
        "organ": "liver", "total_cells_chip": 1e6, "chip_volume_ul": 1000,
        "flow_rate_uLmin": 100, "target_transit_s": 600,
    }})
    assert result["status"] == "ok", result["verification_summary"]
    d = result["design"]["scaling"]
    assert d["organ_flow_fraction"] == pytest.approx(0.27, rel=1e-9)
    assert d["organ_flow_rate_mlmin"] == pytest.approx(1350.0, rel=1e-9)
    assert d["cells_in_organ"] == pytest.approx(21428.57, rel=1e-4)
    assert d["allometric_scale"] == pytest.approx(0.0562, rel=1e-2)
    assert d["transit_time_s"] == pytest.approx(600.0, rel=1e-9)
    assert d["residence_time_match_error_s"] == pytest.approx(0.0, rel=1e-9)


def test_scaling_gate_rejects_derived():
    with pytest.raises(ValueError, match="organ_flow_fraction"):
        _submit({"scaling": {
            "organ": "liver", "total_cells_chip": 1e6, "organ_flow_fraction": 0.27,
        }})


# ---------------------------------------------------------------------------
# Gradient (chemotaxis source-sink)
# ---------------------------------------------------------------------------


def test_gradient_design_derives_source_sink():
    result = _submit({"gradient": {
        "chemoattractant": "CXCL12", "source_conc_um": 100, "sink_conc_um": 0,
        "distance_um": 1000, "experiment_hours": 24,
    }})
    assert result["status"] == "ok", result["verification_summary"]
    d = result["design"]["gradient"]
    assert d["steepness_um_per_mm"] == pytest.approx(100.0, rel=1e-9)
    assert d["midpoint_conc_um"] == pytest.approx(50.0, rel=1e-9)
    assert d["relaxation_time_s"] == pytest.approx(2000.0, rel=1e-9)
    assert d["flux_mol_m2s"] == pytest.approx(5e-8, rel=1e-6)


def test_gradient_warns_unstable_short_experiment():
    # 1 h experiment with a 2000 s τ gradient → reads the transient, not steady state.
    result = _submit({"gradient": {
        "chemoattractant": "CXCL12", "source_conc_um": 100, "sink_conc_um": 0,
        "distance_um": 1000, "experiment_hours": 1,
    }})
    msgs = _warnings(result)
    assert any("10τ" in m for m in msgs)
    assert any(i["field"] == "gradient.experiment_hours" for i in result["verification"])


def test_gradient_gate_rejects_derived():
    with pytest.raises(ValueError, match="steepness_um_per_mm"):
        _submit({"gradient": {
            "chemoattractant": "CXCL12", "source_conc_um": 100, "sink_conc_um": 0,
            "distance_um": 1000, "experiment_hours": 24, "steepness_um_per_mm": 100,
        }})
