"""Tests for verifying the internal consistency of published protocols."""

import json
import subprocess
import sys

import pytest

from labwright.published import verify_published_protocol


def _self_consistent_claims():
    # 400 µm × 100 µm × 20 mm channel at 2 µL/min, water-like medium.
    chip = {"width_um": 400, "height_um": 100, "length_mm": 20}
    flow = {"flow_rate_uLmin": 2, "viscosity_pas": 0.001}
    computed = {
        "shear_pa": 0.05,          # 6·1e-3·(2e-6/60)/(400e-6·(100e-6)^2) = 0.05
        "channel_volume_ul": 0.8,  # 400·100·20000 µm^3 = 0.8 µL
        "residence_time_s": 24.0,  # 0.8 µL / (2 µL/min) = 24 s
        "mean_velocity_mms": 0.833,
    }
    return chip, flow, computed


def test_consistent_claims_pass():
    chip, flow, claimed = _self_consistent_claims()
    result = verify_published_protocol(chip=chip, flow=flow, claimed=claimed, reference="10.1000/self-consistent")
    assert result["status"] == "ok"
    assert result["n_discrepancies"] == 0
    verdicts = {c["field"]: c["verdict"] for c in result["checks"]}
    assert verdicts["shear_pa"] == "consistent"
    assert verdicts["channel_volume_ul"] == "consistent"


def test_discrepancy_is_flagged():
    chip, flow, _ = _self_consistent_claims()
    # Claim 10× the real shear — e.g. a paper that states shear from a wrong formula.
    claimed = {"shear_pa": 0.5}
    result = verify_published_protocol(chip=chip, flow=flow, claimed=claimed, reference="10.1000/suspicious")
    assert result["status"] == "review_required"
    assert result["n_discrepancies"] == 1
    shear = next(c for c in result["checks"] if c["field"] == "shear_pa")
    assert shear["verdict"] == "discrepancy"
    assert shear["relative_error"] == pytest.approx(9.0, abs=0.01)


def test_reference_is_required():
    result = verify_published_protocol(chip={}, flow={}, claimed={}, reference="")
    assert result["status"] == "validation_error"


def test_unclaimed_fields_marked_not_claimed():
    chip, flow, _ = _self_consistent_claims()
    result = verify_published_protocol(chip=chip, flow=flow, claimed={}, reference="10.1000/x")
    assert all(c["verdict"] == "not_claimed" for c in result["checks"])


def test_invalid_geometry_returns_error():
    result = verify_published_protocol(
        chip={"width_um": -1, "height_um": 100, "length_mm": 20},
        flow={"flow_rate_uLmin": 2},
        claimed={},
        reference="10.1000/x",
    )
    assert result["status"] == "validation_error"


def test_cli_verify_protocol_demo(tmp_path):
    """End-to-end: the CLI catches the same inconsistency a reader would."""
    payload = {
        "chip": {"width_um": 400, "height_um": 100, "length_mm": 20},
        "flow": {"flow_rate_uLmin": 2, "viscosity_pas": 0.001},
        "claimed": {"shear_pa": 0.5},  # wrong
        "reference": "10.1000/cli-demo",
    }
    f = tmp_path / "protocol.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "labwright.cli", "verify-protocol", str(f)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0
    assert "shear_pa" in out.stdout
    assert "discrepancy" in out.stdout
    assert "do not follow from the reported inputs" in out.stdout
