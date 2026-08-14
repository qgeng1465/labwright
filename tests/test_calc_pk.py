"""Unit tests for the perfused-system PK calculator (:mod:`labwright.calc.pk`).

Each function is tested against its governing equation plus the unit traps that
matter in practice (mM vs uM, min vs h, the 6e-5 mass factor).
"""

import math

import pytest

from labwright.calc import pk
from labwright.design import derive_pk


def test_extraction_ratio():
    assert pk.extraction_ratio(10, 7) == pytest.approx(0.3)
    assert pk.extraction_ratio(10, 10) == pytest.approx(0.0)  # no clearance
    assert pk.extraction_ratio(10, 0) == pytest.approx(1.0)  # complete clearance


def test_extraction_ratio_negative_is_secretion():
    # Outlet exceeding inlet is returned as-is (net secretion), not clamped.
    assert pk.extraction_ratio(10, 12) == pytest.approx(-0.2)


def test_clearance_is_E_times_Q():
    e = pk.extraction_ratio(10, 7)
    assert pk.clearance_uLmin(10, 7, 2) == pytest.approx(e * 2)
    assert pk.clearance_uLmin(10, 7, 2) == pytest.approx(0.6)


def test_half_life_from_volume_and_clearance():
    # ln2·V/Cl is in minutes; the calculator converts to hours.
    t = pk.half_life_h(200, 0.6)
    assert t == pytest.approx(math.log(2) * 200 / 0.6 / 60)
    assert t == pytest.approx(3.8508176697774745)


def test_accumulation_ratio_matches_gibaldi():
    # R = 1/(1 − e^(−k·τ)), k = ln2/t½.
    t_half = pk.half_life_h(200, 0.6)
    r = pk.accumulation_ratio(t_half, 24)
    assert r == pytest.approx(1 / (1 - math.exp(-math.log(2) / t_half * 24)))
    assert r > 1.0
    # Interval ≫ half-life → R ≈ 1 (no accumulation).
    assert pk.accumulation_ratio(2, 200) == pytest.approx(1.0, abs=1e-3)
    # Interval = 2×half-life → R = 4/3.
    assert pk.accumulation_ratio(12, 24) == pytest.approx(4 / 3)


def test_mass_cleared_uses_6e5_factor():
    # Cl·C_in·MW·6e-5: 0.6 uL/min × 10 uM × 464 g/mol → 0.167 ug/h.
    assert pk.mass_cleared_ug_h(0.6, 10, 464) == pytest.approx(0.16704)
    # Unit consistency: a mM inlet (1000× larger concentration) is 1000× mass.
    assert pk.mass_cleared_ug_h(0.6, 10e3, 464) == pytest.approx(167.04)


def test_check_compound_mw_gate():
    # The gate hole: "warfarin, MW 464" must fail loudly, not pass silently.
    with pytest.raises(ValueError, match="inconsistent"):
        pk.check_compound_mw("warfarin", 464.0)
    with pytest.raises(ValueError, match="inconsistent"):
        pk.check_compound_mw("DICLOFENAC", 500.0)
    # Matches (case-insensitive, innocuous rounding) pass.
    pk.check_compound_mw("Warfarin", 308.3)
    pk.check_compound_mw("warfarin", 308.0)  # 0.1% off — within the 1% band
    # Unknown compound (the user's own drug) is never checked.
    pk.check_compound_mw("my-novel-drug", 464.0)
    with pytest.raises(ValueError, match="not finite"):
        pk.check_compound_mw("warfarin", float("nan"))


def test_derive_pk_rejects_fabricated_molecular_weight():
    # The gate hole end-to-end: a design that names warfarin but reports a
    # made-up MW must fail derive_pk, which is what the benchmark counts as an
    # unverifiable answer rather than letting it pass as a usable design.
    with pytest.raises(ValueError, match="inconsistent"):
        derive_pk(
            {
                "compound": "warfarin",
                "molecular_weight_g_mol": 464.0,
                "inlet_concentration_uM": 10.0,
                "outlet_concentration_uM": 7.0,
                "flow_rate_uLmin": 2.0,
            }
        )
    # A consistent pair still derives every number normally.
    out = derive_pk(
        {
            "compound": "warfarin",
            "molecular_weight_g_mol": 308.3,
            "inlet_concentration_uM": 10.0,
            "outlet_concentration_uM": 7.0,
            "flow_rate_uLmin": 2.0,
        }
    )
    assert out["extraction_ratio"] == pytest.approx(0.3)
    assert out["mass_cleared_ug_h"] is not None


def test_all_functions_reject_non_finite():
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            pk.extraction_ratio(bad, 7)
        with pytest.raises(ValueError):
            pk.clearance_uLmin(10, 7, bad)


def test_clearance_rejects_negative_flow():
    with pytest.raises(ValueError):
        pk.clearance_uLmin(10, 7, -2)


def test_extraction_ratio_rejects_zero_inlet():
    # E = 1 − C_out/C_in is undefined at a zero inlet (0/0, or a blow-up for
    # outlet > 0). This is what the benchmark harness relies on to score a
    # nonsense "inlet 0 uM" answer as unverifiable instead of crashing.
    for outlet in (0.0, 3.0):
        with pytest.raises(ValueError):
            pk.extraction_ratio(0.0, outlet)
    with pytest.raises(ValueError):
        pk.clearance_uLmin(0.0, 0.0, 2.0)


def test_half_life_and_accumulation_reject_non_positive():
    with pytest.raises(ValueError):
        pk.half_life_h(0, 0.6)
    with pytest.raises(ValueError):
        pk.accumulation_ratio(0, 24)
    with pytest.raises(ValueError):
        pk.mass_cleared_ug_h(0.6, 10, 0)
