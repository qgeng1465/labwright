"""Tests for the pumpless (rocking/tilting) gravity-flow calculators.

The equations must reproduce the MIMETAS-style rocker physics: the hydrostatic
head from tilt, the Hagen–Poiseuille flow it drives, the peak wall shear, the
half-cycle displacement, the oscillatory shear index and the cycle count.
"""

import pytest

from labwright.calc import pumpless as p


# -- Driving pressure ---------------------------------------------------------


def test_hydrostatic_head_known_value():
    # ρ=1000, θ=15°, L=20 mm → 1000·9.81·0.02·sin(15°) ≈ 50.78 Pa.
    assert p.hydrostatic_pressure_pa(1000, 15, 20) == pytest.approx(50.78, rel=1e-2)


def test_hydrostatic_head_scales_with_sin_tilt():
    assert p.hydrostatic_pressure_pa(1000, 15, 20) > p.hydrostatic_pressure_pa(1000, 10, 20)


def test_head_rejects_over_max_tilt():
    with pytest.raises(ValueError):
        p.hydrostatic_pressure_pa(1000, 30, 20)  # beyond MIMETAS 25° limit


# -- Flow and shear from head -------------------------------------------------


def test_flow_from_head_known_value():
    # ΔP = 50.78 Pa, w=1000 µm, h=100 µm, L=20 mm, μ=1e-3.
    q = p.flow_rate_from_pressure_head(50.78, 1000, 100, 20, 1e-3)
    # Q = ΔP·w·h³/(12·μ·L) = 50.78·1e-3·1e-12/(12·1e-3·0.02) m³/s
    #   = 50.78·1e-15/2.4e-4 = 2.116e-10 m³/s = 2.116e-1 µL/s ≈ 12.7 µL/min.
    assert q == pytest.approx(12.7, rel=1e-1)


def test_peak_wall_shear_from_head():
    # τ = ΔP·h/(2L) = 50.78·1e-4/(0.04) = 0.127 Pa.
    assert p.peak_wall_shear_from_head(50.78, 1000, 100, 20) == pytest.approx(0.127, rel=1e-2)


# -- Rocking dynamics ---------------------------------------------------------


def test_rocking_volume_per_half_cycle():
    # 12.7 µL/min × 30 s / 60 = 6.35 µL.
    assert p.rocking_volume_per_half_cycle_ul(12.7, 30) == pytest.approx(6.35, rel=1e-2)


def test_osi_symmetric_rocking_is_half():
    # Forward == backward → OSI → 0.5.
    assert p.oscillatory_shear_index(0.1, 0.1) == pytest.approx(0.5, rel=1e-9)


def test_osi_unidirectional_is_zero():
    # Tesla-valve chip: no reverse flow → OSI → 0.
    assert p.oscillatory_shear_index(0.1, 0.0) == pytest.approx(0.0, rel=1e-9)


def test_osi_monotonic_with_reversal():
    assert p.oscillatory_shear_index(0.1, 0.1) > p.oscillatory_shear_index(0.1, 0.01)


def test_cycles_per_hour():
    assert p.cycles_per_hour(30) == pytest.approx(60.0, rel=1e-9)
    assert p.cycles_per_hour(5) == pytest.approx(360.0, rel=1e-9)


# -- Physiology comparison ----------------------------------------------------


def test_shear_ratio_vs_physiological():
    out = p.shear_ratio_vs_physiological(0.03, 0.03)
    assert out["ratio"] == pytest.approx(1.0, rel=1e-4)
    assert out["in_range"] is True
    out2 = p.shear_ratio_vs_physiological(0.1, 0.03)
    assert out2["in_range"] is False  # 3.3× target


# -- Validation ---------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        p.flow_rate_from_pressure_head(0, 1000, 100, 20, 1e-3)
    with pytest.raises(ValueError):
        p.oscillatory_shear_index(0.0, 0.0)  # no flow at all
    with pytest.raises(ValueError):
        p.hydrostatic_pressure_pa(1000, 30, 20)  # over the 25° rocker limit
