"""Tests for the pulsatile / cardiac-waveform calculators.

The equations must reproduce the heart-on-chip waveform physics: the Womersley
unsteadiness number, the reversal metric (OSI) of a sinusoidal shear waveform,
the Gosling pulsatility index and the Pa↔dyn/cm² shear conversions.
"""

import pytest

from labwright.calc import pulsatile as p


# -- Womersley number ---------------------------------------------------------


def test_womersley_known_value():
    # 100 µm channel, 1.2 Hz, water-like medium → α ≈ 0.14 (half-height).
    a = p.womersley_number(1.2, 100, 1e-3, 1000.0)
    assert a == pytest.approx(0.137, rel=1e-2)


def test_womersley_scales_with_half_height():
    assert p.womersley_number(1.2, 200, 1e-3, 1000.0) == pytest.approx(
        p.womersley_number(1.2, 100, 1e-3, 1000.0) * 2.0, rel=1e-9
    )


def test_womersley_low_frequency_is_quasi_steady():
    assert p.womersley_number(0.01, 100, 1e-3, 1000.0) < 0.02


def test_womersley_rejects_invalid():
    with pytest.raises(ValueError):
        p.womersley_number(0, 100, 1e-3, 1000.0)
    with pytest.raises(ValueError):
        p.womersley_number(1.2, -5, 1e-3, 1000.0)


# -- Oscillatory shear index --------------------------------------------------


def test_osi_steady_is_zero():
    assert p.oscillatory_shear_index_from_sinusoid(0.59, 0.0) == pytest.approx(0.0, rel=1e-9)


def test_osi_never_reversing_is_zero():
    # amp < mean → the waveform stays positive → no reversal → OSI 0.
    assert p.oscillatory_shear_index_from_sinusoid(0.59, 0.3) == pytest.approx(0.0, rel=1e-9)


def test_osi_touching_zero_is_zero():
    # mean == amp → the waveform just touches zero each cycle, never reverses.
    assert p.oscillatory_shear_index_from_sinusoid(0.59, 0.59) == pytest.approx(0.0, rel=1e-9)


def test_osi_purely_oscillatory_is_half():
    # mean = 0 → purely reversing waveform → OSI 0.5.
    assert p.oscillatory_shear_index_from_sinusoid(0.0, 0.5) == pytest.approx(0.5, rel=1e-9)


def test_osi_monotonic_with_reversal_depth():
    assert p.oscillatory_shear_index_from_sinusoid(0.59, 0.60) > p.oscillatory_shear_index_from_sinusoid(0.59, 0.59)
    assert p.oscillatory_shear_index_from_sinusoid(0.59, 0.8) > p.oscillatory_shear_index_from_sinusoid(0.59, 0.6)


def test_osi_bounded_in_unit_interval():
    for amp in (0.6, 1.0, 2.0, 5.0):
        osi = p.oscillatory_shear_index_from_sinusoid(0.59, amp)
        assert 0.0 <= osi <= 0.5


def test_osi_rejects_invalid():
    with pytest.raises(ValueError):
        p.oscillatory_shear_index_from_sinusoid(-0.1, 0.5)


# -- Pulsatility index --------------------------------------------------------


def test_pulsatility_index_known_value():
    # peak 10, min 2, mean 6 → (10−2)/6 = 1.333.
    assert p.pulsatility_index(10, 2, 6) == pytest.approx(8 / 6, rel=1e-9)


def test_pulsatility_index_steady_flow():
    # peak == min == mean → PI 1.
    assert p.pulsatility_index(5, 5, 5) == pytest.approx(0.0, rel=1e-9)


def test_pulsatility_index_validates():
    with pytest.raises(ValueError):
        p.pulsatility_index(2, 5, 6)  # peak < min
    with pytest.raises(ValueError):
        p.pulsatility_index(10, 2, 0)  # mean must be > 0


# -- Shear helpers ------------------------------------------------------------


def test_peak_shear_of_sinusoid():
    assert p.peak_shear_of_sinusoid(0.59, 0.59) == pytest.approx(1.18, rel=1e-9)


def test_dyn_per_cm2_conversion():
    # The published demo aortic inflow profile: 0.59 Pa = 5.9 dyn/cm².
    assert p.shear_dyn_per_cm2_from_pa(0.59) == pytest.approx(5.9, rel=1e-9)
    assert p.shear_dyn_per_cm2_from_pa(0.0) == pytest.approx(0.0, rel=1e-9)
