"""Tests for the breathing-function calculators (ALI + cyclic stretch).

The equations must reproduce the ventilated-lung-on-chip physics: breaths/min
from actuator frequency, the physiological strain window, the diaphragm stroke
a target strain needs, the cycle budget and the ALI film thickness.
"""

import pytest

from labwright.calc import breathing as b


# -- Breathing rate -----------------------------------------------------------


def test_breaths_per_minute():
    assert b.breaths_per_minute(0.2) == pytest.approx(12.0, rel=1e-9)
    assert b.breaths_per_minute(0.25) == pytest.approx(15.0, rel=1e-9)


def test_breaths_per_minute_rejects_nonpositive():
    with pytest.raises(ValueError):
        b.breaths_per_minute(0.0)


# -- Strain physiology --------------------------------------------------------


def test_strain_physiological_window():
    assert b.linear_strain_pct_is_physiological(10)["physiological"] is True
    assert b.linear_strain_pct_is_physiological(10)["pathological"] is False


def test_strain_above_20_is_pathological():
    out = b.linear_strain_pct_is_physiological(25)
    assert out["physiological"] is False
    assert out["pathological"] is True


def test_strain_between_windows_neither_flag():
    # (12, 20] % is neither physiological nor (by this criterion) pathological.
    out = b.linear_strain_pct_is_physiological(15)
    assert out["physiological"] is False
    assert out["pathological"] is False


def test_strain_rejects_negative():
    with pytest.raises(ValueError):
        b.linear_strain_pct_is_physiological(-1)


# -- Stretch kinematics -------------------------------------------------------


def test_cyclic_displacement_known_value():
    # 10% over the 250 µm alveolar-sac default span → 25 µm edge stroke.
    assert b.cyclic_displacement_um(10) == pytest.approx(25.0, rel=1e-9)
    assert b.cyclic_displacement_um(10, 250) == pytest.approx(25.0, rel=1e-9)


def test_displacement_scales_with_span():
    assert b.cyclic_displacement_um(10, 500) == pytest.approx(50.0, rel=1e-9)


def test_strain_rate_known_value():
    # 10% at 0.2 Hz → 0.02 /s (linearised, one full cycle to change length).
    assert b.strain_rate_per_s(10, 0.2) == pytest.approx(0.02, rel=1e-9)


def test_total_cycles_known_value():
    # 24 h at 0.2 Hz → 17 280 cycles.
    assert b.total_cycles(24, 0.2) == pytest.approx(17280.0, rel=1e-9)


def test_stretch_duty_fraction():
    assert b.stretch_duty_fraction(0.3, 1.0) == pytest.approx(0.3, rel=1e-9)
    assert b.stretch_duty_fraction(0.0, 1.0) == pytest.approx(0.0, rel=1e-9)
    assert b.stretch_duty_fraction(1.0, 1.0) == pytest.approx(1.0, rel=1e-9)


def test_duty_fraction_validates_range():
    with pytest.raises(ValueError):
        b.stretch_duty_fraction(1.5, 1.0)  # stretch longer than the cycle


# -- Air-liquid interface -----------------------------------------------------


def test_ali_film_known_value():
    # 20 µL over a 0.33 cm² 24-well Transwell → ≈ 606 µm film.
    assert b.ali_liquid_film_um(20, 0.33) == pytest.approx(606.06, rel=1e-2)


def test_ali_film_zero_volume():
    assert b.ali_liquid_film_um(0, 0.33) == pytest.approx(0.0, rel=1e-9)


def test_ali_film_rejects_bad_area():
    with pytest.raises(ValueError):
        b.ali_liquid_film_um(20, 0.0)
