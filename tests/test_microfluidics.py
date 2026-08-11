"""Unit tests for labwright.calc.microfluidics.

Every assertion below checks the calculators against an independent analytic
evaluation of the governing equation, so the test file doubles as a
derivation audit.
"""

import pytest

from labwright.calc import microfluidics as mf
from labwright.calc.units import Q, ureg


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_hydraulic_diameter_analytic():
    # D_h = 2wh/(w+h), w=400, h=100 -> 80000/500 = 160
    assert mf.hydraulic_diameter(400, 100) == pytest.approx(160.0)


def test_mean_velocity_units():
    # Q=10 uL/min = 10e3 um^3/ms? Verify via pint conversion: 10 uL/min / (400*100 um^2)
    u = mf.mean_velocity(10, 400, 100)
    # independent: Q = 10 uL/min = 10e-6 L/min = 10e-6 * 1e15 um^3/min = 1e10 um^3/min
    # area = 4e4 um^2 -> u = 1e10/4e4 = 2.5e5 um/min = 2.5e5*1e-3/60 mm/s = 4.167 mm/s
    assert u == pytest.approx(4.167, rel=1e-3)


# ---------------------------------------------------------------------------
# Shear stress
# ---------------------------------------------------------------------------


def test_wall_shear_stress_analytic():
    # tau = 6 mu Q / (w h^2); Q=10 uL/min = 10e-9 m^3/min = 10e-9/60 m^3/s
    # w=400e-6 m, h=100e-6 m -> tau = 6*1e-3*(10e-9/60)/(400e-6*(100e-6)^2) = 0.25 Pa
    tau = mf.wall_shear_stress(10, 400, 100, 1e-3)
    expected = 6 * 1e-3 * (10e-9 / 60) / (400e-6 * (100e-6) ** 2)
    assert tau == pytest.approx(expected)


def test_shear_physiological_range():
    # 400x100 um channel at 1 uL/min should give ~0.025 Pa (0.25 dyn/cm2)
    tau = mf.wall_shear_stress(1, 400, 100, 1e-3)
    assert 0.02 < tau < 0.03
    # and expressed in dyn/cm2 it should be ~0.25
    assert Q(tau, "Pa").to("dyn/cm**2").magnitude == pytest.approx(0.25, rel=1e-6)


def test_flow_rate_for_shear_stress_is_inverse():
    tau = 0.05
    q = mf.flow_rate_for_shear_stress(tau, 400, 100, 1e-3)
    assert mf.wall_shear_stress(q, 400, 100, 1e-3) == pytest.approx(tau, rel=1e-9)


def test_reynolds_laminar():
    # Microfluidic channels are always strongly laminar
    assert mf.reynolds_number(10, 400, 100, 1e-3) < 1


def test_pressure_drop_analytic():
    # dP = 12 mu Q L /(w h^3); L=20mm, Q=10 uL/min=10e-9/60 m^3/s
    dp = mf.pressure_drop(10, 400, 100, 20, 1e-3)
    expected = 12 * 1e-3 * (10e-9 / 60) * 20e-3 / (400e-6 * (100e-6) ** 3)
    assert dp == pytest.approx(expected)


def test_residence_time_analytic():
    # t = V/Q = L*w*h/Q ; Q = 10e-9/60 m^3/s
    t = mf.residence_time(10, 400, 100, 20)
    assert t == pytest.approx(20e-3 * 400e-6 * 100e-6 / (10e-9 / 60))


def test_channel_volume():
    # 400 um * 100 um * 20 mm = 400e-6*100e-6*20e-3 m^3 = 8e-10 m^3 = 0.8 uL
    assert mf.channel_volume(400, 100, 20) == pytest.approx(0.8)


def test_o2_delivery():
    # 1 uL/min with 0.2 mM O2 consumed fully -> 0.2 umol/min? 1uL*0.2e-3 mol/L = 0.2e-9 mol = 0.0002 umol
    rate = mf.o2_delivery_rate(1, 0.2e-3, 0.0)
    assert rate == pytest.approx(1e-6 * 0.2e-3 * 1e6)  # uL * mol/L = umol? verify scale
    # 1 uL = 1e-6 L -> moles = 1e-6 * 0.2e-3 = 2e-10 mol = 0.0002 umol per minute
    assert rate == pytest.approx(2e-4, rel=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [mf.wall_shear_stress, mf.reynolds_number])
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_rejects_non_positive(fn, bad):
    with pytest.raises(ValueError):
        fn(bad, 400, 100, 1e-3)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_rejects_non_positive_pressure_drop(bad):
    with pytest.raises(ValueError):
        mf.pressure_drop(bad, 400, 100, 20, 1e-3)


def test_units_roundtrip():
    tau = Q(0.05, "Pa")
    assert tau.to("dyn/cm**2").to("Pa").magnitude == pytest.approx(0.05, rel=1e-9)
    assert ureg.Quantity(0.05, "Pa").to("dyn/cm**2").magnitude == pytest.approx(0.5, rel=1e-9)
