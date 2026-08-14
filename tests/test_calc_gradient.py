"""Tests for the steady concentration-gradient calculators.

The equations must reproduce the source-sink diffusion-bridge physics: the
linear steady-state profile, its steepness, the relaxation time, Fick flux and
the 10τ stability rule for chemotaxis experiments.
"""

import pytest

from labwright.calc import gradient as g


# -- Steady gradient profile --------------------------------------------------


def test_steepness_known_value():
    # 100 µM source vs buffer across a 1 mm gap → 100 µM/mm.
    assert g.linear_gradient_steepness_um_per_mm(100, 0, 1000) == pytest.approx(100.0, rel=1e-9)


def test_steepness_scales_with_drop():
    assert g.linear_gradient_steepness_um_per_mm(200, 0, 1000) == pytest.approx(200.0, rel=1e-9)
    assert g.linear_gradient_steepness_um_per_mm(100, 50, 1000) == pytest.approx(50.0, rel=1e-9)


def test_steady_state_profile_midpoint():
    # Linear Fick profile: midpoint of 100 µM vs 0 over 1 mm reads 50 µM.
    assert g.steady_state_profile_conc_um(100, 0, 1000, 500) == pytest.approx(50.0, rel=1e-9)
    assert g.steady_state_profile_conc_um(100, 0, 1000, 0) == pytest.approx(100.0, rel=1e-9)
    assert g.steady_state_profile_conc_um(100, 0, 1000, 1000) == pytest.approx(0.0, rel=1e-9)


# -- Dynamics and flux --------------------------------------------------------


def test_relaxation_time_known_value():
    # 1 mm gap at small-molecule D → τ = (1e-3)²/5e-10 = 2000 s ≈ 33 min.
    assert g.diffusive_relaxation_time_s(1000) == pytest.approx(2000.0, rel=1e-9)


def test_relaxation_time_scales_with_square_of_distance():
    assert g.diffusive_relaxation_time_s(2000) == pytest.approx(4 * g.diffusive_relaxation_time_s(1000), rel=1e-9)


def test_diffusive_flux_known_value():
    # J = D·ΔC/L = 5e-10·0.1/0.001 = 5e-8 mol/m²/s.
    assert g.diffusive_flux_mol_m2s(100, 0, 1000) == pytest.approx(5e-8, rel=1e-6)


def test_stability_10tau_rule():
    # 24 h ≫ 10τ (20000 s) → stable; 1 h < 10τ → not.
    assert g.gradient_stability_check(2000, 24)["stable"] is True
    assert g.gradient_stability_check(2000, 1)["stable"] is False


# -- Inverse design -----------------------------------------------------------


def test_spacing_inverse_of_steepness():
    # 100 µM/mm from 100 µM vs 0 → 1 mm spacing.
    spacing = g.source_sink_channel_spacing_um_from_gradient(100, 100, 0)
    assert spacing == pytest.approx(1000.0, rel=1e-9)
    # Round-trip: the spacing reproduces the target steepness.
    assert g.linear_gradient_steepness_um_per_mm(100, 0, spacing) == pytest.approx(100.0, rel=1e-9)


def test_spacing_rejects_no_drop():
    with pytest.raises(ValueError):
        g.source_sink_channel_spacing_um_from_gradient(100, 50, 50)  # source == sink


# -- Validation ---------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        g.linear_gradient_steepness_um_per_mm(100, 0, 0)
    with pytest.raises(ValueError):
        g.diffusive_relaxation_time_s(-100)
    with pytest.raises(ValueError):
        g.steady_state_profile_conc_um(100, 0, 1000, 2000)  # x beyond the gap
    with pytest.raises(ValueError):
        g.gradient_stability_check(2000, 0)
