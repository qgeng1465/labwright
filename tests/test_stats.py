"""Unit tests for labwright.calc.stats.

Closed-form reference values are computed from the standard normal quantiles
(z_{0.975}=1.96, z_{0.80}=0.842) so the tests are fully independent of the
implementation.
"""

import pytest

from labwright.calc import stats

Z_975 = 1.959964  # z for alpha=0.05 two-sided
Z_80 = 0.8416212  # z for power=0.80


def test_sample_size_reference_value():
    # n = 2(z_a+z_b)^2 * sd^2/d^2 ; delta=1, sd=1 -> 2*(1.96+0.842)^2 = 15.71 -> 16
    assert stats.sample_size_per_group(1.0, 1.0) == 16


def test_sample_size_scales_with_effect():
    # Larger effect -> fewer replicates
    n_small = stats.sample_size_per_group(0.5, 1.0)
    n_large = stats.sample_size_per_group(2.0, 1.0)
    assert n_large < n_small


def test_sample_size_scales_with_variance():
    n_low = stats.sample_size_per_group(1.0, 0.5)
    n_high = stats.sample_size_per_group(1.0, 2.0)
    assert n_high > n_low


def test_power_at_reference_n():
    # n=16, d=1, sd=1 should recover power ~0.80
    assert stats.power_for_sample_size(16, 1.0, 1.0) == pytest.approx(0.80, abs=0.02)


def test_power_monotonic_in_n():
    assert stats.power_for_sample_size(20, 1.0, 1.0) > stats.power_for_sample_size(8, 1.0, 1.0)


def test_cohens_d():
    assert stats.cohens_d(1.5, 1.0) == pytest.approx(1.5)


def test_min_detectable_effect():
    # delta_min = sd*(z_a+z_b)*sqrt(2/n); n=16, sd=1 -> 1*(2.802)*sqrt(2/16)=0.99
    assert stats.min_detectable_effect(16, 1.0) == pytest.approx(2.801 * (2 / 16) ** 0.5, rel=1e-3)


def test_technical_replicates():
    # CV=10%, precision=5%, 95% -> (1.96*10/5)^2 = 15.4 -> 16
    assert stats.technical_replicates(10, 5) == 16


def test_technical_replicates_at_least_one():
    assert stats.technical_replicates(2, 10) >= 1


def test_invalid_inputs():
    with pytest.raises(ValueError):
        stats.sample_size_per_group(0, 1.0)
    with pytest.raises(ValueError):
        stats.sample_size_per_group(1.0, -1.0)
    with pytest.raises(ValueError):
        stats.power_for_sample_size(1, 1.0, 1.0)
