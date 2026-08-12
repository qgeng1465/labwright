"""Tests for the benchmark's Wilson score intervals (eval/ci.py)."""

from __future__ import annotations

import pytest

from eval.ci import format_ci, wilson_ci


def test_wilson_ci_zero_of_n_has_positive_upper():
    """0 / n never reads as an exact zero — the honest upper bound is finite."""
    lo, hi = wilson_ci(0, 30)
    assert lo == 0.0
    # z^2 / (n + z^2) — closed form at k=0, 95% z ≈ 1.9599…
    assert hi == pytest.approx(3.8416 / (30 + 3.8416), rel=1e-3)
    assert 0.10 < hi < 0.12  # ~0.114


def test_wilson_ci_all_of_n_has_positive_lower():
    """n / n never reads as an exact certain 1.0 either (mirror of k=0)."""
    lo, hi = wilson_ci(24, 24)
    assert hi == 1.0
    # closed form at k=n: 1 / (1 + z^2/n) = n / (n + z^2)
    assert lo == pytest.approx(24 / (24 + 3.8416), rel=1e-3)
    assert 0.86 < lo < 0.87  # ~0.862


def test_wilson_ci_contains_point_estimate():
    for k, n in [(1, 10), (5, 10), (9, 10), (12, 24), (40, 100)]:
        lo, hi = wilson_ci(k, n)
        assert lo <= k / n <= hi


def test_wilson_ci_symmetric_at_half():
    lo, hi = wilson_ci(12, 24)
    assert lo == pytest.approx(1 - hi, abs=1e-9)


def test_wilson_ci_shrinks_with_sample_size():
    """Same proportion, twice the data → strictly narrower interval."""
    lo_small, hi_small = wilson_ci(6, 24)     # 0.25
    lo_large, hi_large = wilson_ci(12, 48)    # 0.25
    assert hi_large - lo_large < hi_small - lo_small


def test_wilson_ci_monotone_in_k():
    """More successes at fixed n ⇒ the interval moves strictly up."""
    prev = wilson_ci(0, 24)
    for k in range(1, 25):
        lo, hi = wilson_ci(k, 24)
        assert lo >= prev[0] and hi >= prev[1]
        prev = (lo, hi)


def test_wilson_ci_degenerate_n_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    assert wilson_ci(5, 0) == (0.0, 0.0)


def test_format_ci_docstring_examples():
    assert format_ci(24, 24) == "1.000 [0.862, 1.000]"
    assert format_ci(0, 24) == "0.000 [0.000, 0.138]"


def test_format_ci_custom_width():
    assert format_ci(12, 24, width=1) == "0.5 [0.3, 0.7]"
