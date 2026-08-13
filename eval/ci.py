"""Proportion confidence intervals for the benchmark (Wilson score interval).

The benchmark's headline rates — "usable designs", "hallucination rate" — are
binomial proportions over a finite gold set. The Wilson score interval is the
honest interval for a small-n proportion: unlike a naive normal approximation
it never collapses to zero width at 0 % or 100 %, so "0.000 hallucination over
24 goals" is reported as ``0.000 [0.000, 0.138]`` (the 95 % upper bound), not a
false exact zero. It is also the interval a reviewer can re-derive from
``k`` and ``n`` alone — no sampler, no prior.

Reference: E. B. Wilson, "Probable inference, the law of succession, and
statistical inference", JASA 22 (1927) 209–212.  DOI 10.1080/01621459.1927.10502953
"""

from __future__ import annotations

import math

#: Normal quantile for a two-sided 95 % interval (1.95996398454…).
Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` trials.

    Returns ``(lower, upper)`` as proportions in [0, 1].  The degenerate
    ``n == 0`` case returns ``(0.0, 0.0)`` (no evidence, no interval).
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def format_ci(k: int, n: int, z: float = Z_95, width: int = 3) -> str:
    """Format ``k/n`` as a proportion with its 95 % interval.

    >>> format_ci(24, 24)
    '1.000 [0.862, 1.000]'
    >>> format_ci(0, 24)
    '0.000 [0.000, 0.138]'
    """
    p = k / n if n else 0.0
    lo, hi = wilson_ci(k, n, z)
    return f"{p:.{width}f} [{lo:.{width}f}, {hi:.{width}f}]"


__all__ = ["Z_95", "wilson_ci", "format_ci"]
