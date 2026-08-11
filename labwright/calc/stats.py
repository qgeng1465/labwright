"""Statistical design calculators — sample size, power, replicates.

A leading cause of unreproducible wet-lab results is underpowered design:
too few biological replicates to detect the effect that is actually present.
These functions turn "how many replicates do I need?" into a number, using
the standard closed-form approximations (normal approximation to the
two-sample t-test; coefficient-of-variation precision formula for assays).
"""

from __future__ import annotations

import math

from scipy import stats

# ---------------------------------------------------------------------------
# Two-sample t-test design
# ---------------------------------------------------------------------------


def sample_size_per_group(
    effect_size: float,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Biological replicates per group for a two-sample t-test.

    .. math::
        n = \\frac{2\\,(z_{1-\\alpha/2} + z_{1-\\beta})^2\\,\\sigma^2}{\\delta^2}

    Parameters
    ----------
    effect_size : float
        Expected difference between group means (delta), in measurement units.
    std_dev : float
        Expected pooled standard deviation of the measurement, same units.
    alpha : float, default 0.05
        Type-I error rate.
    power : float, default 0.80
        Target statistical power (1 - type-II error).
    two_sided : bool, default True
        Use a two-sided test (recommended).

    Returns
    -------
    int
        Number of replicates per group (rounded up).
    """
    _validate_prob(effect_size=effect_size, std_dev=std_dev)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power!r}")
    if effect_size <= 0:
        raise ValueError(f"effect_size must be positive, got {effect_size!r}")
    z_alpha = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = 2 * (z_alpha + z_beta) ** 2 * std_dev**2 / effect_size**2
    return max(2, math.ceil(n))


def power_for_sample_size(
    n_per_group: int,
    effect_size: float,
    std_dev: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power achievable with a given per-group replicate count.

    .. math:: z_{1-\\beta} = \\frac{|\\delta|}{\\sigma}\\sqrt{\\frac{n}{2}} - z_{1-\\alpha/2}

    Parameters
    ----------
    n_per_group : int
        Replicates per group.
    effect_size : float
        Expected between-group difference, in measurement units.
    std_dev : float
        Pooled standard deviation, same units.
    alpha : float, default 0.05
        Type-I error rate.
    two_sided : bool, default True
        Two-sided test.

    Returns
    -------
    float
        Statistical power in [0, 1].
    """
    if not isinstance(n_per_group, int) or n_per_group < 2:
        raise ValueError(f"n_per_group must be an integer >= 2, got {n_per_group!r}")
    _validate_prob(effect_size=effect_size, std_dev=std_dev)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    z_alpha = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_beta = abs(effect_size) / std_dev * math.sqrt(n_per_group / 2) - z_alpha
    return float(stats.norm.cdf(z_beta))


def cohens_d(effect_size: float, pooled_std: float) -> float:
    """Standardized effect size.

    .. math:: d = \\frac{\\delta}{\\sigma}

    Conventions: small ~0.2, medium ~0.5, large ~0.8.

    Parameters
    ----------
    effect_size : float
        Difference between group means.
    pooled_std : float
        Pooled standard deviation.

    Returns
    -------
    float
        Cohen's d.
    """
    _validate_prob(effect_size=effect_size, pooled_std=pooled_std)
    return effect_size / pooled_std


def min_detectable_effect(n_per_group: int, std_dev: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest effect a design can detect.

    .. math:: \\delta_{min} = \\sigma (z_{1-\\alpha/2} + z_{1-\\beta}) \\sqrt{2/n}

    Parameters
    ----------
    n_per_group : int
        Replicates per group.
    std_dev : float
        Pooled standard deviation.
    alpha : float, default 0.05
        Type-I error rate.
    power : float, default 0.80
        Target power.

    Returns
    -------
    float
        Minimum detectable effect in measurement units.
    """
    if not isinstance(n_per_group, int) or n_per_group < 2:
        raise ValueError(f"n_per_group must be an integer >= 2, got {n_per_group!r}")
    _validate_prob(std_dev=std_dev)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return std_dev * z * math.sqrt(2 / n_per_group)


# ---------------------------------------------------------------------------
# Technical replicate precision (assays)
# ---------------------------------------------------------------------------


def technical_replicates(cv_pct: float, precision_pct: float, alpha: float = 0.05) -> int:
    """Number of technical replicates to keep the relative error under a bound.

    .. math:: n = \\left(\\frac{z_{1-\\alpha/2}\\, \\text{CV}}{\\epsilon}\\right)^2

    The standard error of a mean measured with coefficient of variation CV
    shrinks as 1/sqrt(n). This answers "I pipette a 96-well plate: how many
    wells per condition?"

    Parameters
    ----------
    cv_pct : float
        Assay coefficient of variation in percent (e.g. 5 = 5 %).
    precision_pct : float
        Desired relative precision (half-width of the 95 % CI) in percent.
    alpha : float, default 0.05
        Confidence level.

    Returns
    -------
    int
        Number of technical replicates (rounded up, >= 1).
    """
    _validate_prob(cv_pct=cv_pct, precision_pct=precision_pct)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    z = stats.norm.ppf(1 - alpha / 2)
    n = (z * cv_pct / precision_pct) ** 2
    return max(1, math.ceil(n))


def _validate_prob(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "sample_size_per_group",
    "power_for_sample_size",
    "cohens_d",
    "min_detectable_effect",
    "technical_replicates",
]
