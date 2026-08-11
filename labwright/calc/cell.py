"""Cell culture calculators — seeding, expansion, confluence.

Simple, standard growth models used for culture planning in organ-on-chip
experiments. Assumes log-phase exponential growth, which is accurate enough
for planning seeding densities and harvest timelines.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seeding_cell_count(density_cells_cm2: float, area_cm2: float) -> float:
    """Number of cells to seed on a given culture area.

    .. math:: N = \\rho \\cdot A

    Parameters
    ----------
    density_cells_cm2 : float
        Seeding density in cells/cm^2 (e.g. HepG2 ~5e4-1e5, primary
        hepatocytes ~1e5 cells/cm^2).
    area_cm2 : float
        Culture area in cm^2 (a 400 µm x 4 mm x 100 µm channel ≈ 0.016 cm^2).

    Returns
    -------
    float
        Total cell count to seed.
    """
    _validate_positive(density_cells_cm2=density_cells_cm2, area_cm2=area_cm2)
    return density_cells_cm2 * area_cm2


def culture_area(width_um: float, length_mm: float) -> float:
    """Plan-view culture area of a channel.

    Parameters
    ----------
    width_um : float
        Channel width in micrometres.
    length_mm : float
        Channel length in millimetres.

    Returns
    -------
    float
        Area in cm^2.
    """
    _validate_positive(width_um=width_um, length_mm=length_mm)
    return (width_um * 1e-6) * (length_mm * 1e-3) * 1e4  # m^2 -> cm^2


# ---------------------------------------------------------------------------
# Growth / expansion
# ---------------------------------------------------------------------------


def cell_count_after_time(seed_count: float, doubling_time_h: float, elapsed_h: float) -> float:
    """Exponential growth prediction.

    .. math:: N(t) = N_0 \\, 2^{t/t_d}

    Parameters
    ----------
    seed_count : float
        Number of cells seeded at t=0.
    doubling_time_h : float
        Population doubling time in hours (HepG2 ~30-40 h; HepaRG ~50 h;
        primary hepatocytes do not divide).
    elapsed_h : float
        Elapsed time in hours.

    Returns
    -------
    float
        Predicted cell count.
    """
    _validate_positive(seed_count=seed_count, doubling_time_h=doubling_time_h)
    if elapsed_h < 0:
        raise ValueError(f"elapsed_h must be >= 0, got {elapsed_h!r}")
    return seed_count * 2 ** (elapsed_h / doubling_time_h)


def time_to_confluence(seed_count: float, confluence_count: float, doubling_time_h: float) -> float:
    """Hours until the culture reaches a target (confluent) cell number.

    .. math:: t = t_d \\log_2(N_c / N_0)

    Parameters
    ----------
    seed_count : float
        Seeded cell count.
    confluence_count : float
        Cell count at the desired confluence.
    doubling_time_h : float
        Doubling time in hours.

    Returns
    -------
    float
        Hours to reach ``confluence_count`` (0 if already reached).
    """
    _validate_positive(seed_count=seed_count, confluence_count=confluence_count, doubling_time_h=doubling_time_h)
    if confluence_count <= seed_count:
        return 0.0
    return doubling_time_h * math.log2(confluence_count / seed_count)


def required_expansion_factor(target_count: float, seed_count: float) -> float:
    """Fold-expansion needed to go from seed to target count.

    .. math:: f = N_t / N_0

    Parameters
    ----------
    target_count : float
        Desired final cell count.
    seed_count : float
        Seeded cell count.

    Returns
    -------
    float
        Expansion factor (>= 1).
    """
    _validate_positive(target_count=target_count, seed_count=seed_count)
    if target_count < seed_count:
        raise ValueError("target_count must be >= seed_count for an expansion")
    return target_count / seed_count


def viable_cells(total_cells: float, viability_pct: float) -> float:
    """Live cells given a viability percentage.

    Parameters
    ----------
    total_cells : float
        Total (live + dead) cell count.
    viability_pct : float
        Viability in percent (0-100).

    Returns
    -------
    float
        Live cell count.
    """
    _validate_positive(total_cells=total_cells)
    if not 0 <= viability_pct <= 100:
        raise ValueError(f"viability_pct must be in [0, 100], got {viability_pct!r}")
    return total_cells * viability_pct / 100.0


def _validate_positive(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "seeding_cell_count",
    "culture_area",
    "cell_count_after_time",
    "time_to_confluence",
    "required_expansion_factor",
    "viable_cells",
]
