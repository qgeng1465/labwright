"""Plate-based cell culture calculators — wells, seeding, counting, viability.

Standard tissue-culture arithmetic used to plan plate experiments: well surface
area and working volume from the standard well-plate table, cells-per-well from
a seeding density, hemocytometer counting, trypan-blue viability, confluence
prediction and passage splitting. Pure functions: same inputs, same outputs,
unit-tested against the governing equations.

References
----------
- Counting cells with a hemocytometer — Cold Spring Harb Protoc (2008),
  doi:10.1101/pdb.prot4498. Cells/mL = mean cells per 1 mm^2 square × dilution
  × 10^4 (each corner square is 1 mm × 1 mm × 0.1 mm = 0.1 µL).
- Well-plate growth areas and recommended working volumes — standard
  Corning/Falcon multi-well plate dimensions ("Growth Area and Recommended
  Working Volume", e.g. Corning Life Sciences product documentation).
- Primary human hepatocyte sandwich-culture plating density 1.5e5 cells/cm^2 —
  Bioengineering 2023;10(2):131, doi:10.3390/bioengineering10020131.
- HepG2 growth-phase seeding range 2e4-6.3e4 cells/cm^2 — Sci Rep 2021,
  doi:10.1038/s41598-021-81733-3.
"""

from __future__ import annotations

import math

#: Standard multi-well plate growth area (cm^2 per well) and recommended
#: working volume (mL per well). Values from the standard Corning/Falcon
#: well-plate table; do not change without re-pinning the source.
PLATE_FORMATS: dict[str, dict[str, float]] = {
    "6": {"area_cm2": 9.6, "volume_ml": 2.7},
    "12": {"area_cm2": 3.8, "volume_ml": 1.35},
    "24": {"area_cm2": 1.9, "volume_ml": 0.68},
    "48": {"area_cm2": 0.95, "volume_ml": 0.34},
    "96": {"area_cm2": 0.32, "volume_ml": 0.17},
}

# ---------------------------------------------------------------------------
# Plate geometry
# ---------------------------------------------------------------------------


def _normalize_format(plate_format: str) -> str:
    """Normalise a plate-format string to a bare format key.

    Accepts ``"6"``, ``"6-well"``, ``"6well"``, ``"96 well"``, … Returns the
    canonical key (``"6"`` … ``"96"``) or raises ``ValueError``.

    Parameters
    ----------
    plate_format : str
        Plate format, e.g. ``"96-well"``.

    Returns
    -------
    str
        Canonical format key.

    Raises
    ------
    ValueError
        If the format is not one of the standard plates (6/12/24/48/96).
    """
    digits = "".join(ch for ch in str(plate_format).lower() if ch.isdigit())
    if digits not in PLATE_FORMATS:
        raise ValueError(
            f"plate_format must be one of 6/12/24/48/96-well, got {plate_format!r}"
        )
    return digits


def well_surface_area_cm2(plate_format: str) -> float:
    """Growth area of a single well (cm^2).

    Parameters
    ----------
    plate_format : str
        Plate format (e.g. ``"96-well"``).

    Returns
    -------
    float
        Well growth area in cm^2.
    """
    return PLATE_FORMATS[_normalize_format(plate_format)]["area_cm2"]


def medium_volume_per_well(
    plate_format: str, volume_per_area_ml_cm2: float | None = None
) -> float:
    """Standard working medium volume for one well (mL).

    Parameters
    ----------
    plate_format : str
        Plate format (e.g. ``"96-well"``).
    volume_per_area_ml_cm2 : float, optional
        If given, volume is computed as ``area × volume_per_area`` (e.g. a
        shallow-well low-volume protocol); otherwise the standard recommended
        working volume for the format is used.

    Returns
    -------
    float
        Medium volume in mL.
    """
    pf = _normalize_format(plate_format)
    if volume_per_area_ml_cm2 is not None:
        _validate_positive(volume_per_area_ml_cm2=volume_per_area_ml_cm2)
        return PLATE_FORMATS[pf]["area_cm2"] * volume_per_area_ml_cm2
    return PLATE_FORMATS[pf]["volume_ml"]


def cells_per_well(seed_density_cells_cm2: float, plate_format: str) -> float:
    """Cells to seed per well at a given density.

    .. math:: N = \\rho \\cdot A_\\text{well}

    Parameters
    ----------
    seed_density_cells_cm2 : float
        Seeding density in cells/cm^2 (HepG2 growth-phase 2e4-6.3e4, primary
        hepatocyte sandwich 1.5e5).
    plate_format : str
        Plate format (e.g. ``"96-well"``).

    Returns
    -------
    float
        Cells per well.
    """
    _validate_positive(seed_density_cells_cm2=seed_density_cells_cm2)
    return seed_density_cells_cm2 * well_surface_area_cm2(plate_format)


# ---------------------------------------------------------------------------
# Counting & viability
# ---------------------------------------------------------------------------


def hemocytometer_count(avg_cells_per_square: float, dilution_factor: float) -> float:
    """Cell concentration from a hemocytometer count.

    .. math:: C = \\bar{N}_\\text{sq} \\times D \\times 10^4

    Each 1 mm^2 corner square holds 0.1 µL, so mean cells per square × dilution
    × 10^4 gives cells/mL. Reference: CSH Protocols 2008, doi:10.1101/pdb.prot4498.

    Parameters
    ----------
    avg_cells_per_square : float
        Mean cells per 1 mm^2 corner square.
    dilution_factor : float
        Sample dilution factor (1 = neat).

    Returns
    -------
    float
        Concentration in cells/mL.
    """
    _validate_positive(avg_cells_per_square=avg_cells_per_square)
    if dilution_factor < 1:
        raise ValueError(f"dilution_factor must be >= 1, got {dilution_factor!r}")
    return avg_cells_per_square * dilution_factor * 1e4


def trypan_blue_viability(live_cells: float, dead_cells: float) -> float:
    """Viability percent from a trypan-blue (live/dead) count.

    .. math:: V = \\frac{N_\\text{live}}{N_\\text{live} + N_\\text{dead}} \\times 100

    Parameters
    ----------
    live_cells : float
        Live (unstained) cell count.
    dead_cells : float
        Dead (blue-stained) cell count.

    Returns
    -------
    float
        Viability in percent.
    """
    for name, val in (("live_cells", live_cells), ("dead_cells", dead_cells)):
        if not math.isfinite(float(val)) or float(val) < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {val!r}")
    total = live_cells + dead_cells
    if total <= 0:
        raise ValueError("live_cells + dead_cells must be > 0")
    return live_cells / total * 100.0


def viable_cells_in_suspension(
    total_cells_per_ml: float, volume_ml: float, viability_pct: float
) -> float:
    """Total viable cells in a suspension volume.

    .. math:: N = C \\times V \\times V_\\text{ability}

    Parameters
    ----------
    total_cells_per_ml : float
        Total (live + dead) concentration, cells/mL.
    volume_ml : float
        Suspension volume, mL.
    viability_pct : float
        Viability in percent (0-100).

    Returns
    -------
    float
        Live cells in the volume.
    """
    _validate_positive(total_cells_per_ml=total_cells_per_ml, volume_ml=volume_ml)
    if not 0 <= viability_pct <= 100:
        raise ValueError(f"viability_pct must be in [0, 100], got {viability_pct!r}")
    return total_cells_per_ml * volume_ml * viability_pct / 100.0


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------


def confluence_to_cell_count(
    confluence_pct: float, confluent_density_cells_cm2: float, area_cm2: float
) -> float:
    """Cell count implied by a target confluence.

    .. math:: N = \\frac{pct}{100} \\times \\rho_\\text{confluent} \\times A

    Parameters
    ----------
    confluence_pct : float
        Target confluence in percent.
    confluent_density_cells_cm2 : float
        Cells/cm^2 at 100% confluence for this cell type (an *input* — it is
        cell-type and lab dependent, never assumed by this module).
    area_cm2 : float
        Culture area, cm^2.

    Returns
    -------
    float
        Cell count at the target confluence.
    """
    _validate_positive(
        confluent_density_cells_cm2=confluent_density_cells_cm2, area_cm2=area_cm2
    )
    if not 0 <= confluence_pct <= 100:
        raise ValueError(f"confluence_pct must be in [0, 100], got {confluence_pct!r}")
    return confluence_pct / 100.0 * confluent_density_cells_cm2 * area_cm2


def cell_count_to_confluence(
    cell_count: float, confluent_density_cells_cm2: float, area_cm2: float
) -> float:
    """Confluence percent implied by a cell count.

    .. math:: pct = \\frac{N}{\\rho_\\text{confluent} \\times A} \\times 100

    Parameters
    ----------
    cell_count : float
        Observed/predicted cell count.
    confluent_density_cells_cm2 : float
        Cells/cm^2 at 100% confluence (an input, as above).
    area_cm2 : float
        Culture area, cm^2.

    Returns
    -------
    float
        Confluence in percent (may exceed 100 for over-confluent cultures).
    """
    _validate_positive(
        cell_count=cell_count, confluent_density_cells_cm2=confluent_density_cells_cm2,
        area_cm2=area_cm2,
    )
    return cell_count / (confluent_density_cells_cm2 * area_cm2) * 100.0


def time_to_confluence_pct(
    seed_count: float,
    confluent_density_cells_cm2: float,
    area_cm2: float,
    target_confluence_pct: float,
    doubling_time_h: float,
) -> float:
    """Hours until a seeded culture reaches a target confluence.

    Uses exponential growth (see :func:`labwright.calc.cell.time_to_confluence`)
    against the cell count implied by the target confluence. Returns 0 if the
    culture is already at or above the target.

    Parameters
    ----------
    seed_count : float
        Cells seeded at t=0.
    confluent_density_cells_cm2 : float
        Cells/cm^2 at 100% confluence (an input).
    area_cm2 : float
        Culture area, cm^2.
    target_confluence_pct : float
        Target confluence in percent.
    doubling_time_h : float
        Population doubling time, hours.

    Returns
    -------
    float
        Hours to reach the target confluence (0 if already there).
    """
    from labwright.calc import cell

    target = confluence_to_cell_count(
        target_confluence_pct, confluent_density_cells_cm2, area_cm2
    )
    return cell.time_to_confluence(seed_count, target, doubling_time_h)


# ---------------------------------------------------------------------------
# Passage & other single-step arithmetic
# ---------------------------------------------------------------------------


def passage_split_ratio(
    cells_at_harvest: float, seed_density_cells_cm2: float, plate_format: str
) -> float:
    """Split ratio (fold) at passage for a plate reseed.

    .. math:: f = \\frac{N_\\text{harvest}}{\\rho \\times A_\\text{well}}

    Parameters
    ----------
    cells_at_harvest : float
        Cells harvested (per well) at passage.
    seed_density_cells_cm2 : float
        Reseeding density, cells/cm^2.
    plate_format : str
        Plate format for reseeding.

    Returns
    -------
    float
        Split ratio as ``1:X`` (must be >= 1).

    Raises
    ------
    ValueError
        If the harvest is too small to split at the reseeding density.
    """
    per_well = cells_per_well(seed_density_cells_cm2, plate_format)
    ratio = cells_at_harvest / per_well
    if ratio < 1:
        raise ValueError(
            f"harvest ({cells_at_harvest}) < reseed demand per well ({per_well}); "
            "cannot split"
        )
    return ratio


def moi_virus_volume(moi: float, cell_count: float, titer_pfu_ml: float) -> float:
    """Virus stock volume for a target multiplicity of infection.

    .. math:: V = \\frac{\\text{MOI} \\times N}{\\text{titer}}

    Parameters
    ----------
    moi : float
        Multiplicity of infection, PFU/cell.
    cell_count : float
        Cells at infection.
    titer_pfu_ml : float
        Virus titer, PFU/mL.

    Returns
    -------
    float
        Virus stock volume in mL.
    """
    _validate_positive(moi=moi, cell_count=cell_count, titer_pfu_ml=titer_pfu_ml)
    return moi * cell_count / titer_pfu_ml


def cryo_vial_count(total_cells: float, cells_per_vial: float) -> int:
    """Number of cryo-vials needed to bank a cell total.

    .. math:: n = \\lceil N / N_\\text{vial} \\rceil

    Parameters
    ----------
    total_cells : float
        Total cells to bank.
    cells_per_vial : float
        Cells per vial.

    Returns
    -------
    int
        Number of vials.
    """
    _validate_positive(total_cells=total_cells, cells_per_vial=cells_per_vial)
    return int(math.ceil(total_cells / cells_per_vial))


def _validate_positive(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "PLATE_FORMATS",
    "well_surface_area_cm2",
    "medium_volume_per_well",
    "cells_per_well",
    "hemocytometer_count",
    "trypan_blue_viability",
    "viable_cells_in_suspension",
    "confluence_to_cell_count",
    "cell_count_to_confluence",
    "time_to_confluence_pct",
    "passage_split_ratio",
    "moi_virus_volume",
    "cryo_vial_count",
]
