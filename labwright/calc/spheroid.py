"""Spheroid / 3D culture calculators — geometry, size-from-cells, media volume.

Standard 3D-culture arithmetic used to plan spheroid experiments: spheroid
volume and surface area from a diameter (pure geometry), the cell count behind
a target spheroid size (cells packed as a solid sphere of ``N`` cells, each
with a mean single-cell volume), how many spheroids a counted suspension can
form, and the standard working volume of the common spheroid vessels. Pure
functions: same inputs, same outputs, unit-tested against the governing
equations.

References
----------
- Spheroid volume/surface area from diameter — solid-sphere geometry (no
  source needed; ``V = 4/3*pi*r^3``, ``A = 4*pi*r^2``).
- 96-well round-bottom ULA plates: 100 uL working volume per well, one spheroid
  per well — standard Corning spheroid-microplate protocol (CLS-AN-235) and the
  Sartorius/IncuCyte spheroid-assay protocol; PerkinElmer ATPlite 3D is run
  directly in 100 uL per well.
- 384-well ULA plates: 50 uL working volume per well — InSphero Akura 384
  spheroid microplate specification (total well volume 120 uL).
- Hanging-drop spheroids: recommended drop volume 10-20 uL, with 20-50 uL
  viable — Wanigasekara et al., PLOS ONE 2023, doi:10.1371/journal.pone.0276248.
- Functional hepatocyte/liver spheroids stay below ~1.5-2k cells/spheroid;
  larger spheroids develop necrotic cores — Drug Metab Dispos 2024,
  doi:10.1124/dmd.124.001653. Spheroids under ~200 um in diameter avoid
  necrotic cores (oxygen diffuses roughly 200 um from the surface).
- Cells-per-spheroid from a diameter uses each cell's mean single-cell volume
  (a dense-sphere packing approximation; e.g. a 20 um hepatocyte ≈ 4.2 pL, so
  1000 cells ≈ a 200 um spheroid).
"""

from __future__ import annotations

import math

#: Standard working volume (uL) per spheroid for the common spheroid vessels.
#: Each vessel holds one spheroid (one spheroid per well in 96/384-ULA round
#: or hanging drop), so per-vessel volume = per-spheroid volume. Values from the
#: references above; do not change without re-pinning the source.
SPHEROID_FORMATS: dict[str, dict[str, float]] = {
    "96-ula": {"volume_ul": 100.0},       # Corning / IncuCyte / PerkinElmer
    "384-ula": {"volume_ul": 50.0},       # InSphero Akura 384
    "hanging-drop": {"volume_ul": 20.0},  # Wanigasekara et al. 2023 (10-20 uL)
}

#: 1 um^3 in uL; 1 mm^2 in um^2.
_UM3_PER_UL = 1e9
_UM2_PER_MM2 = 1e6


def _normalize_format(spheroid_format: str) -> str:
    """Normalise a spheroid-vessel string to a canonical format key.

    Accepts ``"96-ula"``, ``"96ula"``, ``"96 well ULA"``, ``"hanging drop"``,
    ``"hanging-drop"``, … Returns the canonical key (``"96-ula"``,
    ``"384-ula"``, ``"hanging-drop"``) or raises ``ValueError``.

    Parameters
    ----------
    spheroid_format : str
        Spheroid vessel/format.

    Returns
    -------
    str
        Canonical format key.

    Raises
    ------
    ValueError
        If the format is not a recognised spheroid vessel.
    """
    s = " ".join(str(spheroid_format).lower().split()).replace(" ", "-")
    s = s.replace("well", "").replace("_", "-")
    candidates = {  # loose aliases -> canonical key
        "96-ula": "96-ula", "96ula": "96-ula", "96-ull": "96-ula",
        "96-ula-round-bottom": "96-ula", "ula-96": "96-ula",
        "384-ula": "384-ula", "384ula": "384-ula", "ula-384": "384-ula",
        "hanging-drop": "hanging-drop", "hanging-drop-plate": "hanging-drop",
        "hanging-drop-dish": "hanging-drop", "perfecta3d": "hanging-drop",
    }
    if s in candidates:
        return candidates[s]
    # tolerate "96ula" spelled with spaces/dashes in odd places
    compact = "".join(ch for ch in str(spheroid_format).lower() if ch.isalnum())
    # drop an embedded "well" so "96-well ULA" / "96 well ULA plate" compact to
    # the same token as "96ula" (the docstring promised these, but the bare
    # compact form kept the word and raised)
    compact_nw = compact.replace("well", "")
    if compact in ("96ula", "96ull", "96ulaplate") or compact_nw in ("96ula", "96ull", "96ulaplate"):
        return "96-ula"
    if compact in ("384ula", "384ulaplate") or compact_nw in ("384ula", "384ulaplate"):
        return "384-ula"
    if compact in ("hangingdrop", "hangingdropplate") or compact_nw in ("hangingdrop", "hangingdropplate"):
        return "hanging-drop"
    raise ValueError(
        f"spheroid_format must be one of 96-ula / 384-ula / hanging-drop, got "
        f"{spheroid_format!r}"
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def spheroid_volume_ul(diameter_um: float) -> float:
    """Volume of a solid sphere from its diameter (uL).

    .. math:: V = \\frac{4}{3}\\pi\\left(\\frac{d}{2}\\right)^3

    Parameters
    ----------
    diameter_um : float
        Spheroid diameter in micrometres.

    Returns
    -------
    float
        Spheroid volume in uL (a 300 um spheroid ≈ 0.014 uL ≈ 14 nL).
    """
    _validate_positive(diameter_um=diameter_um)
    r = diameter_um / 2.0
    return 4.0 / 3.0 * math.pi * r ** 3 / _UM3_PER_UL


def spheroid_diameter_um(volume_ul: float) -> float:
    """Diameter of a solid sphere from its volume (um).

    .. math:: d = \\sqrt[3]{\\frac{6V}{\\pi}}

    Parameters
    ----------
    volume_ul : float
        Spheroid volume in uL.

    Returns
    -------
    float
        Spheroid diameter in micrometres.
    """
    _validate_positive(volume_ul=volume_ul)
    v_um3 = volume_ul * _UM3_PER_UL
    return (6.0 * v_um3 / math.pi) ** (1.0 / 3.0)


def spheroid_surface_area_mm2(diameter_um: float) -> float:
    """Surface area of a solid sphere from its diameter (mm^2).

    .. math:: A = 4\\pi\\left(\\frac{d}{2}\\right)^2

    Parameters
    ----------
    diameter_um : float
        Spheroid diameter in micrometres.

    Returns
    -------
    float
        Spheroid surface area in mm^2 (a 200 um spheroid ≈ 0.126 mm^2).
    """
    _validate_positive(diameter_um=diameter_um)
    r = diameter_um / 2.0
    return 4.0 * math.pi * r ** 2 / _UM2_PER_MM2


# ---------------------------------------------------------------------------
# Cell count <-> size
# ---------------------------------------------------------------------------


def cell_volume_ul(cell_diameter_um: float) -> float:
    """Mean volume of a single cell treated as a sphere (uL).

    .. math:: V_\\text{cell} = \\frac{4}{3}\\pi\\left(\\frac{d_\\text{cell}}{2}\\right)^3

    Parameters
    ----------
    cell_diameter_um : float
        Mean single-cell diameter in micrometres (hepatocytes ≈ 20 um ≈ 4.2 pL).

    Returns
    -------
    float
        Single-cell volume in uL.
    """
    _validate_positive(cell_diameter_um=cell_diameter_um)
    r = cell_diameter_um / 2.0
    return 4.0 / 3.0 * math.pi * r ** 3 / _UM3_PER_UL


def spheroid_volume_from_cells(
    cells_per_spheroid: float, cell_diameter_um: float, packing_fraction: float = 1.0
) -> float:
    """Spheroid volume implied by a cell count (uL).

    Treats the spheroid as ``N`` cells of the given mean diameter packed into a
    sphere. ``packing_fraction`` is the fraction of spheroid volume actually
    occupied by cell bodies (dense-sphere packing): 1.0 = a solid sphere of
    cell material — the *upper bound* on volume; real spheroids pack cells with
    small intercellular space, ~0.65–0.75 (e.g. hepatocyte spheroids, Lee et al.,
    Biofabrication 2013). The default 1.0 keeps the documented "1000 cells of
    20 µm → ≈200 µm spheroid" convention; pass a realistic fraction to model a
    looser aggregate.

    .. math:: V = \\frac{N \\cdot V_\\text{cell}}{f_\\text{pack}}

    Parameters
    ----------
    cells_per_spheroid : float
        Cells seeded per spheroid.
    cell_diameter_um : float
        Mean single-cell diameter in micrometres.
    packing_fraction : float, default 1.0
        Volume fraction of the spheroid occupied by cells (0 < f ≤ 1).

    Returns
    -------
    float
        Spheroid volume in uL.
    """
    _validate_positive(
        cells_per_spheroid=cells_per_spheroid,
        cell_diameter_um=cell_diameter_um,
        packing_fraction=packing_fraction,
    )
    _validate_packing_fraction(packing_fraction)
    return cells_per_spheroid * cell_volume_ul(cell_diameter_um) / packing_fraction


def spheroid_diameter_from_cells(
    cells_per_spheroid: float, cell_diameter_um: float, packing_fraction: float = 1.0
) -> float:
    """Spheroid diameter implied by a cell count (um).

    .. math:: d = \\sqrt[3]{\\frac{6 N V_\\text{cell}}{\\pi f_\\text{pack}}}

    1000 cells of 20 um → ≈ 200 um spheroid at packing 1.0, ≈ 222 um at the
    realistic 0.74; 100 cells → ≈ 93 um / ≈ 103 um.

    Parameters
    ----------
    cells_per_spheroid : float
        Cells seeded per spheroid.
    cell_diameter_um : float
        Mean single-cell diameter in micrometres.
    packing_fraction : float, default 1.0
        Volume fraction of the spheroid occupied by cells (see
        :func:`spheroid_volume_from_cells`).

    Returns
    -------
    float
        Spheroid diameter in micrometres.
    """
    _validate_positive(
        cells_per_spheroid=cells_per_spheroid,
        cell_diameter_um=cell_diameter_um,
        packing_fraction=packing_fraction,
    )
    _validate_packing_fraction(packing_fraction)
    v_um3 = (
        spheroid_volume_from_cells(cells_per_spheroid, cell_diameter_um, packing_fraction)
        * _UM3_PER_UL
    )
    return (6.0 * v_um3 / math.pi) ** (1.0 / 3.0)


def cells_per_spheroid_for_diameter(
    target_diameter_um: float, cell_diameter_um: float, packing_fraction: float = 1.0
) -> float:
    """Cells needed to reach a target spheroid diameter.

    .. math:: N = \\frac{f_\\text{pack}\\, V_\\text{spheroid}}{V_\\text{cell}}

    Parameters
    ----------
    target_diameter_um : float
        Target spheroid diameter in micrometres.
    cell_diameter_um : float
        Mean single-cell diameter in micrometres.
    packing_fraction : float, default 1.0
        Volume fraction of the spheroid occupied by cells (see
        :func:`spheroid_volume_from_cells`).

    Returns
    -------
    float
        Cells per spheroid (≈1000 for a 200 um spheroid of 20 um cells).
    """
    _validate_positive(
        target_diameter_um=target_diameter_um,
        cell_diameter_um=cell_diameter_um,
        packing_fraction=packing_fraction,
    )
    _validate_packing_fraction(packing_fraction)
    v_sph = spheroid_volume_ul(target_diameter_um) * _UM3_PER_UL
    v_cell = cell_volume_ul(cell_diameter_um) * _UM3_PER_UL
    return v_sph * packing_fraction / v_cell


# ---------------------------------------------------------------------------
# Seeding arithmetic & media volume
# ---------------------------------------------------------------------------


def spheroid_count_from_suspension(
    total_cells: float, cells_per_spheroid: float
) -> int:
    """How many spheroids a counted suspension can form.

    .. math:: n = \\lfloor N_\\text{total} / N_\\text{spheroid} \\rfloor

    Parameters
    ----------
    total_cells : float
        Total viable cells in the suspension.
    cells_per_spheroid : float
        Cells per spheroid.

    Returns
    -------
    int
        Number of complete spheroids.
    """
    _validate_positive(total_cells=total_cells, cells_per_spheroid=cells_per_spheroid)
    return int(math.floor(total_cells / cells_per_spheroid))


def cells_needed_for_spheroids(
    spheroid_count: float, cells_per_spheroid: float
) -> float:
    """Total cells needed to form ``spheroid_count`` spheroids.

    .. math:: N = n \\cdot N_\\text{spheroid}

    Parameters
    ----------
    spheroid_count : float
        Number of spheroids to form.
    cells_per_spheroid : float
        Cells seeded per spheroid.

    Returns
    -------
    float
        Total cells required.
    """
    _validate_positive(spheroid_count=spheroid_count, cells_per_spheroid=cells_per_spheroid)
    return spheroid_count * cells_per_spheroid


def medium_volume_per_spheroid(spheroid_format: str) -> float:
    """Standard working medium volume for one spheroid (uL).

    Parameters
    ----------
    spheroid_format : str
        Vessel/format (``"96-ula"``, ``"384-ula"``, ``"hanging-drop"``).

    Returns
    -------
    float
        Working volume in uL.
    """
    return SPHEROID_FORMATS[_normalize_format(spheroid_format)]["volume_ul"]


def total_medium_volume(
    spheroid_count: float, per_spheroid_ul: float, dead_volume_ul: float = 0.0
) -> float:
    """Total medium volume for ``spheroid_count`` spheroids (mL).

    The *working* volume is ``n · V_spheroid``; the volume you actually need to
    prep adds the system's dead volume — the fluid trapped in reservoirs,
    tubing or a pump circuit that never reaches a well. Perfusion and
    semi-static chips routinely lose 1–5 mL to tubing + reservoir, so planning
    on working volume alone leaves a culture short.

    .. math:: V_\\text{total} = \\frac{n \\cdot V_\\text{spheroid} + V_\\text{dead}}{1000}

    Parameters
    ----------
    spheroid_count : float
        Number of spheroids.
    per_spheroid_ul : float
        Medium volume per spheroid in uL.
    dead_volume_ul : float, default 0.0
        Reservoir/tubing dead volume in uL (0 when dispensing from a pipette
        into wells only).

    Returns
    -------
    float
        Total medium volume to prepare, in mL.
    """
    _validate_positive(spheroid_count=spheroid_count, per_spheroid_ul=per_spheroid_ul)
    if not math.isfinite(float(dead_volume_ul)) or float(dead_volume_ul) < 0:
        raise ValueError(
            f"dead_volume_ul must be a finite number >= 0, got {dead_volume_ul!r}"
        )
    return (spheroid_count * per_spheroid_ul + dead_volume_ul) / 1000.0


def _validate_packing_fraction(packing_fraction: float) -> None:
    """A volume fraction is between 0 (exclusive) and 1 (inclusive)."""
    if not math.isfinite(float(packing_fraction)) or not (0.0 < float(packing_fraction) <= 1.0):
        raise ValueError(f"packing_fraction must be in (0, 1], got {packing_fraction!r}")


def _validate_positive(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "SPHEROID_FORMATS",
    "spheroid_volume_ul",
    "spheroid_diameter_um",
    "spheroid_surface_area_mm2",
    "cell_volume_ul",
    "spheroid_volume_from_cells",
    "spheroid_diameter_from_cells",
    "cells_per_spheroid_for_diameter",
    "spheroid_count_from_suspension",
    "cells_needed_for_spheroids",
    "medium_volume_per_spheroid",
    "total_medium_volume",
]
