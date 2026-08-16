"""Bioprinting / micro-extrusion calculators — G-code travel to deposited volume.

Extrusion-based bioprinters lay down filaments of cell-laden bioink through a
nozzle; a G-code program moves the head and the extruder is turned on for a
path segment. The deposited volume is the cross-sectional area of the orifice
times the travel distance along the segment (a cylinder of ink), and the
dwell/print time is the travel distance over the programmed feed rate.

The reviewer-facing quantities this module produces:

1. **Micro-extrusion volume per G-code move** — for a given nozzle diameter and
   travel distance (from a coordinate offset), the ink volume laid down. This is
   the "3号低温喷头 vs 5号光固化喷头" class of question: the same G-code path
   deposits a different volume through a different nozzle.
2. **Print time** for a path at a feed rate, and the deposition *rate* (volume
   per minute) a design actually achieves.
3. **Filament mass** from volume and ink density (e.g. a gel or cryoprotectant
   slurry), and the number of lines needed to cover a footprint at a chosen line
   pitch.

The math is textbook geometry/kinematics (cylinder volume, distance = speed ×
time); the only "data" are the nozzle diameters, which are equipment
conventions (labelled as such, not literature values). Always state the assumed
feed rate and nozzle when quoting a micro-extrusion volume — the same path gives
a different volume through a different nozzle, which is exactly the trap these
calculators prevent.

References
----------
- V = A·L (cylinder): standard filament-extrusion geometry used across the
  bioprinting literature and slicers (e.g. pressure-extrusion volume = orifice
  area × path length).
- Nozzle diameters are equipment-spec conventions (manufacturer catalogues),
  not physiological claims.
"""

from __future__ import annotations

import math

#: Canonical multi-nozzle table for the "低温 / 光固化 / 标准" classes a
#: biofabrication rig commonly carries. Diameters are equipment conventions
#: (typical commercial glass/steel dispensing nozzles), labelled as such — not
#: literature measurements. ``kind`` describes the dispensing class.
NOZZLE_TABLE: dict[str, dict[str, float | str]] = {
    "nozzle_1": {"kind": "cryo", "diameter_um": 250.0},
    "nozzle_2": {"kind": "cryo", "diameter_um": 410.0},
    "nozzle_3": {"kind": "cryo", "diameter_um": 500.0},
    "nozzle_4": {"kind": "photocuring", "diameter_um": 200.0},
    "nozzle_5": {"kind": "photocuring", "diameter_um": 300.0},
    "nozzle_6": {"kind": "standard", "diameter_um": 150.0},
}

_NOZZLE_ALIASES: dict[str, str] = {
    "1": "nozzle_1", "nozzle1": "nozzle_1", "低温1": "nozzle_1", "cryo1": "nozzle_1",
    "2": "nozzle_2", "nozzle2": "nozzle_2", "低温2": "nozzle_2", "cryo2": "nozzle_2",
    "3": "nozzle_3", "nozzle3": "nozzle_3", "低温3": "nozzle_3", "cryo3": "nozzle_3",
    "4": "nozzle_4", "nozzle4": "nozzle_4", "光固化4": "nozzle_4", "uv4": "nozzle_4",
    "5": "nozzle_5", "nozzle5": "nozzle_5", "光固化5": "nozzle_5", "uv5": "nozzle_5",
    "6": "nozzle_6", "nozzle6": "nozzle_6", "standard6": "nozzle_6",
}


def nozzle_diameter_um(nozzle_id: str) -> float:
    """Diameter (µm) of a registered nozzle, resolved by id/number/kind alias.

    ``nozzle_id`` may be a table key (``nozzle_3``), a plain number
    (``"3"``), or a class label (``"cryo3"``). Raises ``ValueError`` for an
    unknown nozzle so a mistyped id surfaces instead of silently using one.
    """
    q = str(nozzle_id).strip().lower()
    if q in NOZZLE_TABLE:
        return float(NOZZLE_TABLE[q]["diameter_um"])  # type: ignore[arg-type]
    if q in _NOZZLE_ALIASES:
        return float(NOZZLE_TABLE[_NOZZLE_ALIASES[q]]["diameter_um"])  # type: ignore[arg-type]
    raise ValueError(
        f"unknown nozzle {nozzle_id!r}; known: {', '.join(sorted(NOZZLE_TABLE))} "
        "(or a number/class alias like '3' / 'cryo3')"
    )


def nozzle_kind(nozzle_id: str) -> str:
    """Dispensing class of a nozzle: ``cryo`` / ``photocuring`` / ``standard``."""
    q = str(nozzle_id).strip().lower()
    key = q if q in NOZZLE_TABLE else _NOZZLE_ALIASES.get(q)
    if key is None:
        raise ValueError(f"unknown nozzle {nozzle_id!r}")
    return str(NOZZLE_TABLE[key]["kind"])


def extrusion_volume_nl(travel_distance_um: float, nozzle_diameter_um: float) -> float:
    """Ink volume extruded over a straight G-code path segment, nL.

    .. math:: V = \\pi\\, (d/2)^2\\, L

    with nozzle diameter and travel in µm, then µm³ → nL (1 nL = 10⁶ µm³).
    The path is assumed a clean straight move; curvature and start/stop
    transients are the printer's, not modelled here.

    Parameters
    ----------
    travel_distance_um : float
        Cartesian move length from the G-code coordinate offset (µm; e.g. a
        10 mm offset = 10000 µm).
    nozzle_diameter_um : float
        Orifice inner diameter (µm).

    Returns
    -------
    float
        Deposited ink volume in nanolitres.
    """
    if travel_distance_um <= 0:
        raise ValueError(f"travel_distance_um must be > 0, got {travel_distance_um!r}")
    if nozzle_diameter_um <= 0:
        raise ValueError(f"nozzle_diameter_um must be > 0, got {nozzle_diameter_um!r}")
    area_um2 = math.pi * (nozzle_diameter_um / 2.0) ** 2
    return area_um2 * travel_distance_um * 1e-6


def print_time_s(travel_distance_um: float, feed_rate_mm_min: float) -> float:
    """Time to traverse a path segment at a G-code feed rate, s.

    .. math:: t = L / v,  \\quad v = \\text{feed\\ rate}

    Distance is in µm, feed rate in mm/min; the result is seconds.
    """
    if travel_distance_um <= 0:
        raise ValueError(f"travel_distance_um must be > 0, got {travel_distance_um!r}")
    if feed_rate_mm_min <= 0:
        raise ValueError(f"feed_rate_mm_min must be > 0, got {feed_rate_mm_min!r}")
    distance_mm = travel_distance_um * 1e-3
    return distance_mm / feed_rate_mm_min * 60.0


def extrusion_rate_nl_min(volume_nl: float, print_time_s: float) -> float:
    """Effective deposition rate of a segment, nL/min.

    ``volume`` over the segment's traversal time — the throughput a design
    actually delivers, useful for matching a cell-droplet count or a target
    fill time.
    """
    if volume_nl <= 0 or print_time_s <= 0:
        raise ValueError("volume_nl and print_time_s must be > 0")
    return volume_nl / print_time_s * 60.0


def filament_mass_ug(volume_nl: float, density_g_cm3: float) -> float:
    """Mass of the deposited ink, µg.

    .. math:: m = \\rho V

    1 nL ink at density ρ (g/cm³ = g/mL) — 1 nL = 10⁻³ µL = 10⁻³ mg = 1 µg at
    ρ = 1 g/cm³, so m[µg] = volume[nL] × ρ[g/cm³].
    """
    if volume_nl <= 0:
        raise ValueError(f"volume_nl must be > 0, got {volume_nl!r}")
    if density_g_cm3 <= 0:
        raise ValueError(f"density_g_cm3 must be > 0, got {density_g_cm3!r}")
    return volume_nl * density_g_cm3


def lines_to_cover(footprint_width_um: float, line_pitch_um: float) -> float:
    """Number of parallel fill lines to cover a footprint of a given width.

    .. math:: n = \\lceil \\text{width} / \\text{pitch} \\rceil

    With ``pitch`` = centre-to-centre spacing (≈ filament width for a dense
    layer). Pure coverage arithmetic; the number of layers is separate.
    """
    if footprint_width_um <= 0:
        raise ValueError(f"footprint_width_um must be > 0, got {footprint_width_um!r}")
    if line_pitch_um <= 0:
        raise ValueError(f"line_pitch_um must be > 0, got {line_pitch_um!r}")
    return math.ceil(footprint_width_um / line_pitch_um)


def path_length_from_offset(dx_um: float, dy_um: float, dz_um: float = 0.0) -> float:
    """Cartesian length of a G-code coordinate offset (µm).

    .. math:: L = \\sqrt{\\Delta x^2 + \\Delta y^2 + \\Delta z^2}

    The length an extrusion volume is computed over from the raw coordinate
    displacement the G-code actually encodes.
    """
    if min(dx_um, dy_um, dz_um) < 0:
        # Offsets can legitimately be any sign; the length only needs |·|.
        pass
    return math.sqrt(dx_um * dx_um + dy_um * dy_um + dz_um * dz_um)


__all__ = [
    "NOZZLE_TABLE",
    "nozzle_diameter_um",
    "nozzle_kind",
    "extrusion_volume_nl",
    "print_time_s",
    "extrusion_rate_nl_min",
    "filament_mass_ug",
    "lines_to_cover",
    "path_length_from_offset",
]
