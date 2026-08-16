"""Solvent handling & evaporation — hanging-drop and multi-well plate models.

Crystallization screens (96-well *sitting*/*hanging* drop) and long incubations
on multi-well plates both ask the same question: how much solvent is left, and
how fast is it leaving, under a given humidity and temperature?

Two documented models are used, each source-pinned and each parameterised from
**stated** conditions — no hidden literature claims:

**Langmuir diffusion-limited droplet evaporation (the d²-law).** A small
spherical droplet of water loses mass at the diffusion-limited rate

.. math:: \\frac{dM}{dt} = 4\\pi\\, r\\, D\\,(C_{sat} - C_\\infty)

with :math:`D` the diffusivity of water vapour in air, :math:`C_{sat}` the
saturation concentration at the drop surface and :math:`C_\\infty` the ambient
vapour concentration. Writing :math:`C_\\infty = \\mathrm{RH}\\,C_{sat}` and
:math:`V = \\frac{4}{3}\\pi r^3` turns this into a constant rate of decline of
:math:`V^{2/3}` — the classic *d²-law*:

.. math:: V^{2/3}(t) = V_0^{2/3} - c\\,t, \\qquad
   c = \\tfrac{2}{3}\\left(\\tfrac{3}{4\\pi}\\right)^{1/3}\\frac{4\\pi\\,D\\,C_{sat}(1-\\mathrm{RH})}{\\rho}

so a volume can be projected forward from *initial volume + temperature +
relative humidity + elapsed time*. ``C_sat`` comes from the Magnus formula for
saturation vapour pressure and the ideal-gas conversion to g/m³. The
*instantaneous* rate is

.. math:: \\dot V = \\frac{4\\pi\\,D\\,C_{sat}(1-\\mathrm{RH})}{\\rho}\\,r

which is what ``evaporation_rate_ul_hr`` returns at a given drop size (larger
drops have a larger surface area and evaporate faster).

**Plate edge effect.** Peripheral wells of a 96-well plate (rows A/H, columns
1/12) are documented to evaporate several-fold faster than interior wells under
the same lid conditions, because they sit at the plate perimeter where local
humidity is lower and airflow higher. This module exposes the factor as a
**documented-range parameter** (default 1.5×, documented range ≈ 1.4–2.0×),
never as a measurement.

These two models are standard physical/engineering descriptions (aerosol
physics and microplate culture practice), *not* measurements made in this
project. Any humidity/temperature/evaporation constant is either a documented
standard value (D, Magnus coefficients) or an explicit parameter the design
states.

References
----------
- d²-law / diffusion-limited droplet evaporation: Langmuir, Phys. Rev.
  12:368–370 (1918); standard in aerosol physics (Hinds, *Aerosol Technology*).
- Water-vapour diffusivity in air D ≈ 2.4e-5 m²/s at ~25 °C: standard tabulated
  value (e.g. Perry's Chemical Engineers' Handbook, transport properties).
- Magnus formula for saturation vapour pressure: standard engineering
  approximation (e.g. commonly cited in hygrometry standards).
- Plate edge-effect on evaporation: documented microplate-practice phenomenon
  (peripheral-well evaporation bias in long incubations); the factor is a
  documented range, parameterised here.
"""

from __future__ import annotations

import math

#: Diffusivity of water vapour in air, m²/s (~25 °C), standard tabulated value.
DIFFUSIVITY_WATER_AIR = 2.4e-5

#: Density of liquid water, g/m³.
WATER_DENSITY_G_M3 = 997_000.0

#: Water molar mass, kg/mol (for the ideal-gas concentration conversion).
WATER_MOLAR_MASS_KG_MOL = 0.018015

#: Magnus-formula coefficients (saturation vapour pressure, Pa) — standard set.
_MAGNUS_A = 610.94
_MAGNUS_B = 17.625
_MAGNUS_C = 243.04

#: Documented edge-effect range and default for 96-well plates.
EDGE_FACTOR_RANGE = (1.4, 2.0)
EDGE_FACTOR_DEFAULT = 1.5


def saturation_conc_g_m3(temp_c: float) -> float:
    """Saturation water-vapour concentration, g/m³ (Magnus + ideal gas).

    .. math:: P_{sat} = 610.94\\,\\exp\\left(\\frac{17.625\\,T}{T+243.04}\\right)
    .. math:: C_{sat} = 1000\\,\\frac{M_w\\,P_{sat}}{R\\,T}

    T in °C; the Magnus formula yields Pa and the ideal-gas conversion (×1000 to
    go from kg/m³ to g/m³) yields g/m³. Valid across typical lab temperatures
    (0–50 °C).
    """
    if temp_c < 0 or temp_c > 50:
        raise ValueError(f"temp_c must be in [0, 50] °C, got {temp_c!r}")
    t_k = temp_c + 273.15
    p_sat_pa = _MAGNUS_A * math.exp(_MAGNUS_B * temp_c / (_MAGNUS_C + temp_c))
    return p_sat_pa * WATER_MOLAR_MASS_KG_MOL / (8.314462618 * t_k) * 1000.0


def _rate_coefficient_m2s(temp_c: float, rh: float) -> float:
    """Langmuir d²-law coefficient, m²/s.

    .. math:: K = \\frac{4\\pi\\,D\\,C_{sat}(1-\\mathrm{RH})}{\\rho}

    so that dV/dt = K·r with r in m (m³/s) — the rate per metre of drop radius.
    """
    c_sat_g_m3 = saturation_conc_g_m3(temp_c)
    if not 0.0 <= rh <= 1.0:
        raise ValueError(f"rh must be in [0, 1], got {rh!r}")
    return (4.0 * math.pi * DIFFUSIVITY_WATER_AIR * c_sat_g_m3
            * (1.0 - rh) / WATER_DENSITY_G_M3)


def _radius_m(volume_ul: float) -> float:
    """Radius of a spherical drop of the given volume, m."""
    return (3.0 * (volume_ul * 1e-9) / (4.0 * math.pi)) ** (1.0 / 3.0)


def evaporation_rate_ul_hr(drop_volume_ul: float, temp_c: float = 25.0,
                           rh: float = 0.60) -> float:
    """Instantaneous evaporation rate of a drop at its current volume, µL/hr.

    .. math:: \\dot V = K\\, r(V), \\quad K = \\frac{4\\pi\\,D\\,C_{sat}(1-\\mathrm{RH})}{\\rho}

    Larger drops evaporate faster (bigger surface area); the rate is what the
    d²-law rate *is* at that instant.
    """
    if drop_volume_ul <= 0:
        raise ValueError(f"drop_volume_ul must be > 0, got {drop_volume_ul!r}")
    k = _rate_coefficient_m2s(temp_c, rh)
    rate_m3_s = k * _radius_m(drop_volume_ul)  # m³/s
    return rate_m3_s * 1e9 * 3600.0  # m³/s → µL/hr


def drop_volume_after_time(drop_volume_ul: float, hours: float,
                           temp_c: float = 25.0, rh: float = 0.60,
                           evaporation_factor: float = 1.0) -> float:
    """Remaining hanging-drop volume after a time at a stated humidity, µL.

    Integrates the d²-law to a closed form. Because the drop shrinks as it
    evaporates, the *rate slows* over time — the volume after ``hours`` is

    .. math:: V(t) = \\left(V_0^{2/3} - c\\,f\\,t\\right)^{3/2}

    with :math:`c = \\tfrac{2}{3}K(3/4\\pi)^{1/3}` (in V^⅔ units) and ``f`` an
    optional per-well rate multiplier (e.g. the plate edge factor), clipped at 0
    (the drop dries out; volume can never go negative).
    """
    if drop_volume_ul <= 0:
        raise ValueError(f"drop_volume_ul must be > 0, got {drop_volume_ul!r}")
    if hours < 0:
        raise ValueError(f"hours must be >= 0, got {hours!r}")
    if evaporation_factor <= 0:
        raise ValueError(f"evaporation_factor must be > 0, got {evaporation_factor!r}")
    k = _rate_coefficient_m2s(temp_c, rh)
    # c in (µL)^(2/3) / hr: m²/s × (µL^(2/3)/m² = 1e6) × (3600 s/hr)
    c_v23_per_hr = (2.0 / 3.0) * k * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * 1e6 * 3600.0
    v23 = max(0.0, drop_volume_ul ** (2.0 / 3.0) - c_v23_per_hr * evaporation_factor * hours)
    return v23 ** (3.0 / 2.0)


def edge_well_factor(row: str, col: int, edge_factor: float = EDGE_FACTOR_DEFAULT) -> float:
    """Evaporation multiplier for a 96-well position (edge wells evaporate faster).

    Peripheral wells — row A/H or column 1/12 — get the ``edge_factor`` (default
    1.5×, documented range 1.4–2.0×); interior wells get 1.0×. The factor is a
    documented phenomenon parameter, not a measurement.
    """
    r = str(row).strip().upper()
    if len(r) != 1 or r not in "ABCDEFGH":
        raise ValueError(f"row must be A–H, got {row!r}")
    if not isinstance(col, int) or not 1 <= col <= 12:
        raise ValueError(f"col must be an int in 1–12, got {col!r}")
    if not EDGE_FACTOR_RANGE[0] <= edge_factor <= EDGE_FACTOR_RANGE[1]:
        raise ValueError(f"edge_factor must be in {EDGE_FACTOR_RANGE}, got {edge_factor!r}")
    if r in "AH" or col in (1, 12):
        return float(edge_factor)
    return 1.0


def effective_evaporation_rate_ul_hr(drop_volume_ul: float, row: str, col: int,
                                     temp_c: float = 25.0, rh: float = 0.60,
                                     edge_factor: float = EDGE_FACTOR_DEFAULT) -> float:
    """Per-well effective evaporation rate, µL/hr — interior rate × edge factor.

    The quantity a crystallization screen or cell-culture evaporation budget
    actually needs per well: the Langmuir interior rate corrected for the well's
    plate position.
    """
    base = evaporation_rate_ul_hr(drop_volume_ul, temp_c=temp_c, rh=rh)
    return base * edge_well_factor(row, col, edge_factor=edge_factor)


def evaporation_gradient_ul_hr(volumes_ul: list[float], positions: list[tuple[str, int]],
                               temp_c: float = 25.0, rh: float = 0.60) -> list[float]:
    """Per-well effective rates across a plate — an evaporation *gradient*.

    ``volumes_ul[i]`` and ``positions[i] = (row, col)`` are zipped; interior
    wells differ from edge wells by the documented edge factor. This is the
    "evaporation gradient across the plate" a multi-well assay must account for.
    """
    if len(volumes_ul) != len(positions):
        raise ValueError("volumes_ul and positions must have equal length")
    return [
        effective_evaporation_rate_ul_hr(v, row, col, temp_c=temp_c, rh=rh)
        for v, (row, col) in zip(volumes_ul, positions)
    ]


__all__ = [
    "DIFFUSIVITY_WATER_AIR",
    "EDGE_FACTOR_RANGE",
    "EDGE_FACTOR_DEFAULT",
    "saturation_conc_g_m3",
    "drop_volume_after_time",
    "evaporation_rate_ul_hr",
    "edge_well_factor",
    "effective_evaporation_rate_ul_hr",
    "evaporation_gradient_ul_hr",
]
