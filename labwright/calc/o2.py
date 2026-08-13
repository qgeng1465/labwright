"""Oxygen transport calculators — whether a design keeps cells oxygenated.

Organ-on-chip and spheroid culture live or die on oxygen: a perfused channel
that delivers less O2 than the cells demand goes hypoxic, and a spheroid larger
than its oxygen penetration depth grows a necrotic core. This module computes
those limits so the copilot can answer "will this design starve my cells?"
instead of leaving O2 to prose.

The physics
-----------
1. **Henry's law** — dissolved O2 follows its partial pressure
   (``pO2``). Air-equilibrated culture medium at 37 °C is ~0.2 mM
   (~150 mmHg pO2), the ceiling for a medium that has not been oxygenated.
2. **Krogh penetration depth** — a consuming cell layer depletes O2 as it
   diffuses inward; the penetration depth is :math:`\\delta = \\sqrt{2DC_0/q}`
   (Fick + zero-order consumption), the thickness beyond which O2 is gone.
3. **Spheroid necrotic core** — a sphere of radius ``R`` with penetration
   ``δ`` keeps a shell oxygenated and leaves a necrotic core of radius
   ``R − δ`` (volume fraction ``(1 − δ/R)³``).
4. **Péclet / Damköhler** — how fast flow carries O2 past the cells versus how
   fast diffusion and consumption act. Péclet ≪ 1 means diffusion dominates
   (well mixed); Damköhler > 1 means O2 is consumed faster than flow supplies
   it — depletion risk.
5. **Supply vs demand** — the direct check: perfused O2 delivery at
   air-saturation against the estimated cellular demand.

Units are plain floats named by the argument docstrings. Reference constants
are the standard textbook values, sourced below; a caller with better numbers
(cell-specific OCR measured in-house) should pass them explicitly.

References
----------
- O2 solubility in water/plasma at 37 °C ≈ 1.34 µmol/L/mmHg — the classical
  coefficient α = 0.003 mL O2/dL/mmHg (standard physiology, e.g. Thews et al.,
  *Physiologie des Menschen*). 150 mmHg → ~0.2 mM, the standard "air-equilibrated
  medium ≈ 200 µM" figure used throughout the OOC literature.
- O2 diffusivity in water/tissue at 37 °C ≈ 2 × 10⁻⁹ m²/s (1.5–3 × 10⁻⁹ range;
  diffusion coefficients of O2 in water/tissue, standard tables).
- Krogh penetration model: standard tissue-oxygen mass transfer (Krogh 1919;
  "Oxygen diffusion through tissue" in any tissue-engineering text).
- Spheroid necrotic cores above ~400 µm / O2 diffusion ~200 µm: Drug Metab
  Dispos 2024, doi:10.1124/dmd.124.001653 (also cited in calc/spheroid.py).
"""

from __future__ import annotations

import math

from labwright.calc import microfluidics as mf

#: O2 solubility coefficient in water/plasma at 37 °C, µmol/L per mmHg.
#: 150 mmHg (air, water-saturated) → ~0.2 mM.
O2_HENRY_UMOL_L_MMHG = 1.34

#: Air-equilibrated culture medium at 37 °C, mM. The standard 200 µM.
AIR_SATURATED_O2_MM = 0.2

#: O2 diffusivity in water/tissue at 37 °C, m²/s (1.5–3 × 10⁻⁹ range).
O2_DIFFUSIVITY_M2S = 2e-9


# ---------------------------------------------------------------------------
# Henry's law
# ---------------------------------------------------------------------------


def o2_conc_mm_from_po2(po2_mmHg: float) -> float:
    """Dissolved O2 concentration from partial pressure (Henry's law).

    .. math:: C = \\alpha\\, pO_2

    Parameters
    ----------
    po2_mmHg : float
        O2 partial pressure in mmHg (room air ≈ 150–160 mmHg; pure O2 ≈ 760).

    Returns
    -------
    float
        Dissolved O2 in mM (150 mmHg → ≈ 0.2 mM).
    """
    if not math.isfinite(float(po2_mmHg)) or po2_mmHg < 0:
        raise ValueError(f"po2_mmHg must be finite and >= 0, got {po2_mmHg!r}")
    return po2_mmHg * O2_HENRY_UMOL_L_MMHG * 1e-3


def o2_po2_mmhg_from_conc(conc_mM: float) -> float:
    """O2 partial pressure that gives a dissolved concentration (Henry's law).

    .. math:: pO_2 = C / \\alpha

    Parameters
    ----------
    conc_mM : float
        Dissolved O2 concentration in mM.

    Returns
    -------
    float
        Equivalent pO2 in mmHg (0.2 mM → ≈ 149 mmHg).
    """
    if not math.isfinite(float(conc_mM)) or conc_mM < 0:
        raise ValueError(f"conc_mM must be finite and >= 0, got {conc_mM!r}")
    return conc_mM / (O2_HENRY_UMOL_L_MMHG * 1e-3)


# ---------------------------------------------------------------------------
# Consumption and penetration
# ---------------------------------------------------------------------------


def volumetric_o2_consumption(per_cell_fmol_s: float, cell_density_cells_ml: float) -> float:
    """O2 consumed per volume of tissue (mol/m³/s).

    .. math:: q = \\dot{n}_{cell}\\, \\rho

    Parameters
    ----------
    per_cell_fmol_s : float
        O2 consumption rate per cell, fmol/s (1 fmol/s = 1e-15 mol/s).
        Hepatocytes ≈ 0.4–4.7 nmol/min per 10⁶ cells ≈ 0.007–0.08 fmol/s/cell.
    cell_density_cells_ml : float
        Cell density, cells/mL (a 200 µm spheroid of 1000 × 20 µm cells ≈
        2.4e11 cells/mL).

    Returns
    -------
    float
        Volumetric consumption in mol/m³/s.
    """
    if not math.isfinite(float(per_cell_fmol_s)) or per_cell_fmol_s < 0:
        raise ValueError(f"per_cell_fmol_s must be finite and >= 0, got {per_cell_fmol_s!r}")
    if not math.isfinite(float(cell_density_cells_ml)) or cell_density_cells_ml < 0:
        raise ValueError(f"cell_density_cells_ml must be finite and >= 0, got {cell_density_cells_ml!r}")
    return per_cell_fmol_s * 1e-15 * cell_density_cells_ml * 1e6


def o2_penetration_depth_um(
    volumetric_consumption_mol_m3s: float,
    surface_conc_mM: float = AIR_SATURATED_O2_MM,
    diffusivity_m2s: float = O2_DIFFUSIVITY_M2S,
) -> float:
    """Krogh O2 penetration depth into a consuming tissue layer (µm).

    .. math:: \\delta = \\sqrt{\\frac{2\\,D\\,C_0}{q}}

    The thickness of a uniformly consuming slab that O2 diffusing from the
    surface can still oxygenate. Past this depth the tissue is anoxic.

    Parameters
    ----------
    volumetric_consumption_mol_m3s : float
        Volumetric O2 consumption, mol/m³/s (see
        :func:`volumetric_o2_consumption`).
    surface_conc_mM : float, default 0.2
        O2 concentration at the oxygenated surface, mM (air-saturated medium).
    diffusivity_m2s : float, default 2e-9
        O2 diffusivity in the tissue/medium, m²/s.

    Returns
    -------
    float
        Penetration depth in µm. Dense tissue with high consumption gives
        ~10–100 µm; the empirical "oxygen diffuses ~200 µm" rule corresponds to
        a lower effective consumption (~0.02 mol/m³/s).
    """
    if volumetric_consumption_mol_m3s <= 0:
        raise ValueError(f"volumetric_consumption_mol_m3s must be > 0, got {volumetric_consumption_mol_m3s!r}")
    if surface_conc_mM <= 0:
        raise ValueError(f"surface_conc_mM must be > 0, got {surface_conc_mM!r}")
    if diffusivity_m2s <= 0:
        raise ValueError(f"diffusivity_m2s must be > 0, got {diffusivity_m2s!r}")
    c0 = surface_conc_mM  # 1 mM = 1 mol/m³
    delta_m = math.sqrt(2.0 * diffusivity_m2s * c0 / volumetric_consumption_mol_m3s)
    return delta_m * 1e6


def spheroid_necrotic_fraction(diameter_um: float, penetration_um: float) -> float:
    """Volume fraction of a spheroid that is anoxic (necrotic-core estimate).

    A sphere of radius ``R = diameter/2`` stays oxygenated to depth ``δ``; the
    core beyond is anoxic. Volume fraction of the necrotic core:

    .. math:: f = \\left(1 - \\frac{\\delta}{R}\\right)^3 \\quad (\\delta < R)

    0 when the spheroid is smaller than twice the penetration depth.

    Parameters
    ----------
    diameter_um : float
        Spheroid diameter in µm.
    penetration_um : float
        O2 penetration depth in µm (see :func:`o2_penetration_depth_um`).

    Returns
    -------
    float
        Necrotic volume fraction in [0, 1) (0 = fully oxygenated).
    """
    if diameter_um <= 0:
        raise ValueError(f"diameter_um must be > 0, got {diameter_um!r}")
    if penetration_um <= 0:
        raise ValueError(f"penetration_um must be > 0, got {penetration_um!r}")
    radius = diameter_um / 2.0
    if penetration_um >= radius:
        return 0.0
    return (1.0 - penetration_um / radius) ** 3


# ---------------------------------------------------------------------------
# Demand and dimensionless numbers
# ---------------------------------------------------------------------------


def o2_demand_umol_min(total_cells: float, per_cell_fmol_s: float) -> float:
    """Cellular O2 demand, µmol/min.

    .. math:: \\dot{n} = N\\, \\dot{n}_{cell}\\, 60\\, \\times 10^{6}

    (fmol/s × 1e-15 = mol/s; × 60 = mol/min; × 1e6 = µmol/min.)

    Parameters
    ----------
    total_cells : float
        Total cells in the culture.
    per_cell_fmol_s : float
        O2 consumption rate per cell, fmol/s.

    Returns
    -------
    float
        Demand in µmol/min.
    """
    if not math.isfinite(float(total_cells)) or total_cells < 0:
        raise ValueError(f"total_cells must be finite and >= 0, got {total_cells!r}")
    if not math.isfinite(float(per_cell_fmol_s)) or per_cell_fmol_s < 0:
        raise ValueError(f"per_cell_fmol_s must be finite and >= 0, got {per_cell_fmol_s!r}")
    return total_cells * per_cell_fmol_s * 1e-15 * 60.0 * 1e6


def peclet_number(velocity_mms: float, length_mm: float, diffusivity_m2s: float = O2_DIFFUSIVITY_M2S) -> float:
    """Péclet number — advection vs diffusion of O2 along the channel.

    .. math:: Pe = \\frac{u\\,L}{D}

    Pe ≪ 1: diffusion dominates (well mixed by diffusion); Pe ≫ 1: flow sweeps
    O2 past before it can diffuse to the cells — a diffusion barrier at the
    channel walls.

    Parameters
    ----------
    velocity_mms : float
        Mean flow velocity, mm/s.
    length_mm : float
        Channel length, mm.
    diffusivity_m2s : float, default 2e-9
        O2 diffusivity, m²/s.

    Returns
    -------
    float
        Dimensionless Péclet number.
    """
    if not math.isfinite(float(velocity_mms)) or velocity_mms < 0:
        raise ValueError(f"velocity_mms must be finite and >= 0, got {velocity_mms!r}")
    if length_mm <= 0 or diffusivity_m2s <= 0:
        raise ValueError("length_mm and diffusivity_m2s must be > 0")
    u = velocity_mms * 1e-3  # m/s
    length_m = length_mm * 1e-3  # m
    return u * length_m / diffusivity_m2s


def damkohler_number(
    volumetric_consumption_mol_m3s: float,
    length_mm: float,
    velocity_mms: float,
    inlet_o2_mM: float = AIR_SATURATED_O2_MM,
) -> float:
    """Damköhler I — O2 consumption vs advective supply along the channel.

    .. math:: Da = \\frac{q\\,L}{u\\,C_0}

    Da > 1: cells consume O2 faster than flow supplies it — the outlet is
    depleted and downstream cells are hypoxic. Da ≪ 1: supply is ample.

    Parameters
    ----------
    volumetric_consumption_mol_m3s : float
        Volumetric consumption, mol/m³/s.
    length_mm : float
        Channel length, mm.
    velocity_mms : float
        Mean flow velocity, mm/s.
    inlet_o2_mM : float, default 0.2
        Inlet O2 concentration, mM (air-saturated medium).

    Returns
    -------
    float
        Dimensionless Damköhler number.
    """
    if volumetric_consumption_mol_m3s <= 0:
        raise ValueError(f"volumetric_consumption_mol_m3s must be > 0, got {volumetric_consumption_mol_m3s!r}")
    if length_mm <= 0 or velocity_mms <= 0 or inlet_o2_mM <= 0:
        raise ValueError("length_mm, velocity_mms and inlet_o2_mM must be > 0")
    q = volumetric_consumption_mol_m3s
    length_m = length_mm * 1e-3
    u = velocity_mms * 1e-3
    c0 = inlet_o2_mM  # 1 mM = 1 mol/m³
    return q * length_m / (u * c0)


# ---------------------------------------------------------------------------
# Supply vs demand
# ---------------------------------------------------------------------------


def o2_supply_vs_demand(
    flow_rate_uLmin: float,
    total_cells: float,
    per_cell_fmol_s: float,
    o2_in_mM: float = AIR_SATURATED_O2_MM,
) -> dict[str, float]:
    """Compare perfused O2 supply with cellular demand (best case).

    Supply uses :func:`labwright.calc.microfluidics.o2_delivery_rate` with the
    inlet at ``o2_in_mM`` and outlet at 0 (maximal possible extraction) — the
    most generous reading. If even that cannot meet demand, the culture is
    unambiguously hypoxic.

    Parameters
    ----------
    flow_rate_uLmin : float
        Perfusion flow rate, µL/min.
    total_cells : float
        Total cells in the perfused culture.
    per_cell_fmol_s : float
        O2 consumption rate per cell, fmol/s.
    o2_in_mM : float, default 0.2
        Inlet dissolved O2, mM (air-saturated medium).

    Returns
    -------
    dict
        ``{"supply_umol_min", "demand_umol_min", "ratio", "hypoxic"}`` where
        ``ratio = supply / demand`` and ``hypoxic`` is True when supply < demand.
    """
    supply = mf.o2_delivery_rate(flow_rate_uLmin, o2_in_mM * 1e-3)  # µmol/min
    demand = o2_demand_umol_min(total_cells, per_cell_fmol_s)
    ratio = supply / demand if demand > 0 else float("inf")
    return {
        "supply_umol_min": round(supply, 6),
        "demand_umol_min": round(demand, 6),
        "ratio": round(ratio, 4),
        "hypoxic": bool(supply < demand),
    }


def nmol_min_per_1e6_to_fmol_s(nmol_min_1e6: float) -> float:
    """Convert the registry's OCR unit (nmol/min per 10⁶ cells) to fmol/s/cell.

    .. math:: f = n / 60

    (1 nmol/min per 10⁶ cells = 1 fmol/min per cell = 1/60 fmol/s per cell.)

    Parameters
    ----------
    nmol_min_1e6 : float
        O2 consumption in nmol/min per 10⁶ cells.

    Returns
    -------
    float
        Rate in fmol/s per cell.
    """
    return nmol_min_1e6 / 60.0


__all__ = [
    "O2_HENRY_UMOL_L_MMHG",
    "AIR_SATURATED_O2_MM",
    "O2_DIFFUSIVITY_M2S",
    "o2_conc_mm_from_po2",
    "o2_po2_mmhg_from_conc",
    "volumetric_o2_consumption",
    "o2_penetration_depth_um",
    "spheroid_necrotic_fraction",
    "o2_demand_umol_min",
    "peclet_number",
    "damkohler_number",
    "o2_supply_vs_demand",
    "nmol_min_per_1e6_to_fmol_s",
]
