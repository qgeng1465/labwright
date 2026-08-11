"""Microfluidics calculators for organ-on-chip (OOC) device design.

These functions implement the standard analytic solutions used in the
organ-on-chip literature for rectangular microchannels under laminar flow.
Every formula is documented with its equation and reference so values can be
traced and reproduced.

Standard assumptions
--------------------
- Rectangular cross-section, ``height < width`` (typical for OOC culture
  channels: height 50-200 µm, width 400-1000 µm).
- Fully developed, steady, incompressible, laminar flow (Re << 2300).
- Newtonian fluid (cell culture medium is close to water viscosity).
- Low-aspect-ratio approximation: ``w >> h``, for which the wall shear stress
  is ``tau = 6·mu·Q / (w·h^2)`` (error < 1 % once ``w/h > 10``).

Units convention
----------------
All functions take and return plain floats in the units named by their
argument/return docstrings (micrometres, µL/min, Pa, seconds, ...). This keeps
the interface simple enough for an LLM tool-caller to use reliably, while
:mod:`labwright.calc.units` remains available for explicit conversion.
"""

from __future__ import annotations

import math

from labwright.calc.units import Q, ureg

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def hydraulic_diameter(width_um: float, height_um: float) -> float:
    """Hydraulic diameter of a rectangular channel.

    .. math:: D_h = \\frac{2wh}{w + h}

    Parameters
    ----------
    width_um, height_um : float
        Channel width/height in micrometres.

    Returns
    -------
    float
        Hydraulic diameter in micrometres.
    """
    _validate_positive(width_um=width_um, height_um=height_um)
    w, h = Q(width_um, "um"), Q(height_um, "um")
    dh = 2 * w * h / (w + h)
    return dh.to("um").magnitude


def mean_velocity(flow_rate_uLmin: float, width_um: float, height_um: float) -> float:
    """Mean flow velocity in the channel.

    .. math:: \\bar{u} = \\frac{Q}{wh}

    Parameters
    ----------
    flow_rate_uLmin : float
        Volumetric flow rate in µL/min.
    width_um, height_um : float
        Channel width/height in micrometres.

    Returns
    -------
    float
        Mean velocity in mm/s.
    """
    _validate_positive(flow_rate_uLmin=flow_rate_uLmin, width_um=width_um, height_um=height_um)
    q = Q(flow_rate_uLmin, "uL/min")
    area = Q(width_um, "um") * Q(height_um, "um")
    u = q / area
    return u.to("mm/s").magnitude


# ---------------------------------------------------------------------------
# Flow / shear
# ---------------------------------------------------------------------------


def wall_shear_stress(flow_rate_uLmin: float, width_um: float, height_um: float, viscosity_pas: float) -> float:
    """Wall shear stress in a low-aspect-ratio rectangular channel.

    .. math:: \\tau = \\frac{6\\,\\mu Q}{w\\,h^2}

    Reference
    ---------
    Standard parallel-plate / wide-rectangular-channel solution; widely used
    in organ-on-chip design papers (e.g. Bhatia & Ingber, Nat. Biotechnol.
    2014; Huh et al., Science 2010 report ~0.03 Pa microvascular shear).

    Parameters
    ----------
    flow_rate_uLmin : float
        Volumetric flow rate in µL/min.
    width_um, height_um : float
        Channel width/height in micrometres.
    viscosity_pas : float
        Dynamic viscosity in Pa·s (water ~1e-3; culture medium ~0.9-1.1e-3).

    Returns
    -------
    float
        Wall shear stress in Pa. Multiply by 10 for dyn/cm^2.
    """
    _validate_positive(
        flow_rate_uLmin=flow_rate_uLmin, width_um=width_um, height_um=height_um, viscosity_pas=viscosity_pas
    )
    q = Q(flow_rate_uLmin, "uL/min")
    w, h = Q(width_um, "um"), Q(height_um, "um")
    mu = Q(viscosity_pas, "Pa*s")
    tau = 6 * mu * q / (w * h**2)
    return tau.to("Pa").magnitude


def flow_rate_for_shear_stress(
    target_shear_pa: float, width_um: float, height_um: float, viscosity_pas: float
) -> float:
    """Inverse of :func:`wall_shear_stress` — flow rate needed for a target shear.

    .. math:: Q = \\frac{\\tau\\, w\\, h^2}{6\\mu}

    Useful for "I want physiological shear X, what flow do I set?"

    Parameters
    ----------
    target_shear_pa : float
        Target wall shear stress in Pa.
    width_um, height_um : float
        Channel width/height in micrometres.
    viscosity_pas : float
        Dynamic viscosity in Pa·s.

    Returns
    -------
    float
        Required volumetric flow rate in µL/min.
    """
    _validate_positive(
        target_shear_pa=target_shear_pa, width_um=width_um, height_um=height_um, viscosity_pas=viscosity_pas
    )
    tau = Q(target_shear_pa, "Pa")
    w, h = Q(width_um, "um"), Q(height_um, "um")
    mu = Q(viscosity_pas, "Pa*s")
    q = tau * w * h**2 / (6 * mu)
    return q.to("uL/min").magnitude


def reynolds_number(
    flow_rate_uLmin: float, width_um: float, height_um: float, viscosity_pas: float, density_kgm3: float = 1000.0
) -> float:
    """Reynolds number in a rectangular channel.

    .. math:: Re = \\frac{\\rho\\,\\bar{u}\\,D_h}{\\mu}

    Re << 1 (microfluidics) confirms laminar Stokes flow — a sanity check that
    the channel design is in a regime where the analytic formulas hold.

    Parameters
    ----------
    flow_rate_uLmin : float
        Volumetric flow rate in µL/min.
    width_um, height_um : float
        Channel width/height in micrometres.
    viscosity_pas : float
        Dynamic viscosity in Pa·s.
    density_kgm3 : float, default 1000
        Fluid density in kg/m^3.

    Returns
    -------
    float
        Dimensionless Reynolds number.
    """
    _validate_positive(
        flow_rate_uLmin=flow_rate_uLmin,
        width_um=width_um,
        height_um=height_um,
        viscosity_pas=viscosity_pas,
        density_kgm3=density_kgm3,
    )
    rho = Q(density_kgm3, "kg/m**3")
    mu = Q(viscosity_pas, "Pa*s")
    dh = Q(hydraulic_diameter(width_um, height_um), "um")
    u = Q(mean_velocity(flow_rate_uLmin, width_um, height_um), "mm/s")
    return float((rho * u * dh / mu).to_base_units())


def pressure_drop(
    flow_rate_uLmin: float, width_um: float, height_um: float, length_mm: float, viscosity_pas: float
) -> float:
    """Laminar pressure drop along a rectangular channel (low aspect ratio).

    .. math:: \\Delta P = \\frac{12\\,\\mu Q\\,L}{w\\,h^3}

    Reference
    ---------
    Hagen–Poiseuille solution for a wide rectangular duct (Bruns, *Theoretical
    Microfluidics*, 2008, eq. 1.31).

    Parameters
    ----------
    flow_rate_uLmin : float
        Volumetric flow rate in µL/min.
    width_um, height_um : float
        Channel width/height in micrometres.
    length_mm : float
        Channel length in millimetres.
    viscosity_pas : float
        Dynamic viscosity in Pa·s.

    Returns
    -------
    float
        Pressure drop in Pa (1 Pa = 0.0075 mmHg).
    """
    _validate_positive(
        flow_rate_uLmin=flow_rate_uLmin,
        width_um=width_um,
        height_um=height_um,
        length_mm=length_mm,
        viscosity_pas=viscosity_pas,
    )
    q = Q(flow_rate_uLmin, "uL/min")
    w, h = Q(width_um, "um"), Q(height_um, "um")
    L = Q(length_mm, "mm")
    mu = Q(viscosity_pas, "Pa*s")
    dp = 12 * mu * q * L / (w * h**3)
    return dp.to("Pa").magnitude


def residence_time(flow_rate_uLmin: float, width_um: float, height_um: float, length_mm: float) -> float:
    """Mean residence time of fluid in the channel.

    .. math:: t = \\frac{L\\,w\\,h}{Q}

    Parameters
    ----------
    flow_rate_uLmin : float
        Volumetric flow rate in µL/min.
    width_um, height_um : float
        Channel width/height in micrometres.
    length_mm : float
        Channel length in millimetres.

    Returns
    -------
    float
        Residence time in seconds.
    """
    _validate_positive(
        flow_rate_uLmin=flow_rate_uLmin, width_um=width_um, height_um=height_um, length_mm=length_mm
    )
    q = Q(flow_rate_uLmin, "uL/min")
    volume = Q(width_um, "um") * Q(height_um, "um") * Q(length_mm, "mm")
    return (volume / q).to("s").magnitude


def channel_volume(width_um: float, height_um: float, length_mm: float) -> float:
    """Total volume of the culture channel.

    Parameters
    ----------
    width_um, height_um : float
        Channel width/height in micrometres.
    length_mm : float
        Channel length in millimetres.

    Returns
    -------
    float
        Channel volume in µL.
    """
    _validate_positive(width_um=width_um, height_um=height_um, length_mm=length_mm)
    v = Q(width_um, "um") * Q(height_um, "um") * Q(length_mm, "mm")
    return v.to("uL").magnitude


def o2_delivery_rate(flow_rate_uLmin: float, o2_in_mol_L: float, o2_out_mol_L: float = 0.0) -> float:
    """Oxygen delivered to cells per unit time by a perfused channel.

    .. math:: \\dot{n}_{O_2} = Q\\,(C_{in} - C_{out})

    A first-order check that perfusion meets cell O2 demand. Culture medium
    equilibrated with air contains ~200 µM O2 (at 37 °C, ~0.2 mM); many OOC
    studies deliberately deplete it to model hypoxia.

    Parameters
    ----------
    flow_rate_uLmin : float
        Volumetric flow rate in µL/min.
    o2_in_mol_L : float
        Dissolved O2 concentration entering the channel, mol/L.
    o2_out_mol_L : float, default 0.0
        Dissolved O2 concentration leaving the channel, mol/L.

    Returns
    -------
    float
        O2 delivery rate in µmol/min.
    """
    _validate_positive(flow_rate_uLmin=flow_rate_uLmin, o2_in_mol_L=o2_in_mol_L)
    q = Q(flow_rate_uLmin, "uL/min")
    dc = Q(o2_in_mol_L - o2_out_mol_L, "mol/L")
    return (q * dc).to("umol/min").magnitude


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_positive(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "hydraulic_diameter",
    "mean_velocity",
    "wall_shear_stress",
    "flow_rate_for_shear_stress",
    "reynolds_number",
    "pressure_drop",
    "residence_time",
    "channel_volume",
    "o2_delivery_rate",
]
