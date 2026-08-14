"""Pumpless perfusion calculators — gravity-driven (rocking/tilting) chips.

Organ-on-chip culture is often perfused without a syringe pump: the chip sits
on a rocking platform that tilts back and forth, and gravity supplies the
driving pressure. No pump, no tubing, no bubbles — the medium's own weight
does the work. This module turns the platform settings (tilt angle, rocking
period) into the numbers a design actually needs: the hydrostatic pressure
head, the resulting flow rate, the wall shear the cells feel, and how that
shear compares with the physiological target.

The physics
-----------
1. **Hydrostatic head** — tilting a channel of length ``L`` by ``θ`` puts one
   end ``Δh = L·sin(θ)`` above the other, giving ``ΔP = ρ·g·Δh``. This is the
   sole driving force for a rocking platform; flow rate scales with tilt angle
   and oscillation frequency.
2. **Hagen–Poiseuille duct flow** — the head drives a laminar flow through the
   wide rectangular culture channel (``Q = ΔP·w·h³/(12·μ·L)``), and the wall
   shear follows (``τ = ΔP·h/(2·L)`` at peak flow).
3. **Oscillation** — rocking alternates flow direction every half-period. The
   displaced volume per half-cycle sets medium turnover; the oscillatory shear
   index (OSI) separates symmetric bidirectional rocking (OSI → 0.5) from
   unidirectional Tesla-valve chips (OSI → 0).

Source-pinned values (do not change without re-deriving)
--------------------------------------------------------
- Practical rocker tilt angle 0–25°, programmable — MIMETAS OrganoFlow rocker
  (tilt angles to 25°, rocking interval from 5 s).
- Liver sinusoidal endothelial physiological wall shear 0.1–0.5 dyn/cm²
  (= 0.01–0.05 Pa), the range cited for gravity-driven Tesla-valve chips.
- Flow rate scales with tilt angle and oscillation frequency (COMSOL laminar
  simulation of rocking-platform flow).
- Typical rocking half-periods 5–60 s for organ chips.

Standard constants: ``g = 9.81 m/s²``, culture-medium density 1000 kg/m³,
viscosity 1e-3 Pa·s. Reference physiology lives in
:mod:`labwright.physiology`, not here.

References
----------
- MIMETAS OrganoFlow rocker — tilt angles to 25°, rocking interval from 5 s:
  https://www.mimetas.com/organoflow
- Gravity-driven rocking organ-chip platforms (Tesla valves; liver sinusoidal
  WSS 0.1–0.5 dyn/cm² = 0.01–0.05 Pa), Advanced Science 2020:
  https://advanced.onlinelibrary.wiley.com/doi/full-xml/10.1002/advs.202004856
- COMSOL laminar simulation of rocking-platform flow — flow rate scales with
  tilt angle and oscillation frequency:
  https://translation-cn.comsol.com/paper/download/194513/srinivasan_abstract.pdf
- Wide-rectangular-duct Hagen–Poiseuille solution: Bruns, *Theoretical
  Microfluidics*, 2008 (see also calc/microfluidics.py).
"""

from __future__ import annotations

import math

#: Gravitational acceleration, m/s².
GRAVITY_M_S2 = 9.81

#: Culture-medium density, kg/m³ (≈ water).
CULTURE_MEDIUM_DENSITY_KGM3 = 1000.0

#: Culture-medium dynamic viscosity, Pa·s (≈ water).
CULTURE_MEDIUM_VISCOSITY_PAS = 1e-3

#: Practical rocker tilt limit, degrees (MIMETAS OrganoFlow).
ROCKER_TILT_MAX_DEG = 25.0

#: Typical rocking half-periods for organ chips, seconds.
ROCKER_HALF_PERIOD_MIN_S = 5.0
ROCKER_HALF_PERIOD_MAX_S = 60.0

#: Liver sinusoidal endothelial physiological wall shear, Pa
#: (0.1–0.5 dyn/cm²; cited for gravity-driven Tesla-valve chips).
LIVER_SINUSOID_WSS_MIN_PA = 0.01
LIVER_SINUSOID_WSS_MAX_PA = 0.05


# ---------------------------------------------------------------------------
# Driving pressure
# ---------------------------------------------------------------------------


def hydrostatic_pressure_pa(rho_kgm3: float, tilt_angle_deg: float, channel_length_mm: float) -> float:
    """Hydrostatic pressure head from platform tilt, Pa.

    .. math:: \\Delta P = \\rho\\,g\\,L\\,\\sin\\theta

    Tilting a channel of length ``L`` by ``θ`` from horizontal raises one end
    ``Δh = L·sin(θ)`` above the other; the weight of that fluid column is the
    driving pressure of a rocking-platform chip.

    Parameters
    ----------
    rho_kgm3 : float
        Fluid density, kg/m³ (culture medium ≈ 1000).
    tilt_angle_deg : float
        Platform tilt from horizontal, degrees. Practical rocker limit is
        0–25° (MIMETAS OrganoFlow), enforced here.
    channel_length_mm : float
        Length of the perfused channel along the tilt axis, mm.

    Returns
    -------
    float
        Pressure head in Pa.
    """
    if rho_kgm3 <= 0:
        raise ValueError(f"rho_kgm3 must be > 0, got {rho_kgm3!r}")
    if channel_length_mm <= 0:
        raise ValueError(f"channel_length_mm must be > 0, got {channel_length_mm!r}")
    if tilt_angle_deg < 0 or tilt_angle_deg > ROCKER_TILT_MAX_DEG:
        raise ValueError(
            f"tilt_angle_deg must be in [0, {ROCKER_TILT_MAX_DEG}], got {tilt_angle_deg!r} "
            "(practical rocker limit)"
        )
    delta_h_m = channel_length_mm * 1e-3 * math.sin(math.radians(tilt_angle_deg))
    return rho_kgm3 * GRAVITY_M_S2 * delta_h_m


# ---------------------------------------------------------------------------
# Flow and shear from head
# ---------------------------------------------------------------------------


def flow_rate_from_pressure_head(
    dP_pa: float,
    width_um: float,
    height_um: float,
    length_mm: float,
    viscosity_pas: float,
) -> float:
    """Flow driven by a pressure head in a wide rectangular channel, µL/min.

    .. math:: Q = \\frac{\\Delta P\\,w\\,h^3}{12\\,\\mu\\,L}

    Hagen–Poiseuille for a wide rectangular duct (``h ≪ w``), the same
    low-aspect approximation used in :mod:`labwright.calc.microfluidics`. Feed
    the head from :func:`hydrostatic_pressure_pa` to get the flow a tilted
    platform produces through a rocking chip.

    Parameters
    ----------
    dP_pa : float
        Pressure head driving the flow, Pa.
    width_um : float
        Channel width, µm.
    height_um : float
        Channel height, µm.
    length_mm : float
        Channel length along the flow path, mm.
    viscosity_pas : float
        Dynamic viscosity, Pa·s (culture medium ≈ 1e-3).

    Returns
    -------
    float
        Flow rate in µL/min.
    """
    if dP_pa <= 0:
        raise ValueError(f"dP_pa must be > 0, got {dP_pa!r}")
    if width_um <= 0 or height_um <= 0 or length_mm <= 0 or viscosity_pas <= 0:
        raise ValueError("width_um, height_um, length_mm and viscosity_pas must be > 0")
    q_m3s = (
        dP_pa
        * (width_um * 1e-6)
        * (height_um * 1e-6) ** 3
        / (12.0 * viscosity_pas * (length_mm * 1e-3))
    )
    return q_m3s * 1e9 * 60.0  # m³/s → µL/min


def peak_wall_shear_from_head(dP_pa: float, width_um: float, height_um: float, length_mm: float) -> float:
    """Peak wall shear stress from a pressure head, Pa.

    .. math:: \\tau = \\frac{\\Delta P\\,h}{2\\,L}

    Derived from ``τ = 6·μ·Q/(w·h²)`` with ``Q = ΔP·w·h³/(12·μ·L)`` — the
    viscosity and width cancel. This is the peak (fully-developed) wall shear
    during one rocking half-cycle. ``width_um`` is kept in the signature for
    symmetry with :func:`flow_rate_from_pressure_head` and validation but does
    not enter the result.

    Parameters
    ----------
    dP_pa : float
        Pressure head driving the flow, Pa.
    width_um : float
        Channel width, µm (validated, not used in the formula).
    height_um : float
        Channel height, µm.
    length_mm : float
        Channel length along the flow path, mm.

    Returns
    -------
    float
        Peak wall shear stress in Pa.
    """
    if dP_pa <= 0:
        raise ValueError(f"dP_pa must be > 0, got {dP_pa!r}")
    if width_um <= 0 or height_um <= 0 or length_mm <= 0:
        raise ValueError("width_um, height_um and length_mm must be > 0")
    return dP_pa * (height_um * 1e-6) / (2.0 * (length_mm * 1e-3))


# ---------------------------------------------------------------------------
# Rocking dynamics
# ---------------------------------------------------------------------------


def rocking_volume_per_half_cycle_ul(flow_rate_uLmin: float, rocking_half_period_s: float) -> float:
    """Volume displaced during one rocking half-cycle, µL.

    .. math:: V = Q\\,\\frac{t}{60}

    The volume that crosses the channel in one half of a rocking cycle (flow
    in one direction). Multiply by 2 for the volume exchanged per full cycle
    and compare with the channel volume
    (:func:`labwright.calc.microfluidics.channel_volume`) to judge medium
    turnover.

    Parameters
    ----------
    flow_rate_uLmin : float
        Flow rate during the half-cycle, µL/min.
    rocking_half_period_s : float
        Duration of one half of a rocking cycle, s (typical organ-chip rocking
        half-periods are 5–60 s).

    Returns
    -------
    float
        Displaced volume in µL.
    """
    if flow_rate_uLmin <= 0:
        raise ValueError(f"flow_rate_uLmin must be > 0, got {flow_rate_uLmin!r}")
    if rocking_half_period_s <= 0:
        raise ValueError(f"rocking_half_period_s must be > 0, got {rocking_half_period_s!r}")
    return flow_rate_uLmin * rocking_half_period_s / 60.0


def oscillatory_shear_index(shear_peak_forward: float, shear_peak_backward: float) -> float:
    """Oscillatory shear index (OSI) for a bidirectional flow profile.

    .. math:: OSI = \\frac{1}{2}\\left(1 - \\frac{|\\tau_f - \\tau_b|}{\\tau_f + \\tau_b}\\right)

    OSI in [0, 0.5]. Symmetric bidirectional rocking (Tesla-free, ``τ_f ≈ τ_b``)
    gives OSI → 0.5; unidirectional net flow (Tesla-valve chip, ``τ_b ≈ 0``)
    gives OSI → 0. OSI is the standard hemodynamic metric for how oscillatory
    the shear regime is.

    Parameters
    ----------
    shear_peak_forward : float
        Peak wall shear in the forward (net-flow) direction, Pa.
    shear_peak_backward : float
        Peak wall shear in the reverse direction, Pa.

    Returns
    -------
    float
        OSI in [0, 0.5] (dimensionless).
    """
    if shear_peak_forward < 0 or shear_peak_backward < 0:
        raise ValueError("shear_peak_forward and shear_peak_backward must be >= 0")
    if shear_peak_forward + shear_peak_backward <= 0:
        raise ValueError(
            "shear_peak_forward + shear_peak_backward must be > 0 "
            "(at least one direction must carry flow)"
        )
    return 0.5 * (
        1.0 - abs(shear_peak_forward - shear_peak_backward) / (shear_peak_forward + shear_peak_backward)
    )


def cycles_per_hour(rocking_half_period_s: float) -> float:
    """Rocking cycles completed per hour for a given half-period.

    .. math:: f = \\frac{3600}{2\\,t}

    Each full cycle is two half-periods (forward + backward). Typical
    organ-chip rocking half-periods are 5–60 s → 30–360 cycles/h.

    Parameters
    ----------
    rocking_half_period_s : float
        Duration of one half of a rocking cycle, s.

    Returns
    -------
    float
        Full cycles per hour.
    """
    if rocking_half_period_s <= 0:
        raise ValueError(f"rocking_half_period_s must be > 0, got {rocking_half_period_s!r}")
    return 3600.0 / (2.0 * rocking_half_period_s)


# ---------------------------------------------------------------------------
# Physiology comparison
# ---------------------------------------------------------------------------


def shear_ratio_vs_physiological(shear_pa: float, target_pa: float) -> dict[str, float | bool]:
    """Compare a chip's wall shear against a physiological target, Pa.

    .. math:: \\text{ratio} = \\frac{\\tau_{chip}}{\\tau_{target}}

    ``in_range`` is True when the ratio is inside the accepted 0.5–2.0 window
    (half to twice the target). Reference target: liver sinusoidal endothelial
    WSS is 0.1–0.5 dyn/cm² (= 0.01–0.05 Pa), the physiological range cited for
    gravity-driven Tesla-valve chips.

    Parameters
    ----------
    shear_pa : float
        Peak wall shear in the chip, Pa.
    target_pa : float
        Physiological target WSS, Pa (e.g.
        :data:`LIVER_SINUSOID_WSS_MIN_PA` / ``_MAX_PA``).

    Returns
    -------
    dict
        ``{"shear_pa", "target_pa", "ratio", "in_range"}`` where
        ``ratio = shear_pa / target_pa`` and ``in_range`` is True when
        ``0.5 ≤ ratio ≤ 2.0``.
    """
    if shear_pa < 0:
        raise ValueError(f"shear_pa must be >= 0, got {shear_pa!r}")
    if target_pa <= 0:
        raise ValueError(f"target_pa must be > 0, got {target_pa!r}")
    ratio = shear_pa / target_pa
    return {
        "shear_pa": shear_pa,
        "target_pa": target_pa,
        "ratio": round(ratio, 4),
        "in_range": bool(0.5 <= ratio <= 2.0),
    }


__all__ = [
    "GRAVITY_M_S2",
    "CULTURE_MEDIUM_DENSITY_KGM3",
    "CULTURE_MEDIUM_VISCOSITY_PAS",
    "ROCKER_TILT_MAX_DEG",
    "ROCKER_HALF_PERIOD_MIN_S",
    "ROCKER_HALF_PERIOD_MAX_S",
    "LIVER_SINUSOID_WSS_MIN_PA",
    "LIVER_SINUSOID_WSS_MAX_PA",
    "hydrostatic_pressure_pa",
    "flow_rate_from_pressure_head",
    "peak_wall_shear_from_head",
    "rocking_volume_per_half_cycle_ul",
    "oscillatory_shear_index",
    "cycles_per_hour",
    "shear_ratio_vs_physiological",
]
