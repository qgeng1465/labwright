"""Pulsatile / oscillatory flow calculators — cardiac-cycle waveforms for heart-on-chip.

Endothelial and valvular cells do not see steady shear: the heart imposes a
time-varying pressure and flow waveform (60–120 bpm ≈ 1–2 Hz), and the cells
transduce that waveform — its magnitude, its frequency content and, critically,
its reversals — into phenotype. A heart-on-chip perfusion design therefore has
to be checked against the *shape* of the shear the cells actually experience,
not just its time-average. This module computes the numbers that describe that
shape so the copilot can answer "does this flow regime reproduce the cardiac
shear waveform my cells need?"

The physics
-----------
1. **Womersley number** :math:`\\alpha` — the unsteadiness of the flow: angular
   frequency versus the viscous diffusion time across the channel. Small
   :math:`\\alpha \\ll 1` means the flow tracks the driving waveform
   quasi-steadily (parabolic profile at every instant); large
   :math:`\\alpha \\gg 1` means inertia dominates and the velocity profile lags
   and flattens. For a rectangular channel the half-height
   :math:`h/2` plays the role of the tube radius.
2. **Oscillatory shear index (OSI)** — flow reversal: the fraction of a cycle
   over which the wall shear points backwards, compressed to a number in
   [0, 0.5]. OSI = 0 is unidirectional (no reversal), OSI = 0.5 is a purely
   reversing waveform (time-average zero). Reversal is the hallmark of
   atheroprone hemodynamics and is what a valve-cell culture must reproduce or
   deliberately avoid.
3. **Gosling pulsatility index (PI)** — the peak-to-trough swing of the flow
   waveform relative to its mean: how "pulsatile" the flow is in an absolute,
   easy-to-read sense.

Units are plain floats named by the argument docstrings. The standard culture
medium is water-like (:math:`\\rho = 1000` kg/m³, :math:`\\mu = 10^{-3}` Pa·s),
available as the module constants below.

References
----------
- Wang, Xinmei, *microfluidic flow profile generator for heart-on-chip*.
  A microfluidic cardiac inflow waveform generator operating 0.8–2 Hz with
  shear stresses up to 20 dyn/cm²; its demo aortic inflow profile delivers
  average shear 5.9 dyn/cm² (= 0.59 Pa) at 1.2 Hz, with Re = 2.75,
  Womersley :math:`\\alpha` = 0.27 and OSI = 0.2.
  https://oula.finna.fi/Primo/Search?lookfor=Wang%2C+Xinmei
- Womersley number :math:`\\alpha = r\\sqrt{\\omega\\rho/\\mu}` (radius
  :math:`r`, angular frequency :math:`\\omega`, density :math:`\\rho`,
  viscosity :math:`\\mu`) — standard hemodynamics (Womersley 1955; any
  cardiovascular-fluid-mechanics text).
- Oscillatory shear index and pulsatility index: standard definitions used
  across the vascular-shear literature (e.g. Ku et al. 1985; Gosling &
  King 1974).
"""

from __future__ import annotations

import math

#: Standard cell-culture medium density, kg/m³ (≈ water at 37 °C).
MEDIUM_DENSITY_KGM3 = 1000.0

#: Standard cell-culture medium dynamic viscosity, Pa·s (≈ water at 37 °C).
MEDIUM_VISCOSITY_PAS = 1e-3

#: Pa → dyn/cm²: 1 Pa = 10 dyn/cm².
PA_TO_DYN_PER_CM2 = 10.0


# ---------------------------------------------------------------------------
# Womersley number
# ---------------------------------------------------------------------------


def womersley_number(
    frequency_hz: float,
    channel_height_um: float,
    viscosity_pas: float,
    density_kgm3: float,
) -> float:
    """Womersley number — flow unsteadiness of the pulsatile waveform.

    .. math:: \\alpha = \\frac{h}{2}\\sqrt{\\frac{\\omega\\,\\rho}{\\mu}}, \\quad \\omega = 2\\pi f

    with :math:`h/2` the channel half-height, the rectangular-channel analog of
    the tube radius. :math:`\\alpha \\ll 1` — flow is quasi-steady, the
    velocity profile stays parabolic and follows the driving waveform
    instantaneously. :math:`\\alpha \\gg 1` — inertia dominates: the profile
    lags and flattens. A 100 µm channel at 1.2 Hz in water-like medium gives
    :math:`\\alpha \\approx 0.14`; the published heart-on-chip demo waveform
    (Wang, see module docstring) reports :math:`\\alpha = 0.27`, which
    corresponds to the full-height convention (:math:`h\\sqrt{\\omega\\rho/\\mu}`)
    on the same geometry.

    Parameters
    ----------
    frequency_hz : float
        Pulsation frequency, Hz (cardiac cycle ≈ 0.8–2 Hz; 1.2 Hz ≈ 72 bpm).
    channel_height_um : float
        Channel height in µm; half of it is the effective radius.
    viscosity_pas : float
        Dynamic viscosity in Pa·s (culture medium ≈ 1e-3).
    density_kgm3 : float
        Fluid density in kg/m³ (culture medium ≈ 1000).

    Returns
    -------
    float
        Dimensionless Womersley number (unitless).
    """
    if not math.isfinite(float(frequency_hz)) or frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be a positive finite number, got {frequency_hz!r}")
    if not math.isfinite(float(channel_height_um)) or channel_height_um <= 0:
        raise ValueError(f"channel_height_um must be a positive finite number, got {channel_height_um!r}")
    if not math.isfinite(float(viscosity_pas)) or viscosity_pas <= 0:
        raise ValueError(f"viscosity_pas must be a positive finite number, got {viscosity_pas!r}")
    if not math.isfinite(float(density_kgm3)) or density_kgm3 <= 0:
        raise ValueError(f"density_kgm3 must be a positive finite number, got {density_kgm3!r}")
    omega = 2.0 * math.pi * frequency_hz
    half_height_m = channel_height_um * 1e-6 / 2.0
    return half_height_m * math.sqrt(omega * density_kgm3 / viscosity_pas)


# ---------------------------------------------------------------------------
# Oscillatory shear index (flow reversal)
# ---------------------------------------------------------------------------


def oscillatory_shear_index_from_sinusoid(shear_mean_pa: float, shear_amplitude_pa: float) -> float:
    """Oscillatory shear index (OSI) for a sinusoidal shear waveform.

    For :math:`\\tau(t) = \\tau_{mean} + \\tau_{amp}\\sin(\\omega t)`, OSI is
    the standard measure of flow reversal:

    .. math:: OSI = \\frac{1}{2}\\left(1 - \\frac{|\\bar{\\tau}|}{\\overline{|\\tau|}}\\right)

    where the bar is a time-average over one cycle and
    :math:`\\overline{|\\tau|}` is the mean of the absolute shear. Closed form
    for a sinusoid with :math:`\\tau_{amp} \\ge \\tau_{mean} \\ge 0`:

    .. math::
        \\overline{|\\tau|} =
        \\frac{2}{\\pi}\\tau_{amp}\\sqrt{1 - \\left(\\frac{\\tau_{mean}}{\\tau_{amp}}\\right)^2}
        + \\frac{2\\tau_{mean}}{\\pi}\\arcsin\\!\\left(\\frac{\\tau_{mean}}{\\tau_{amp}}\\right)

    When :math:`\\tau_{amp} < \\tau_{mean}` the shear never reverses and
    :math:`\\overline{|\\tau|} = \\tau_{mean}`, giving OSI = 0.

    OSI = 0: strictly unidirectional shear (no reversal). OSI = 0.5: purely
    reversing waveform (time-averaged shear zero). A steady waveform gives 0;
    a purely oscillatory one gives 0.5.

    Parameters
    ----------
    shear_mean_pa : float
        Time-averaged shear stress, Pa (≥ 0).
    shear_amplitude_pa : float
        Amplitude of the sinusoidal component, Pa (≥ 0). The waveform
        :math:`\\tau(t) = \\tau_{mean} + \\tau_{amp}\\sin(\\omega t)` reverses
        only when :math:`\\tau_{amp} > \\tau_{mean}`.

    Returns
    -------
    float
        Oscillatory shear index in [0, 0.5].
    """
    if not math.isfinite(float(shear_mean_pa)) or shear_mean_pa < 0:
        raise ValueError(f"shear_mean_pa must be finite and >= 0, got {shear_mean_pa!r}")
    if not math.isfinite(float(shear_amplitude_pa)) or shear_amplitude_pa < 0:
        raise ValueError(f"shear_amplitude_pa must be finite and >= 0, got {shear_amplitude_pa!r}")
    if shear_amplitude_pa == 0.0:
        # Steady flow — no oscillation, no reversal.
        return 0.0
    if shear_amplitude_pa >= shear_mean_pa:
        # Waveform dips to or below zero: closed-form time-mean of |τ(t)|.
        ratio = shear_mean_pa / shear_amplitude_pa
        mean_abs = (2.0 / math.pi) * shear_amplitude_pa * math.sqrt(max(0.0, 1.0 - ratio * ratio))
        mean_abs += (2.0 * shear_mean_pa / math.pi) * math.asin(min(1.0, ratio))
    else:
        # Waveform stays positive — never reverses.
        mean_abs = shear_mean_pa
    if mean_abs <= 0.0:
        return 0.0
    return 0.5 * (1.0 - shear_mean_pa / mean_abs)


# ---------------------------------------------------------------------------
# Pulsatility
# ---------------------------------------------------------------------------


def pulsatility_index(peak_flow_uLmin: float, minimum_flow_uLmin: float, mean_flow_uLmin: float) -> float:
    """Gosling pulsatility index of a flow waveform.

    .. math:: PI = \\frac{Q_{peak} - Q_{min}}{Q_{mean}}

    The peak-to-trough swing normalised by the mean: PI = 1 is a steady flow
    of constant height, larger values mean a more pulsatile waveform. A
    physiological arterial waveform in a heart-on-chip circuit typically reads
    PI ≳ 1.

    Parameters
    ----------
    peak_flow_uLmin : float
        Peak flow rate over the cycle, µL/min.
    minimum_flow_uLmin : float
        Minimum flow rate over the cycle, µL/min (≥ 0).
    mean_flow_uLmin : float
        Time-averaged flow rate over the cycle, µL/min (> 0).

    Returns
    -------
    float
        Dimensionless Gosling pulsatility index.
    """
    if not math.isfinite(float(mean_flow_uLmin)) or mean_flow_uLmin <= 0:
        raise ValueError(f"mean_flow_uLmin must be a positive finite number, got {mean_flow_uLmin!r}")
    if not math.isfinite(float(minimum_flow_uLmin)) or minimum_flow_uLmin < 0:
        raise ValueError(f"minimum_flow_uLmin must be finite and >= 0, got {minimum_flow_uLmin!r}")
    if not math.isfinite(float(peak_flow_uLmin)) or peak_flow_uLmin < minimum_flow_uLmin:
        raise ValueError(
            f"peak_flow_uLmin must be finite and >= minimum_flow_uLmin, got {peak_flow_uLmin!r}"
        )
    return (peak_flow_uLmin - minimum_flow_uLmin) / mean_flow_uLmin


# ---------------------------------------------------------------------------
# Shear helpers
# ---------------------------------------------------------------------------


def peak_shear_of_sinusoid(shear_mean_pa: float, shear_amplitude_pa: float) -> float:
    """Peak wall shear of a sinusoidal waveform, Pa.

    .. math:: \\tau_{peak} = \\tau_{mean} + \\tau_{amp}

    The maximum of :math:`\\tau(t) = \\tau_{mean} + \\tau_{amp}\\sin(\\omega t)`.
    The published heart-on-chip aortic inflow waveform peaks at
    mean + amplitude and is reported in dyn/cm² (÷10 to get Pa).

    Parameters
    ----------
    shear_mean_pa : float
        Time-averaged shear stress, Pa (≥ 0).
    shear_amplitude_pa : float
        Amplitude of the sinusoidal component, Pa (≥ 0).

    Returns
    -------
    float
        Peak shear in Pa.
    """
    if not math.isfinite(float(shear_mean_pa)) or shear_mean_pa < 0:
        raise ValueError(f"shear_mean_pa must be finite and >= 0, got {shear_mean_pa!r}")
    if not math.isfinite(float(shear_amplitude_pa)) or shear_amplitude_pa < 0:
        raise ValueError(f"shear_amplitude_pa must be finite and >= 0, got {shear_amplitude_pa!r}")
    return shear_mean_pa + shear_amplitude_pa


def shear_dyn_per_cm2_from_pa(shear_pa: float) -> float:
    """Convert shear stress from Pa to dyn/cm².

    .. math:: \\tau_{dyn/cm^2} = 10\\,\\tau_{Pa}

    The cardiovascular/OOC literature overwhelmingly reports shear in dyn/cm²
    (the published heart-on-chip demo waveform is stated as 5.9 dyn/cm² =
    0.59 Pa); internal calculations here use Pa.

    Parameters
    ----------
    shear_pa : float
        Shear stress in Pa (≥ 0).

    Returns
    -------
    float
        Shear stress in dyn/cm².
    """
    if not math.isfinite(float(shear_pa)) or shear_pa < 0:
        raise ValueError(f"shear_pa must be finite and >= 0, got {shear_pa!r}")
    return shear_pa * PA_TO_DYN_PER_CM2


__all__ = [
    "MEDIUM_DENSITY_KGM3",
    "MEDIUM_VISCOSITY_PAS",
    "PA_TO_DYN_PER_CM2",
    "womersley_number",
    "oscillatory_shear_index_from_sinusoid",
    "pulsatility_index",
    "peak_shear_of_sinusoid",
    "shear_dyn_per_cm2_from_pa",
]
