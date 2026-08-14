"""Breathing-function calculators — ALI and cyclic mechanical stretch.

A lung-on-chip models the two things a ventilated lung does to its alveolar
epithelium every breath: expose it to air on one side (air-liquid interface,
ALI) and stretch it cyclically. This module turns those design decisions into
numbers the copilot can gate on.

The physics
-----------
1. **Breathing frequency** — the chip is paced at a physiological
   respiratory rate. The reference alveoli-on-chip models run ~0.2 Hz
   (12 breaths/min) to 0.25 Hz (15 breaths/min); the module converts the
   actuator frequency (Hz) to the lab-familiar breaths/min.
2. **Physiological strain window** — healthy alveolar tissue is cycled at
   5–12 % linear strain; strains above 20 % linear strain are pathological
   (a potential contributor to IPF). The module classifies a chosen strain
   against that window.
3. **Stretch kinematics** — a stretch of ``ε%`` across a membrane span
   ``L`` moves each edge by ``ε·L``; the (linearised) strain rate is
   ``ε·f`` per breath cycle. The default span is the alveolar-sac scale,
   200–300 µm (``DEFAULT_MEMBRANE_SPAN_UM`` — an assumption; pass the real
   membrane span of your chip to override).
4. **Cycle budget** — total mechanical cycles = frequency × seconds, and the
   stretch duty fraction (stretch time / cycle time) controls how much of
   each cycle the cells actually spend deformed.
5. **ALI film** — with the apical compartment switched to air, the residual
   liquid left on the epithelial surface is a thin film; its thickness is
   apical volume / surface area. ALI is run on an ultra-thin elastic membrane
   whose diaphragm is deformed by vacuum- or electro-pneumatic pressure
   (Huh-style lung chip).

Units are plain floats named by the argument docstrings. Reference ranges live
in the physiology registry (:mod:`labwright.physiology`), not here.

References
----------
- Physiological breathing frequency ≈ 0.2 Hz (12 breaths/min) in the Stucki
  et al. breathing lung-on-chip; 0.25 Hz (15 breaths/min) for 24 h in a
  patient-derived alveoli-on-chip (2025). Methods of Delivering Mechanical
  Stimuli to Organ-on-Chip review:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6843435/ ; patient-derived
  alveoli-on-chip blog:
  https://www.ufluidix.com/blog/mimicking-breathing-a-novel-alveoli-on-chip-model-using-patient-derived-cells/amp/
- Physiological alveolar strain 5–12 % linear strain; >20 % linear strain
  considered pathological (potential IPF) — same PMC6843435 review.
- ALI on an ultra-thin elastic membrane; vacuum- or electro-pneumatic-driven
  diaphragm deformation of a thin PDMS membrane (Huh-style lung chip).
"""

from __future__ import annotations


#: Assumed alveolar-sac membrane span, µm. Physiological scale 200–300 µm;
#: 250 µm is a representative default — pass the real chip span to
#: :func:`cyclic_displacement_um` to override.
DEFAULT_MEMBRANE_SPAN_UM = 250.0

#: Lower bound of the physiological alveolar linear-strain window, %.
PHYSIOLOGICAL_STRAIN_MIN_PCT = 5.0

#: Upper bound of the physiological alveolar linear-strain window, %.
PHYSIOLOGICAL_STRAIN_MAX_PCT = 12.0

#: Linear strain above which is considered pathological (potential IPF), %.
PATHOLOGICAL_STRAIN_PCT = 20.0


# ---------------------------------------------------------------------------
# Breathing rate
# ---------------------------------------------------------------------------


def breaths_per_minute(frequency_hz: float) -> float:
    """Respiratory rate from actuator frequency, breaths/min.

    .. math:: \\text{breaths/min} = f \\cdot 60

    Parameters
    ----------
    frequency_hz : float
        Breathing/actuation frequency, Hz (physiological ≈ 0.2–0.25 Hz,
        i.e. 12–15 breaths/min).

    Returns
    -------
    float
        Respiratory rate in breaths/min (0.2 Hz → 12.0).
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz!r}")
    return frequency_hz * 60.0


# ---------------------------------------------------------------------------
# Strain physiology
# ---------------------------------------------------------------------------


def linear_strain_pct_is_physiological(strain_pct: float) -> dict[str, bool]:
    """Classify a linear strain against the physiological alveolar window.

    .. math::
        \\text{physiological} &= 5 \\le \\varepsilon \\le 12\\,\\%\\\\
        \\text{pathological} &= \\varepsilon > 20\\,\\%

    Strains in (12, 20] % fall between the two flags — neither physiological
    nor (by this criterion) pathological, which a design review should flag.

    Parameters
    ----------
    strain_pct : float
        Applied linear strain, % (10 = 10 % elongation).

    Returns
    -------
    dict
        ``{"strain_pct", "physiological", "pathological"}`` where the two
        booleans follow the window above (5–12 % physiological; >20 %
        pathological).
    """
    if strain_pct < 0:
        raise ValueError(f"strain_pct must be >= 0, got {strain_pct!r}")
    return {
        "strain_pct": strain_pct,
        "physiological": bool(
            PHYSIOLOGICAL_STRAIN_MIN_PCT <= strain_pct <= PHYSIOLOGICAL_STRAIN_MAX_PCT
        ),
        "pathological": bool(strain_pct > PATHOLOGICAL_STRAIN_PCT),
    }


def cyclic_displacement_um(
    strain_pct: float,
    membrane_span_um: float = DEFAULT_MEMBRANE_SPAN_UM,
) -> float:
    """Edge displacement a target strain needs across a membrane span, µm.

    .. math:: \\Delta L = \\frac{\\varepsilon}{100}\\, L

    How far the membrane edge must actually move (vacuum diaphragm stroke)
    to give a chosen linear strain over a given span.

    Parameters
    ----------
    strain_pct : float
        Target linear strain, %.
    membrane_span_um : float, default 250
        Membrane span in µm. Default is the assumed alveolar-sac diameter
        scale (200–300 µm); pass the real chip geometry to override.

    Returns
    -------
    float
        Required edge displacement in µm (10 % over 250 µm → 25.0 µm).
    """
    if strain_pct < 0:
        raise ValueError(f"strain_pct must be >= 0, got {strain_pct!r}")
    if membrane_span_um <= 0:
        raise ValueError(f"membrane_span_um must be > 0, got {membrane_span_um!r}")
    return (strain_pct / 100.0) * membrane_span_um


def strain_rate_per_s(strain_pct: float, frequency_hz: float) -> float:
    """Linearised strain rate per cycle, 1/s.

    .. math:: \\dot{\\varepsilon} = \\frac{\\varepsilon}{100}\\, f

    Linearised rate (strain fraction per cycle × cycles per second). Assumes
    a sinusoidal/ramp stretch spends one full cycle changing length; a true
    rate depends on the waveform's rise time.

    Parameters
    ----------
    strain_pct : float
        Applied linear strain, %.
    frequency_hz : float
        Breathing frequency, Hz.

    Returns
    -------
    float
        Strain rate in 1/s (10 % at 0.2 Hz → 0.02 /s).
    """
    if strain_pct < 0:
        raise ValueError(f"strain_pct must be >= 0, got {strain_pct!r}")
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz!r}")
    return (strain_pct / 100.0) * frequency_hz


def total_cycles(culture_duration_h: float, frequency_hz: float) -> float:
    """Total mechanical stretch cycles over a culture, count.

    .. math:: N = t_{h}\\cdot 3600 \\cdot f

    Parameters
    ----------
    culture_duration_h : float
        Duration of stretch application, h.
    frequency_hz : float
        Breathing frequency, Hz.

    Returns
    -------
    float
        Total cycles (24 h at 0.2 Hz → 17 280 cycles).
    """
    if culture_duration_h < 0:
        raise ValueError(f"culture_duration_h must be >= 0, got {culture_duration_h!r}")
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz!r}")
    return culture_duration_h * 3600.0 * frequency_hz


def stretch_duty_fraction(stretch_seconds: float, cycle_seconds: float) -> float:
    """Fraction of each cycle the membrane is held stretched, dimensionless.

    .. math:: \\text{duty} = \\frac{t_{stretch}}{t_{cycle}}

    Fraction of each cycle the cells actually spend deformed. The result is in
    (0, 1] for any positive stretch; a duty of 1 means the membrane is held
    stretched continuously (no relaxation phase).

    Parameters
    ----------
    stretch_seconds : float
        Time spent at peak stretch per cycle, s.
    cycle_seconds : float
        Full cycle period, s.

    Returns
    -------
    float
        Duty fraction in (0, 1] (0 when stretch time is 0).
    """
    if cycle_seconds <= 0:
        raise ValueError(f"cycle_seconds must be > 0, got {cycle_seconds!r}")
    if stretch_seconds < 0 or stretch_seconds > cycle_seconds:
        raise ValueError(
            f"stretch_seconds must satisfy 0 <= stretch_seconds <= cycle_seconds, "
            f"got {stretch_seconds!r} with cycle_seconds={cycle_seconds!r}"
        )
    return stretch_seconds / cycle_seconds


# ---------------------------------------------------------------------------
# Air-liquid interface
# ---------------------------------------------------------------------------


def ali_liquid_film_um(apical_volume_ul: float, surface_area_cm2: float) -> float:
    """Thickness of the apical liquid film at ALI, µm.

    .. math:: t = \\frac{V}{A}\\qquad (1\\,\\mu\\text{L} = 10^{-3}\\,\\text{cm}^3)

    Under air-liquid interface the apical compartment is drained to a thin
    liquid film on the epithelial surface; thickness = residual volume / area
    (1 µL = 1e-3 cm³, cm → µm via ×10⁴).

    Parameters
    ----------
    apical_volume_ul : float
        Residual apical liquid volume, µL.
    surface_area_cm2 : float
        Apical epithelial surface area, cm² (24-well Transwell ≈ 0.33).

    Returns
    -------
    float
        Film thickness in µm (20 µL over 0.33 cm² → ≈ 606 µm).
    """
    if apical_volume_ul < 0:
        raise ValueError(f"apical_volume_ul must be >= 0, got {apical_volume_ul!r}")
    if surface_area_cm2 <= 0:
        raise ValueError(f"surface_area_cm2 must be > 0, got {surface_area_cm2!r}")
    volume_cm3 = apical_volume_ul * 1e-3  # µL → cm³
    thickness_cm = volume_cm3 / surface_area_cm2
    return thickness_cm * 1e4  # cm → µm


__all__ = [
    "DEFAULT_MEMBRANE_SPAN_UM",
    "PHYSIOLOGICAL_STRAIN_MIN_PCT",
    "PHYSIOLOGICAL_STRAIN_MAX_PCT",
    "PATHOLOGICAL_STRAIN_PCT",
    "breaths_per_minute",
    "linear_strain_pct_is_physiological",
    "cyclic_displacement_um",
    "strain_rate_per_s",
    "total_cycles",
    "stretch_duty_fraction",
    "ali_liquid_film_um",
]
