"""Allometric / physiological scaling calculators for multi-organ body-on-chip.

A multi-organ MPS ("body-on-chip") only models the human body if its organs are
present in the right proportions: cell numbers scaled to organ-mass ratios and
flows scaled to cardiac-output fractions. Put a liver the size of a kidney on a
chip and the "hepatotoxicity" readout is a kidney with liver-flavoured cells.
This module turns the two physiology tables — fractional cardiac output per
organ and typical organ mass — into the numbers a design actually sets: how
many cells per organ compartment and how fast to perfuse it.

The physics
-----------
1. **Organ flow fractions** — each organ receives a fixed fraction of cardiac
   output (liver 0.27, kidneys 0.22, brain 0.15, ...). A chip perfused at an
   organ's true share of a scaled cardiac output keeps the relative drug
   exposure the body would deliver.
2. **Mass-proportional cell scaling** — an organ's cells on chip are its mass
   fraction of the body times the chip's total cell budget. This preserves
   organ-mass ratios but treats each organ as uniform tissue.
3. **Allometric metabolic scaling** — metabolic rate does not scale linearly
   with mass: Kleiber's law (exponent 0.75) says a 10×-heavier organ needs
   only ~5.6× the metabolic flow. The exponent-0.75 factor is a first-order
   correction to the naive mass fraction.
4. **Transit / residence time** — how long perfusate spends in an organ
   compartment. Matching the in-vivo organ transit time is the flow-side
   objective for a body-on-chip.

*Honest caveat.* Scaling by organ mass is a first approximation. Metabolic rate
scales nonlinearly with mass (Kleiber), organ cell densities and
metabolic rates per cell differ across tissues, and a single "body mass" glosses
over composition. These calculators give a defensible starting design; a lab with
measured cell numbers or per-cell metabolic rates should pass them explicitly.

References
----------
- Ucciferri, Sbrana, & Ahluwalia (2014), "Allometric Scaling and Cell Ratios
  in Multi-Organ in vitro Models of Human Metabolism", Frontiers in
  Bioengineering and Biotechnology, doi:10.3389/fbioe.2014.00074,
  https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2014.00074/text
- Wikswo et al. (2013), "Scaling and systems biology for integrating multiple
  organs-on-a-chip", Lab Chip, https://www.vanderbilt.edu/viibre/organs-on-a-chip.php

Standards used here: adult cardiac output ≈ 5 L/min = 5000 mL/min; adult body
mass ≈ 70 kg = 70000 g.
"""

from __future__ import annotations

import math

#: Fraction of cardiac output perfusing each organ (standard physiology table,
#: adult human). Lungs is the systemic (~0.10) bronchial-plus-total flow;
#: note the lungs receive the full cardiac output as pulmonary flow, which is
#: not the fraction recorded here.
ORGAN_FLOW_FRACTIONS: dict[str, float] = {
    "liver": 0.27,
    "kidneys": 0.22,
    "brain": 0.15,
    "heart": 0.05,
    "gut": 0.15,
    "skin": 0.06,
    "muscle": 0.15,
    "lungs": 0.10,
}

#: Typical adult organ mass, g (order-of-magnitude, standard physiology).
ORGAN_MASS_G: dict[str, float] = {
    "liver": 1500,
    "kidneys": 300,
    "brain": 1400,
    "heart": 300,
    "gut": 1100,
    "skin": 3600,
    "lungs": 1000,
    "muscle": 28000,
}

#: Standard adult cardiac output, mL/min (5 L/min).
CARDIAC_OUTPUT_MLMIN = 5000

#: Standard adult body mass, g (70 kg).
BODY_MASS_G = 70000


# ---------------------------------------------------------------------------
# Flow scaling
# ---------------------------------------------------------------------------


def organ_flow_fraction(organ: str) -> float:
    """Fraction of cardiac output perfusing an organ.

    .. math:: f_{organ} = \\frac{Q_{organ}}{Q_{total}}

    The fraction is the organ's share of total cardiac output (liver 0.27,
    kidneys 0.22, brain 0.15, ...). Perfuse the chip organ at this fraction of
    a scaled cardiac output to reproduce in-vivo relative exposure.

    Parameters
    ----------
    organ : str
        Organ name, one of the keys of :data:`ORGAN_FLOW_FRACTIONS`
        (``liver``, ``kidneys``, ``brain``, ``heart``, ``gut``, ``skin``,
        ``muscle``, ``lungs``).

    Returns
    -------
    float
        Dimensionless fraction of cardiac output (0.05–0.27).
    """
    if organ not in ORGAN_FLOW_FRACTIONS:
        valid = ", ".join(sorted(ORGAN_FLOW_FRACTIONS))
        raise ValueError(
            f"unknown organ {organ!r}; valid organs are: {valid}"
        )
    return ORGAN_FLOW_FRACTIONS[organ]


def organ_flow_rate_mlmin(organ: str, cardiac_output_mlmin: float) -> float:
    """Perfusion flow an organ receives at a cardiac output, mL/min.

    .. math:: Q_{organ} = f_{organ}\\, Q_{total}

    Parameters
    ----------
    organ : str
        Organ name, one of the keys of :data:`ORGAN_FLOW_FRACTIONS`.
    cardiac_output_mlmin : float
        Cardiac output the chip is scaled to, mL/min
        (adult ≈ 5000 mL/min; a 1:1000 scaled body ≈ 5 mL/min).

    Returns
    -------
    float
        Organ perfusion flow in mL/min (liver at 5000 mL/min → ≈ 1350).
    """
    if not math.isfinite(float(cardiac_output_mlmin)) or cardiac_output_mlmin <= 0:
        raise ValueError(
            f"cardiac_output_mlmin must be finite and > 0, got {cardiac_output_mlmin!r}"
        )
    return organ_flow_fraction(organ) * cardiac_output_mlmin


# ---------------------------------------------------------------------------
# Cell and metabolic scaling
# ---------------------------------------------------------------------------


def scale_cell_number(
    organ_mass_g: float,
    body_mass_g: float,
    total_cells_chip: float,
) -> float:
    """Cells an organ compartment should hold, mass-proportional scaling.

    .. math:: N_{organ} = \\frac{m_{organ}}{m_{body}}\\, N_{chip}

    Distributes the chip's total cell budget in proportion to organ mass,
    preserving body organ-mass ratios. A 1.5 kg liver in a 70 kg body with a
    10⁶-cell budget gets ≈ 21429 cells.

    Parameters
    ----------
    organ_mass_g : float
        Mass of the modeled organ, g (see :data:`ORGAN_MASS_G`).
    body_mass_g : float
        Reference body mass, g (adult ≈ 70000 g).
    total_cells_chip : float
        Total cells the chip culture can support.

    Returns
    -------
    float
        Cells assigned to the organ compartment.
    """
    if not math.isfinite(float(organ_mass_g)) or organ_mass_g <= 0:
        raise ValueError(f"organ_mass_g must be finite and > 0, got {organ_mass_g!r}")
    if not math.isfinite(float(body_mass_g)) or body_mass_g <= 0:
        raise ValueError(f"body_mass_g must be finite and > 0, got {body_mass_g!r}")
    if not math.isfinite(float(total_cells_chip)) or total_cells_chip < 0:
        raise ValueError(
            f"total_cells_chip must be finite and >= 0, got {total_cells_chip!r}"
        )
    return (organ_mass_g / body_mass_g) * total_cells_chip


def allometric_metabolic_scale(
    organ_mass_g: float,
    body_mass_g: float,
    exponent: float = 0.75,
) -> float:
    """Allometric metabolic scaling factor for an organ (Kleiber-type).

    .. math:: s = \\left(\\frac{m_{organ}}{m_{body}}\\right)^{k}

    Exponent ``k = 1`` is mass-proportional scaling (uniform tissue assumption).
    Exponent ``k = 0.75`` is Kleiber's metabolic allometry — metabolic rate
    grows slower than mass, so a heavier organ needs proportionally less
    metabolic flow than its mass fraction alone would suggest.

    Parameters
    ----------
    organ_mass_g : float
        Mass of the modeled organ, g.
    body_mass_g : float
        Reference body mass, g.
    exponent : float, default 0.75
        Allometric exponent: 1 = mass-proportional, 0.75 = Kleiber metabolic.

    Returns
    -------
    float
        Dimensionless scaling factor (mass fraction raised to ``exponent``);
        liver (1500/70000)⁰·⁷⁵ ≈ 0.056.
    """
    if not math.isfinite(float(organ_mass_g)) or organ_mass_g <= 0:
        raise ValueError(f"organ_mass_g must be finite and > 0, got {organ_mass_g!r}")
    if not math.isfinite(float(body_mass_g)) or body_mass_g <= 0:
        raise ValueError(f"body_mass_g must be finite and > 0, got {body_mass_g!r}")
    if not math.isfinite(float(exponent)) or exponent <= 0:
        raise ValueError(f"exponent must be finite and > 0, got {exponent!r}")
    return (organ_mass_g / body_mass_g) ** exponent


# ---------------------------------------------------------------------------
# Transit / residence time
# ---------------------------------------------------------------------------


def transit_time_s(volume_ul: float, flow_rate_ulmin: float) -> float:
    """Perfusate transit time through an organ compartment, seconds.

    .. math:: t = \\frac{V}{Q}\\, 60

    Volume/flow in minutes converted to seconds. Long vs short transit changes
    how much metabolism and clearance a drug sees before exiting the organ.

    Parameters
    ----------
    volume_ul : float
        Compartment (channel + chamber) volume, µL.
    flow_rate_ulmin : float
        Perfusion flow through the compartment, µL/min.

    Returns
    -------
    float
        Transit time in seconds (1 mL at 100 µL/min → 600 s).
    """
    if not math.isfinite(float(volume_ul)) or volume_ul < 0:
        raise ValueError(f"volume_ul must be finite and >= 0, got {volume_ul!r}")
    if not math.isfinite(float(flow_rate_ulmin)) or flow_rate_ulmin <= 0:
        raise ValueError(
            f"flow_rate_ulmin must be finite and > 0, got {flow_rate_ulmin!r}"
        )
    return (volume_ul / flow_rate_ulmin) * 60.0


def residence_time_match_error_s(
    volume_ul: float,
    flow_rate_ulmin: float,
    target_transit_s: float,
) -> float:
    """Absolute error between achieved and target transit time, seconds.

    .. math:: \\Delta t = \\left|\\, t_{transit} - t_{target}\\right|

    The flow-side objective for a body-on-chip: pick the flow (or volume) so
    the compartment's transit matches the in-vivo organ transit time. This is
    the residual to minimise — 0 is a perfect match.

    Parameters
    ----------
    volume_ul : float
        Compartment volume, µL.
    flow_rate_ulmin : float
        Perfusion flow, µL/min.
    target_transit_s : float
        Target in-vivo transit time, seconds.

    Returns
    -------
    float
        Absolute difference between achieved and target transit time in seconds.
    """
    transit = transit_time_s(volume_ul, flow_rate_ulmin)
    if not math.isfinite(float(target_transit_s)) or target_transit_s < 0:
        raise ValueError(
            f"target_transit_s must be finite and >= 0, got {target_transit_s!r}"
        )
    return abs(transit - target_transit_s)


# ---------------------------------------------------------------------------
# Chip design
# ---------------------------------------------------------------------------


def chip_scale_factor_for_organ(
    organ_mass_g: float,
    body_mass_g: float,
    chip_cell_capacity: float,
) -> dict:
    """Design summary: scale factor and cells for one organ on a chip.

    .. math:: \\text{scale} = \\frac{m_{organ}}{m_{body}}, \\quad
              N_{chip} = \\text{scale}\\, N_{capacity}

    Convenience wrapper over :func:`scale_cell_number` returning the three
    numbers a designer sets for one organ compartment.

    Parameters
    ----------
    organ_mass_g : float
        Mass of the modeled organ, g.
    body_mass_g : float
        Reference body mass, g.
    chip_cell_capacity : float
        Total cells the chip can culture (the budget being scaled).

    Returns
    -------
    dict
        ``{"organ", "scale", "cells_in_chip"}`` — the ``organ`` mass this
        compartment models, the mass fraction (``scale``), and the cells
        assigned to it from the chip's capacity.
    """
    if not math.isfinite(float(organ_mass_g)) or organ_mass_g <= 0:
        raise ValueError(f"organ_mass_g must be finite and > 0, got {organ_mass_g!r}")
    if not math.isfinite(float(body_mass_g)) or body_mass_g <= 0:
        raise ValueError(f"body_mass_g must be finite and > 0, got {body_mass_g!r}")
    if not math.isfinite(float(chip_cell_capacity)) or chip_cell_capacity < 0:
        raise ValueError(
            f"chip_cell_capacity must be finite and >= 0, got {chip_cell_capacity!r}"
        )
    scale = organ_mass_g / body_mass_g
    return {
        "organ": organ_mass_g,
        "scale": round(scale, 8),
        "cells_in_chip": round(scale * chip_cell_capacity, 4),
    }


__all__ = [
    "ORGAN_FLOW_FRACTIONS",
    "ORGAN_MASS_G",
    "CARDIAC_OUTPUT_MLMIN",
    "BODY_MASS_G",
    "organ_flow_fraction",
    "organ_flow_rate_mlmin",
    "scale_cell_number",
    "allometric_metabolic_scale",
    "transit_time_s",
    "residence_time_match_error_s",
    "chip_scale_factor_for_organ",
]
