"""Barrier-function calculators — TEER and permeability of a cell monolayer.

Epithelial/endothelial barrier lines (Caco-2 intestine, hCMEC/D3 blood-brain,
...) are validated and scored by two numbers:

1. **TEER** (trans-endothelial/epithelial electrical resistance) — the ohmic
   resistance of the monolayer plus the electrode blank, normalised to the
   insert area (Ω·cm²). A planned "R total" only becomes a defensible TEER
   after subtracting the blank and multiplying by the area, which is exactly
   the mistake this module prevents.
2. **Papp** (apparent permeability) — the steady-state flux of a probe across
   the monolayer, normalised by area and donor concentration (cm/s). Together
   with TEER it is the standard QC gate before any transport study.

This module also gives the flux, the clearance (permeability-surface-area
product) and the effective Fick permeability of a membrane of known thickness,
so a design can be checked end-to-end in the units the lab actually records.

Units follow the assay conventions: resistance in Ω (or kΩ), area in cm²,
flux in nmol/min, concentration in µM, Papp in cm/s. Reference ranges live in
the physiology registry (:mod:`labwright.physiology`), not here.

References
----------
- TEER = (R_total − R_blank) × A: standard Transwell QC (e.g. Millipore / Costar
  application notes; "TEER" is area-normalised by definition).
- Papp = (dQ/dt) / (A · C₀): the apparent-permeability equation used by the
  intestinal-absorption and BBB transport literature.
- Effective permeability P = D/L (Fick's first law for a thin membrane).
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# TEER
# ---------------------------------------------------------------------------


def teer_ohm_cm2(
    resistance_total_ohm: float,
    resistance_blank_ohm: float,
    area_cm2: float,
) -> float:
    """TEER from raw resistance readings, Ω·cm².

    .. math:: TEER = (R_{total} - R_{blank})\\, A

    Parameters
    ----------
    resistance_total_ohm : float
        Measured total resistance across the insert, Ω.
    resistance_blank_ohm : float
        Resistance of the empty (cell-free) insert — the electrode/medium
        contribution, Ω.
    area_cm2 : float
        Membrane growth area, cm² (24-well Transwell ≈ 0.33, 12-well ≈ 1.12).

    Returns
    -------
    float
        Area-normalised barrier resistance, Ω·cm².
    """
    if resistance_total_ohm <= 0 or resistance_blank_ohm <= 0:
        raise ValueError("resistances must be > 0")
    if area_cm2 <= 0:
        raise ValueError(f"area_cm2 must be > 0, got {area_cm2!r}")
    if resistance_total_ohm < resistance_blank_ohm:
        raise ValueError(
            f"R_total ({resistance_total_ohm} Ω) must exceed R_blank "
            f"({resistance_blank_ohm} Ω) — a monolayer cannot subtract to negative"
        )
    return (resistance_total_ohm - resistance_blank_ohm) * area_cm2


def transendothelial_resistance_ohm(teer_ohm_cm2: float, area_cm2: float) -> float:
    """Raw resistance the insert should read for a target TEER, Ω.

    .. math:: R = TEER / A

    Inverse of :func:`teer_ohm_cm2` — "what R total should my readout show to
    hit this TEER?".
    """
    if teer_ohm_cm2 <= 0 or area_cm2 <= 0:
        raise ValueError("teer_ohm_cm2 and area_cm2 must be > 0")
    return teer_ohm_cm2 / area_cm2


# ---------------------------------------------------------------------------
# Permeability
# ---------------------------------------------------------------------------


def papp_cm_s(
    flux_nmol_min: float,
    area_cm2: float,
    donor_conc_um: float,
) -> float:
    """Apparent permeability Papp from steady-state flux, cm/s.

    .. math:: P_{app} = \\frac{\\dot{n} / 60}{A\\, C_0}

    with flux in nmol/min (÷60 → nmol/s) and donor concentration in µM
    (= nmol/cm³), so the result is cm/s.

    Parameters
    ----------
    flux_nmol_min : float
        Steady-state probe flux across the monolayer, nmol/min.
    area_cm2 : float
        Membrane growth area, cm².
    donor_conc_um : float
        Donor-chamber probe concentration, µM (1 µM = 1 nmol/cm³).

    Returns
    -------
    float
        Apparent permeability in cm/s. Tight barriers read ~1e-7–1e-6 cm/s;
        leaky or paracellular routes push towards 1e-5 cm/s and above.
    """
    if flux_nmol_min < 0 or area_cm2 <= 0 or donor_conc_um <= 0:
        raise ValueError("flux_nmol_min >= 0 and area_cm2, donor_conc_um > 0")
    flux_nmol_s = flux_nmol_min / 60.0
    return flux_nmol_s / (area_cm2 * donor_conc_um)


def flux_nmol_min(
    papp_cm_s: float,
    donor_conc_um: float,
    area_cm2: float,
) -> float:
    """Steady-state probe flux a Papp predicts, nmol/min.

    .. math:: \\dot{n} = P_{app}\\, C_0\\, A\\, 60

    Inverse of :func:`papp_cm_s`. Use to sanity-check a planned measurement:
    a low-permeability probe across a small insert produces pmol/min fluxes,
    which a plate reader may not resolve.
    """
    if papp_cm_s < 0 or donor_conc_um <= 0 or area_cm2 <= 0:
        raise ValueError("papp_cm_s >= 0 and donor_conc_um, area_cm2 > 0")
    return papp_cm_s * donor_conc_um * area_cm2 * 60.0


def clearance_mL_min(papp_cm_s: float, area_cm2: float) -> float:
    """Permeability-surface-area product (clearance), mL/min.

    .. math:: CL = P_{app}\\, A

    The volume of donor phase a monolayer clears per minute — a flow-like
    number that is easier to reason about than a bare Papp.
    """
    if papp_cm_s < 0 or area_cm2 <= 0:
        raise ValueError("papp_cm_s >= 0 and area_cm2 > 0")
    return papp_cm_s * area_cm2 * 60.0  # cm³/s → mL/min


def effective_permeability_cm_s(diffusivity_m2s: float, thickness_um: float) -> float:
    """Effective Fick permeability of a membrane of known thickness, cm/s.

    .. math:: P = D / L

    A purely diffusive reference: how fast a solute crosses a passive membrane
    (e.g. the PDMS or hydrogel). Compare a cell monolayer's Papp against this
    to separate "diffusion-limited" from "barrier-limited" transport.
    """
    if diffusivity_m2s <= 0 or thickness_um <= 0:
        raise ValueError("diffusivity_m2s and thickness_um must be > 0")
    thickness_m = thickness_um * 1e-6
    p_m_s = diffusivity_m2s / thickness_m
    return p_m_s * 100.0  # m/s → cm/s


__all__ = [
    "teer_ohm_cm2",
    "transendothelial_resistance_ohm",
    "papp_cm_s",
    "flux_nmol_min",
    "clearance_mL_min",
    "effective_permeability_cm_s",
]
