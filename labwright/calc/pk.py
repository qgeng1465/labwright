"""Pharmacokinetics of a perfused organ-on-chip system.

The core question an OOC ADME experiment answers: *how fast does the chip clear
the drug?* Given the inlet and outlet concentrations and the perfusion flow
rate, the chip's first-pass drug handling reduces to three numbers —

- **extraction ratio** ``E = 1 − C_out/C_in`` — the fraction of drug removed in
  one pass (dimensionless, 0 = no clearance, 1 = complete clearance);
- **clearance** ``Cl = Q·E`` — the volume of medium fully cleared of drug per
  unit time (the "volume cleared" formulation, µL/min);
- and, when the recirculating system volume ``V`` is known, **elimination
  half-life** ``t½ = ln2·V/Cl`` and the **accumulation ratio** under repeated
  dosing, ``R = 1/(1 − e^{−ln2·τ/t½})`` — how much a fixed-interval dose
  regimen builds up at steady state.

A fourth number — **mass cleared** ``M = Cl·C_in·MW·6e-5`` — converts the
volume-based clearance into an absolute amount removed (µg/h), which is what a
bioanalysis lab actually measures.

All functions are pure (same inputs → same outputs) and unit-tested against the
governing equations. The units are the easy place to go wrong (µL/min vs
mL/min, µM vs µM vs pmol/µL, and the 6e-5 factor that carries both the
µL→L and mol→µg conversions); the calculator owns that arithmetic so the model
never has to.

References
----------
- Rowland & Tozer, *Clinical Pharmacokinetics and Pharmacodynamics*, 4th ed. —
  extraction ratio E = (C_in − C_out)/C_in, clearance Cl = Q·E, and the
  single-compartment t½ = ln2·V/Cl.
- Gibaldi & Perrier, *Pharmacokinetics*, 2nd ed. — accumulation ratio under
  repeated dosing R = 1/(1 − e^{−k·τ}), k = ln2/t½.
"""

from __future__ import annotations

import math

_LN2 = math.log(2.0)

#: Molecular weights of the probe compounds the extractor was trained on and
#: the benchmark golds use (g/mol). The verifier cross-checks a model-reported
#: molecular weight against this table when the compound is known, so a design
#: cannot claim "warfarin has MW 464" and still pass the gate. A compound not
#: listed here (the user's own drug) is never checked — only probes with pinned
#: values are.
COMPOUND_MW: dict[str, float] = {
    "diclofenac": 296.1,
    "warfarin": 308.3,
    "propranolol": 259.3,
    "antipyrine": 188.2,
    "acetaminophen": 151.2,
    "doxorubicin": 543.5,
}


def check_compound_mw(compound: str, molecular_weight_g_mol: float) -> None:
    """Raise ValueError when a *known* compound's reported MW disagrees.

    ``compound`` is matched case-insensitively; the tolerance is 1 % relative,
    which forgives innocuous rounding (308 vs 308.3) while catching gross
    fabrication (464 for warfarin is a 50 % mismatch). Unknown compounds pass
    through unchecked — we have no pinned value to verify them against.
    """
    known = COMPOUND_MW.get(compound.lower())
    if known is None:
        return
    if not math.isfinite(float(molecular_weight_g_mol)):
        raise ValueError(
            f"molecular weight {molecular_weight_g_mol!r} is not finite for {compound!r}"
        )
    if abs(float(molecular_weight_g_mol) - known) / known > 0.01:
        raise ValueError(
            f"molecular weight {molecular_weight_g_mol} g/mol is inconsistent with "
            f"compound {compound!r} ({known} g/mol)"
        )


def extraction_ratio(inlet_concentration_uM: float, outlet_concentration_uM: float) -> float:
    """Fraction of drug removed from the perfusate in a single pass.

    .. math:: E = 1 - \\frac{C_\\text{out}}{C_\\text{in}}

    E = 0 means no clearance; E = 1 complete clearance; a negative E means the
    outlet *exceeds* the inlet (active secretion, or an extraction error) — the
    value is returned as-is so the sanity band can flag it rather than the
    calculator silently clamping physiology.

    Parameters
    ----------
    inlet_concentration_uM : float
        Drug concentration in the medium entering the chip (µM).
    outlet_concentration_uM : float
        Drug concentration in the medium leaving the chip (µM).

    Returns
    -------
    float
        Extraction ratio (dimensionless).
    """
    _validate_nonneg(
        inlet_concentration_uM=inlet_concentration_uM,
        outlet_concentration_uM=outlet_concentration_uM,
    )
    if inlet_concentration_uM == 0:
        # E = 1 − C_out/C_in is undefined when nothing enters the chip: 0/0 for a
        # clean perfusate, or a sign-flipped blow-up for an outlet > 0. A zero
        # inlet is a bad measurement, not a physiological regime — reject it so
        # the benchmark scores the answer unverifiable instead of crashing.
        raise ValueError("extraction_ratio: inlet concentration must be > 0")
    return 1.0 - outlet_concentration_uM / inlet_concentration_uM


def clearance_uLmin(
    inlet_concentration_uM: float, outlet_concentration_uM: float, flow_rate_uLmin: float
) -> float:
    """Volume of medium fully cleared of drug per minute.

    .. math:: Cl = \\frac{C_\\text{in} - C_\\text{out}}{C_\\text{in}} \\times Q
                = E \\times Q

    This is the classic "volume cleared" formulation of clearance, in the same
    volume units as the flow rate.

    Parameters
    ----------
    inlet_concentration_uM : float
        Inlet concentration (µM).
    outlet_concentration_uM : float
        Outlet concentration (µM).
    flow_rate_uLmin : float
        Perfusion flow rate (µL/min).

    Returns
    -------
    float
        Clearance in µL/min.
    """
    _validate_nonneg(flow_rate_uLmin=flow_rate_uLmin)
    # Delegate the ratio to extraction_ratio so its zero-inlet guard applies
    # here too (E is undefined at a zero inlet; the division must not blow up).
    e = extraction_ratio(inlet_concentration_uM, outlet_concentration_uM)
    return e * flow_rate_uLmin


def half_life_h(system_volume_uL: float, clearance_uLmin: float) -> float:
    """Elimination half-life of the drug in a recirculating system.

    .. math:: t_{1/2} = \\frac{\\ln 2 \\cdot V}{Cl}

    Assumes a well-mixed single compartment of volume ``V`` (recirculating
    reservoir + chip + tubing) cleared at ``Cl``. The minutes from µL/min
    cancel: ``ln2·V/Cl`` is in minutes, so the result is converted to hours.

    Parameters
    ----------
    system_volume_uL : float
        Recirculating medium volume (µL).
    clearance_uLmin : float
        Clearance (µL/min).

    Returns
    -------
    float
        Half-life in hours.
    """
    _validate_positive(system_volume_uL=system_volume_uL, clearance_uLmin=clearance_uLmin)
    return _LN2 * system_volume_uL / clearance_uLmin / 60.0


def accumulation_ratio(half_life_h: float, dose_interval_h: float) -> float:
    """Steady-state accumulation factor for fixed-interval repeated dosing.

    .. math:: R = \\frac{1}{1 - e^{-k\\tau}}, \\qquad k = \\frac{\\ln 2}{t_{1/2}}

    R ≥ 1; R = 1 when the interval is much longer than the half-life (no
    accumulation), and grows as the interval approaches or drops below the
    half-life.

    Parameters
    ----------
    half_life_h : float
        Elimination half-life (h).
    dose_interval_h : float
        Time between doses (h).

    Returns
    -------
    float
        Accumulation ratio (dimensionless, ≥ 1).
    """
    _validate_positive(half_life_h=half_life_h, dose_interval_h=dose_interval_h)
    k = _LN2 / half_life_h
    return 1.0 / (1.0 - math.exp(-k * dose_interval_h))


def mass_cleared_ug_h(
    clearance_uLmin: float, inlet_concentration_uM: float, molecular_weight_g_mol: float
) -> float:
    """Amount of drug the chip removes per hour, in absolute mass.

    .. math:: M = Cl \\cdot C_\\text{in} \\cdot MW \\cdot 6 \\times 10^{-5}

    Derivation of the factor ``6e-5`` (the easy 1000× trap this function owns):

    - ``Cl·C_in`` = (µL/min)·(µM). ``C_in [µM] = C_in·1e-6 mol/L`` and
      ``1 µL/min = 1e-6 L/min``, so ``Cl·C_in = Cl·C_in·1e-12 mol/min``.
    - ``× MW [g/mol]`` gives grams/minute: ``Cl·C_in·MW·1e-12 g/min``.
    - ``× 1e6 µg/g × 60 min/h`` gives ``Cl·C_in·MW·1e-12·1e6·60 = Cl·C_in·MW·6e-5``
      µg/h.

    Parameters
    ----------
    clearance_uLmin : float
        Clearance (µL/min).
    inlet_concentration_uM : float
        Inlet concentration (µM).
    molecular_weight_g_mol : float
        Drug molecular weight (g/mol).

    Returns
    -------
    float
        Mass cleared in µg/h.
    """
    _validate_positive(
        clearance_uLmin=clearance_uLmin,
        inlet_concentration_uM=inlet_concentration_uM,
        molecular_weight_g_mol=molecular_weight_g_mol,
    )
    return clearance_uLmin * inlet_concentration_uM * molecular_weight_g_mol * 6e-5


def _validate_nonneg(**values: float) -> None:
    """Raise ValueError on non-finite or negative inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {val!r}")


def _validate_positive(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "COMPOUND_MW",
    "check_compound_mw",
    "extraction_ratio",
    "clearance_uLmin",
    "half_life_h",
    "accumulation_ratio",
    "mass_cleared_ug_h",
]
