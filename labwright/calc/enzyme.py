"""Enzyme/receptor competitive-binding arithmetic — the OA + UDPGA class of question.

A design that co-incubates a target protein with a substrate and a competing
small molecule (the reviewer's example: oleic acid, OA, as a possible inhibitor
of a UGT conjugation reaction whose cofactor/substrate is UDP-glucuronic acid,
UDPGA) needs three numbers, all of which this module computes deterministically:

1. **Fractional activity under competitive inhibition** — the fraction of the
   uninhibited rate that remains when a competitive inhibitor at ``[I]`` with
   constant ``Ki`` competes with substrate at ``[S]`` with ``Km``.
2. **Potency interconversion** — IC50 ↔ Ki via the Cheng–Prusoff relation for
   a competitive inhibitor (the factor ``(1 + [S]/Km)`` that separates a *run
   condition* IC50 from the *intrinsic* Ki).
3. **Molar amounts and ratios** — the exact mole numbers of protein, substrate
   and inhibitor in a reaction mix, and their ratios, from the stock
   concentrations and pipetting volumes.

The kinetics are textbook (Michaelis–Menten with a competitive inhibitor; the
Cheng–Prusoff correction); every "constant" is either an input or a documented
condition. No binding constant is invented here — a gold entry states the Ki and
Km, and this module turns them into the derived activity/IC50/ratio numbers.

References
----------
- Competitive inhibition, v_i/v_0 = [S]/(K_m(1 + [I]/K_i) + [S]): standard
  enzyme kinetics (Michaelis–Menten with a competitive inhibitor).
- IC50 = Ki·(1 + [S]/Km): Cheng & Prusoff, Biochem. Pharmacol. 22:3099–3108
  (1973), doi:10.1016/0006-2952(73)90196-2.
"""

from __future__ import annotations


def fractional_activity(km: float, s_conc: float, ki: float, i_conc: float) -> float:
    """Fraction of the uninhibited rate remaining under competitive inhibition.

    .. math:: \\frac{v_i}{v_0} = \\frac{[S]}{K_m\\left(1 + \\frac{[I]}{K_i}\\right) + [S]}

    All concentrations and constants in the same molar unit (typically µM).
    Returns a value in (0, 1].
    """
    if km <= 0:
        raise ValueError(f"km must be > 0, got {km!r}")
    if ki <= 0:
        raise ValueError(f"ki must be > 0, got {ki!r}")
    if s_conc < 0:
        raise ValueError(f"s_conc must be >= 0, got {s_conc!r}")
    if i_conc < 0:
        raise ValueError(f"i_conc must be >= 0, got {i_conc!r}")
    denom = km * (1.0 + i_conc / ki) + s_conc
    if denom == 0:
        raise ValueError("km(1 + [I]/ki) + [S] is 0 — all concentrations are 0")
    return s_conc / denom


def percent_inhibition(activity: float) -> float:
    """Percent inhibition from a fractional activity.

    .. math:: \\%\\,I = (1 - v_i/v_0)\\,100
    """
    if not 0.0 <= activity <= 1.0:
        raise ValueError(f"activity must be in [0, 1], got {activity!r}")
    return (1.0 - activity) * 100.0


def ic50_from_ki(km: float, s_conc: float, ki: float) -> float:
    """Cheng–Prusoff IC50 for a *competitive* inhibitor at a given substrate level.

    .. math:: IC_{50} = K_i\\left(1 + \\frac{[S]}{K_m}\\right)
    """
    if km <= 0 or ki <= 0:
        raise ValueError("km and ki must be > 0")
    if s_conc < 0:
        raise ValueError(f"s_conc must be >= 0, got {s_conc!r}")
    return ki * (1.0 + s_conc / km)


def apparent_km_um(km: float, i_conc: float, ki: float) -> float:
    """Apparent Km under competitive inhibition.

    .. math:: K_m^{app} = K_m\\left(1 + \\frac{[I]}{K_i}\\right)

    The substrate half-saturation shifts to higher substrate with a competitive
    inhibitor present; the *catalytic* Km (in the absence of inhibitor) is
    unchanged. This is the number a Lineweaver-Burk or Hanes plot reports when
    the inhibitor is in the mix.
    """
    if km <= 0:
        raise ValueError(f"km must be > 0, got {km!r}")
    if ki <= 0:
        raise ValueError(f"ki must be > 0, got {ki!r}")
    if i_conc < 0:
        raise ValueError(f"i_conc must be >= 0, got {i_conc!r}")
    return km * (1.0 + i_conc / ki)


def velocity_umol_min(vmax_umol_min: float, km: float, s_conc: float,
                      ki: float, i_conc: float) -> float:
    """Reaction velocity under competitive inhibition, µmol/min.

    .. math:: v = V_{max}\\,\\frac{[S]}{K_m\\left(1 + \\frac{[I]}{K_i}\\right) + [S]}

    The inhibited Michaelis-Menten rate — the amount of product an enzyme
    actually turns over per minute at the stated substrate, inhibitor and
    constants. Requires the user to supply ``vmax_umol_min`` (a run condition,
    not a claim of this module).
    """
    if vmax_umol_min <= 0:
        raise ValueError(f"vmax_umol_min must be > 0, got {vmax_umol_min!r}")
    return vmax_umol_min * fractional_activity(km, s_conc, ki, i_conc)


def ki_from_ic50(km: float, s_conc: float, ic50: float) -> float:
    """Intrinsic Ki recovered from a run-condition IC50 (Cheng–Prusoff, competitive).

    .. math:: K_i = \\frac{IC_{50}}{1 + [S]/K_m}
    """
    if km <= 0:
        raise ValueError(f"km must be > 0, got {km!r}")
    if s_conc < 0:
        raise ValueError(f"s_conc must be >= 0, got {s_conc!r}")
    if ic50 <= 0:
        raise ValueError(f"ic50 must be > 0, got {ic50!r}")
    return ic50 / (1.0 + s_conc / km)


def molar_amount_nmol(conc_um: float, volume_ul: float) -> float:
    """Moles of a component in a pipetted volume, nmol.

    .. math:: n = C\\,V

    C in µM (= nmol/mL = pmol/µL) and V in µL gives pmol; ÷1000 → nmol.
    """
    if conc_um < 0:
        raise ValueError(f"conc_um must be >= 0, got {conc_um!r}")
    if volume_ul < 0:
        raise ValueError(f"volume_ul must be >= 0, got {volume_ul!r}")
    return conc_um * volume_ul / 1000.0


def molar_ratio(amount_a: float, amount_b: float) -> float:
    """Stoichiometric ratio A : B (amount_a / amount_b).

    A reaction-mix ratio stated as a single number; ``amount_b`` must be > 0.
    """
    if amount_a < 0:
        raise ValueError(f"amount_a must be >= 0, got {amount_a!r}")
    if amount_b <= 0:
        raise ValueError(f"amount_b must be > 0, got {amount_b!r}")
    return amount_a / amount_b


def dilute_volume_ul(stock_um: float, target_um: float, final_volume_ul: float) -> float:
    """Stock volume to pipette for a target concentration (C1V1 = C2V2), µL.

    .. math:: V_1 = \\frac{C_2\\,V_2}{C_1}
    """
    if stock_um <= 0:
        raise ValueError(f"stock_um must be > 0, got {stock_um!r}")
    if target_um < 0:
        raise ValueError(f"target_um must be >= 0, got {target_um!r}")
    if target_um > stock_um:
        raise ValueError(f"target ({target_um}) cannot exceed stock ({stock_um}) by dilution")
    if final_volume_ul <= 0:
        raise ValueError(f"final_volume_ul must be > 0, got {final_volume_ul!r}")
    return target_um * final_volume_ul / stock_um


def mix_volume_ratio(volume_a_ul: float, volume_b_ul: float) -> float:
    """Volume ratio A : B of a two-component mix (e.g. OA stock : UDPGA cofactor)."""
    if volume_a_ul < 0 or volume_b_ul < 0:
        raise ValueError("volumes must be >= 0")
    if volume_a_ul == 0 and volume_b_ul == 0:
        raise ValueError("at least one volume must be > 0")
    return volume_a_ul / volume_b_ul if volume_b_ul > 0 else float("inf")


__all__ = [
    "fractional_activity",
    "percent_inhibition",
    "ic50_from_ki",
    "apparent_km_um",
    "velocity_umol_min",
    "ki_from_ic50",
    "molar_amount_nmol",
    "molar_ratio",
    "dilute_volume_ul",
    "mix_volume_ratio",
]
