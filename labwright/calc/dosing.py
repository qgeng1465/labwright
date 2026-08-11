"""Dosing calculators — molarity, dilution, drug preparation.

Everything a bench scientist computes before pipetting a compound: stock
preparation from powder, working dilutions, DMSO carry-over checks, and
dose conversions. All functions are pure and unit-strict.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Stock preparation
# ---------------------------------------------------------------------------


def molarity_from_mass(mass_mg: float, molecular_weight_g_mol: float, volume_ml: float) -> float:
    """Molar concentration from dissolving a weighed mass.

    .. math:: C = \\frac{m}{MW \\cdot V}

    Parameters
    ----------
    mass_mg : float
        Mass of compound in mg.
    molecular_weight_g_mol : float
        Molecular weight in g/mol.
    volume_ml : float
        Final volume in mL.

    Returns
    -------
    float
        Concentration in mM.
    """
    _validate_positive(mass_mg=mass_mg, molecular_weight_g_mol=molecular_weight_g_mol, volume_ml=volume_ml)
    return mass_mg / molecular_weight_g_mol / volume_ml  # mM


def mass_for_molarity(concentration_mM: float, molecular_weight_g_mol: float, volume_ml: float) -> float:
    """Mass of powder needed to make a stock of a given concentration.

    .. math:: m = C \\cdot MW \\cdot V

    Parameters
    ----------
    concentration_mM : float
        Desired concentration in mM.
    molecular_weight_g_mol : float
        Molecular weight in g/mol.
    volume_ml : float
        Final volume in mL.

    Returns
    -------
    float
        Mass in mg.
    """
    _validate_positive(concentration_mM=concentration_mM, molecular_weight_g_mol=molecular_weight_g_mol, volume_ml=volume_ml)
    return concentration_mM * molecular_weight_g_mol * volume_ml  # mg


# ---------------------------------------------------------------------------
# Dilution
# ---------------------------------------------------------------------------


def dilution_volume(stock_mM: float, target_mM: float, target_volume_ml: float) -> float:
    """Volume of stock to add for a target working concentration.

    .. math:: C_1 V_1 = C_2 V_2 \\Rightarrow V_1 = C_2 V_2 / C_1

    Parameters
    ----------
    stock_mM : float
        Stock concentration in mM (must be > target).
    target_mM : float
        Desired working concentration in mM.
    target_volume_ml : float
        Final working volume in mL.

    Returns
    -------
    float
        Stock volume to add, in mL.
    """
    _validate_positive(stock_mM=stock_mM, target_mM=target_mM, target_volume_ml=target_volume_ml)
    if target_mM >= stock_mM:
        raise ValueError(f"target_mM ({target_mM}) must be < stock_mM ({stock_mM}) for a dilution")
    return target_mM * target_volume_ml / stock_mM  # mL


def final_concentration_after_dilution(stock_mM: float, stock_volume_ml: float, final_volume_ml: float) -> float:
    """Concentration after adding stock to a known final volume.

    .. math:: C_2 = C_1 V_1 / V_2

    Parameters
    ----------
    stock_mM : float
        Stock concentration in mM.
    stock_volume_ml : float
        Volume of stock added in mL.
    final_volume_ml : float
        Final volume in mL (must be >= stock volume).

    Returns
    -------
    float
        Resulting concentration in mM.
    """
    _validate_positive(
        stock_mM=stock_mM, stock_volume_ml=stock_volume_ml, final_volume_ml=final_volume_ml
    )
    if final_volume_ml < stock_volume_ml:
        raise ValueError("final_volume_ml must be >= stock_volume_ml")
    return stock_mM * stock_volume_ml / final_volume_ml  # mM


def serial_dilution(stock_mM: float, dilution_factor: float, steps: int) -> list[float]:
    """Concentrations along a serial dilution series.

    .. math:: C_i = C_0 / f^i

    Parameters
    ----------
    stock_mM : float
        Top concentration in mM.
    dilution_factor : float
        Per-step dilution factor (e.g. 3 for 1:3, 10 for 1:10).
    steps : int
        Number of dilution steps (>= 1).

    Returns
    -------
    list[float]
        Concentration at each step, from the top concentration onward.
    """
    _validate_positive(stock_mM=stock_mM, dilution_factor=dilution_factor)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    return [stock_mM / (dilution_factor**i) for i in range(steps + 1)]


# ---------------------------------------------------------------------------
# Solvent / dose conversions
# ---------------------------------------------------------------------------


def dmso_fraction(stock_dmso_mM: float, working_mM: float) -> float:
    """Volume fraction of DMSO in the final medium at a working dose.

    .. math:: f = C_w / C_s

    Used to flag solvent toxicity: DMSO above ~0.1-0.5 % v/v harms cells.

    Parameters
    ----------
    stock_dmso_mM : float
        Compound concentration in pure DMSO stock, mM.
    working_mM : float
        Desired working concentration in medium, mM.

    Returns
    -------
    float
        DMSO volume fraction (dimensionless; 0.001 = 0.1 % v/v).
    """
    _validate_positive(stock_dmso_mM=stock_dmso_mM, working_mM=working_mM)
    if working_mM > stock_dmso_mM:
        raise ValueError(f"working_mM ({working_mM}) cannot exceed stock_dmso_mM ({stock_dmso_mM})")
    return working_mM / stock_dmso_mM


def molar_to_ng_per_ml(concentration_mM: float, molecular_weight_g_mol: float) -> float:
    """Convert a molar dose to a mass concentration.

    .. math:: \\text{ng/mL} = C \\,(\\text{mM}) \\times MW \\,(\\text{g/mol})

    Parameters
    ----------
    concentration_mM : float
        Concentration in mM.
    molecular_weight_g_mol : float
        Molecular weight in g/mol.

    Returns
    -------
    float
        Mass concentration in ng/mL.
    """
    _validate_positive(concentration_mM=concentration_mM, molecular_weight_g_mol=molecular_weight_g_mol)
    return concentration_mM * molecular_weight_g_mol * 1e3  # ng/mL


def _validate_positive(**values: float) -> None:
    """Raise ValueError on non-finite or non-positive inputs."""
    for name, val in values.items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")


__all__ = [
    "molarity_from_mass",
    "mass_for_molarity",
    "dilution_volume",
    "final_concentration_after_dilution",
    "serial_dilution",
    "dmso_fraction",
    "molar_to_ng_per_ml",
]
