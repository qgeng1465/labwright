"""Co-culture seeding stoichiometry — ratios, counts and media blends.

Liver-lobule and other tissue models co-seed two (or more) populations —
e.g. endothelial (HUVEC-T1) lining the sinusoidal channels and hepatocytes
(HepG2 / PHH) forming the parenchyma. The reviewer-facing questions this module
answers are the *arithmetic* of such a mixed seeding:

1. **Seed-count split** — given a target total (cells) and the fraction assigned
   to one population, how many of each population go in (or per well / per cm²).
2. **Per-well/per-area allocation** — a total seeding density (cells/cm²) and an
   area → cells of each type, from a chosen A-fraction.
3. **Media blend** — how to mix the two lines' maintenance media at a ratio
   (e.g. 1:1) and what the resulting composition is.
4. **Viability & suspension arithmetic** — viable cells after a thaw/passage,
   suspension density (cells/mL), and the volume of a suspension needed for a
   cell budget.

The math is pure arithmetic; the *choice* of total density and A-fraction is the
designer's input (or a convention stated in the goal, e.g. hepatocyte-dominant
lobule). Where the module needs a default, it documents the convention rather
than presenting it as a measured value. Cell-type reference data lives in
:mod:`labwright.physiology`, not here.

References
----------
- N_A = f·N_total, N_B = N_total − N_A: fraction-to-count arithmetic (standard).
- N_viable = N_total × viability: standard cryo-recovery accounting.
- C = N/V and V = N/C: suspension concentration arithmetic (standard).
"""

from __future__ import annotations


def cells_from_fraction(total_count: float, fraction_a: float) -> tuple[float, float]:
    """Split a total cell budget into populations A and B.

    .. math:: N_A = f\\, N,\\quad N_B = N - N_A

    ``fraction_a`` is the fraction of the total assigned to population A
    (0 ≤ f ≤ 1). Returns ``(N_A, N_B)``.
    """
    if total_count <= 0:
        raise ValueError(f"total_count must be > 0, got {total_count!r}")
    if not 0.0 <= fraction_a <= 1.0:
        raise ValueError(f"fraction_a must be in [0, 1], got {fraction_a!r}")
    count_a = total_count * fraction_a
    return count_a, total_count - count_a


def fraction_from_counts(count_a: float, total_count: float) -> float:
    """Fraction of population A in a mixed suspension.

    .. math:: f = N_A / N
    """
    if total_count <= 0:
        raise ValueError(f"total_count must be > 0, got {total_count!r}")
    if count_a < 0 or count_a > total_count:
        raise ValueError(f"count_a ({count_a}) must lie within [0, {total_count}]")
    return count_a / total_count


def cells_per_well(total_density_cells_cm2: float, area_cm2: float,
                   fraction_a: float) -> tuple[float, float]:
    """Per-well seeding counts of A and B from a total seeding density.

    .. math:: N_{well} = \\rho\\, A;\\quad N_A = f\\,N_{well},\\ N_B = N_{well} - N_A
    """
    if total_density_cells_cm2 <= 0:
        raise ValueError("total_density_cells_cm2 must be > 0")
    if area_cm2 <= 0:
        raise ValueError(f"area_cm2 must be > 0, got {area_cm2!r}")
    if not 0.0 <= fraction_a <= 1.0:
        raise ValueError(f"fraction_a must be in [0, 1], got {fraction_a!r}")
    per_well = total_density_cells_cm2 * area_cm2
    return cells_from_fraction(per_well, fraction_a)


def total_for_two_wells(count_a_per_well: float, count_b_per_well: float,
                        wells: int) -> float:
    """Total cells across a number of wells from per-well counts of A and B."""
    if count_a_per_well < 0 or count_b_per_well < 0:
        raise ValueError("per-well counts must be >= 0")
    if wells < 1:
        raise ValueError(f"wells must be >= 1, got {wells!r}")
    return (count_a_per_well + count_b_per_well) * wells


def total_cells(count_per_well: float, wells: int) -> float:
    """Total cells of one population across a number of wells.

    .. math:: N = N_{well} \\cdot n_{wells}
    """
    if count_per_well < 0:
        raise ValueError(f"count_per_well must be >= 0, got {count_per_well!r}")
    if wells < 1:
        raise ValueError(f"wells must be >= 1, got {wells!r}")
    return count_per_well * wells


def seeding_ratio(count_a: float, count_b: float) -> float:
    """A : B seeding ratio as a single number.

    .. math:: r = N_A / N_B

    The number a design reports as "HUVEC-T1 : HepG2 = r"; ``count_b`` must be
    > 0.
    """
    if count_a < 0:
        raise ValueError(f"count_a must be >= 0, got {count_a!r}")
    if count_b <= 0:
        raise ValueError(f"count_b must be > 0, got {count_b!r}")
    return count_a / count_b


def viable_count(total_count: float, viability_pct: float) -> float:
    """Viable cells after thaw/passage.

    .. math:: N_{viable} = N \\cdot \\frac{v}{100}
    """
    if total_count < 0:
        raise ValueError(f"total_count must be >= 0, got {total_count!r}")
    if not 0.0 <= viability_pct <= 100.0:
        raise ValueError(f"viability_pct must be in [0, 100], got {viability_pct!r}")
    return total_count * viability_pct / 100.0


def media_blend(volume_medium_a_ml: float, volume_medium_b_ml: float) -> tuple[float, float, float]:
    """Blend two media; returns ``(total_ml, fraction_a, fraction_b)``.

    .. math:: V = V_A + V_B,\\quad f_A = V_A / V
    """
    if volume_medium_a_ml < 0 or volume_medium_b_ml < 0:
        raise ValueError("media volumes must be >= 0")
    if volume_medium_a_ml == 0 and volume_medium_b_ml == 0:
        raise ValueError("at least one medium volume must be > 0")
    total = volume_medium_a_ml + volume_medium_b_ml
    return total, volume_medium_a_ml / total, volume_medium_b_ml / total


def suspension_density_cells_ml(cell_count: float, volume_ml: float) -> float:
    """Cell suspension concentration.

    .. math:: C = N / V
    """
    if cell_count < 0:
        raise ValueError(f"cell_count must be >= 0, got {cell_count!r}")
    if volume_ml <= 0:
        raise ValueError(f"volume_ml must be > 0, got {volume_ml!r}")
    return cell_count / volume_ml


def suspension_volume_ml_for_cells(cell_count: float, density_cells_ml: float) -> float:
    """Suspension volume that contains a cell budget.

    .. math:: V = N / C
    """
    if cell_count < 0:
        raise ValueError(f"cell_count must be >= 0, got {cell_count!r}")
    if density_cells_ml <= 0:
        raise ValueError(f"density_cells_ml must be > 0, got {density_cells_ml!r}")
    return cell_count / density_cells_ml


__all__ = [
    "cells_from_fraction",
    "fraction_from_counts",
    "cells_per_well",
    "total_for_two_wells",
    "total_cells",
    "seeding_ratio",
    "viable_count",
    "media_blend",
    "suspension_density_cells_ml",
    "suspension_volume_ml_for_cells",
]
