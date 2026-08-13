"""One declaration per design domain — the Block spec.

A **block** is one named, optional part of a design: the derived flow metrics,
the plate-culture plan, the 3D-spheroid plan, the channel cell plan, the dosing
plan or the statistics plan. Each block is declared **exactly once here**, and
everything the other layers need to know about it *declaratively* lives in that
one entry:

* ``raw_keys`` — the inputs the LLM may propose (nothing derived).
* ``derived_keys`` — the fields the calculators own; the design gate
  (:mod:`labwright.design`) refuses any of these from the model.
* ``consistency_keys`` — the minimal raw set a bare model must report for its
  numbers to be cross-checkable in the benchmark.
* ``field_map`` — how each report/gold key maps to a verifier field name
  (``"spheroid_volume_ul"`` → ``"spheroid.spheroid_volume_ul"``), which the
  benchmark's unit-misread layer looks up in the canonical-unit table.
* ``sanity_bands`` and ``canonical_units`` — the physiological range bands
  (:mod:`labwright.verify.sanity`) and the audit units
  (:mod:`labwright.verify.units`), keyed by verifier field name.

The consumers import their tables from this registry rather than carrying their
own per-domain constants, so adding a domain is: a ``calc/`` module + a derive
function + a schema model + one ``Block`` entry here. ``_validate()`` runs at
import time and raises if a declared derived field is missing its field-map
entry, sanity band or canonical unit — the "one declaration" property is itself
enforced, not just promised. ``tests/test_blocks.py`` pins the same contract.

This is the *design-domain* registry; :mod:`labwright.physiology` is the
companion *physiology-data* registry (cell-type / organ tables consumed by the
derivers, range checks and safety hints).

Data source for every value: ``labwright/verify/sanity.py`` (bands) and
``labwright/verify/units.py`` (units) before they were centralized here — the
numbers are unchanged, only their home moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import dosing as calc_dosing
from labwright.calc import microfluidics as mf
from labwright.calc import spheroid as calc_spheroid
from labwright.calc import stats as calc_stats


@dataclass(frozen=True)
class Band:
    """A physiological range with a hard physical boundary.

    ``None`` bounds are unbounded on that side. ``soft_min/soft_max`` are the
    warning band; ``hard_min/hard_max`` the error band. Hard bounds are always
    wider than (or equal to) the soft bounds.

    Defined here (not in ``verify.sanity``) so a :class:`Block` can carry its
    bands without importing the verifier; ``verify.sanity`` re-exports it.
    """

    soft_min: float | None
    soft_max: float | None
    hard_min: float | None
    hard_max: float | None
    description: str
    units: str


@dataclass(frozen=True)
class Block:
    """The declarative metadata of one design domain."""

    name: str
    #: DesignPlan attribute holding the block's plan (``"derived"`` for flow).
    plan_field: str
    #: DesignInput attribute holding the raw inputs (``None`` for flow, whose
    #: raws span the separate ``chip`` and ``flow`` inputs).
    input_field: str | None
    #: The calculator module that owns this domain's arithmetic.
    calc: ModuleType
    raw_keys: tuple[str, ...]
    derived_keys: tuple[str, ...]
    consistency_keys: tuple[str, ...]
    #: report/gold key -> verifier field name (benchmark unit-misread layer).
    field_map: dict[str, str]
    #: verifier field name -> physiological range band.
    sanity_bands: dict[str, Band]
    #: verifier field name -> canonical unit (audit table).
    canonical_units: dict[str, str]


# ---------------------------------------------------------------------------
# The blocks, in registration order (flow → cells → culture → spheroid →
# dosing → stats). Shared-key precedence in the merged tables follows this
# order: ``total_medium_ml`` is a derived key of both the culture and spheroid
# blocks (each with its own verifier field, both canonical unit "mL"), and the
# merged field map keeps the *first* mapping, i.e. culture's — matching the
# pre-registry ``_FIELD_MAP`` exactly.
# ---------------------------------------------------------------------------


def _flow() -> Block:
    return Block(
        name="flow",
        plan_field="derived",
        input_field=None,
        calc=mf,
        raw_keys=(
            "width_um", "height_um", "length_mm", "flow_rate_uLmin",
            "viscosity_pas", "density_kgm3",
        ),
        derived_keys=(
            "shear_pa", "reynolds", "pressure_drop_pa", "residence_time_s",
            "channel_volume_ul", "mean_velocity_mms",
        ),
        consistency_keys=(
            "width_um", "height_um", "length_mm", "flow_rate_uLmin",
            "viscosity_pas", "density_kgm3",
        ),
        field_map={
            "shear_pa": "derived.shear_pa",
            "reynolds": "derived.reynolds",
            "pressure_drop_pa": "derived.pressure_drop_pa",
            "residence_time_s": "derived.residence_time_s",
            "channel_volume_ul": "derived.channel_volume_ul",
            "mean_velocity_mms": "derived.mean_velocity_mms",
            "flow_rate_uLmin": "flow_rate_uLmin",
        },
        sanity_bands={
            "derived.shear_pa": Band(0.001, 10, 1e-4, 50,
                "wall shear stress in organ-on-chip culture", "Pa"),
            "derived.reynolds": Band(0.001, 200, 0.0, 2300,
                "Reynolds number (flow must be laminar)", "dimensionless"),
            "derived.pressure_drop_pa": Band(1.0, 1e5, 0.0, 1e6,
                "laminar pressure drop along a microchannel", "Pa"),
            "derived.residence_time_s": Band(0.1, 1e4, 1e-4, 1e5,
                "fluid residence time in a channel", "s"),
            "derived.channel_volume_ul": Band(0.01, 100, 1e-6, 1e4,
                "per-channel culture volume", "uL"),
            "derived.mean_velocity_mms": Band(0.01, 1e4, 1e-4, 1e5,
                "mean flow velocity in a channel", "mm/s"),
        },
        canonical_units={
            "derived.shear_pa": "Pa",
            "derived.reynolds": "dimensionless",
            "derived.pressure_drop_pa": "Pa",
            "derived.residence_time_s": "s",
            "derived.channel_volume_ul": "uL",
            "derived.mean_velocity_mms": "mm/s",
            "flow_rate_uLmin": "uL/min",
            "viscosity_pas": "Pa*s",
            "density_kgm3": "kg/m^3",
            "width_um": "um",
            "height_um": "um",
            "length_mm": "mm",
        },
    )


def _cells() -> Block:
    return Block(
        name="cells",
        plan_field="cells",
        input_field="cells",
        calc=calc_cell,
        raw_keys=(
            "seeding_density_cells_cm2", "culture_area_cm2",
            "doubling_time_h", "culture_duration_h",
        ),
        derived_keys=("seed_count",),
        consistency_keys=(),
        field_map={"seed_count": "cells.seed_count"},
        sanity_bands={
            "cells.seed_count": Band(1e2, 1e9, 1.0, 1e10,
                "cells seeded onto a culture area", "cells"),
            "cells.seeding_density_cells_cm2": Band(1e3, 1e7, 1.0, 1e9,
                "cell seeding density", "cells/cm^2"),
            "cells.culture_area_cm2": Band(1e-4, 100, 1e-6, 1e3,
                "cell culture area", "cm^2"),
        },
        canonical_units={
            "cells.seed_count": "cells",
            "cells.seeding_density_cells_cm2": "cells/cm^2",
            "cells.culture_area_cm2": "cm^2",
        },
    )


def _culture() -> Block:
    return Block(
        name="culture",
        plan_field="culture",
        input_field="culture",
        calc=calc_culture,
        raw_keys=(
            "plate_format", "wells", "seeding_density_cells_cm2",
            "viability_pct", "confluent_density_cells_cm2", "doubling_time_h",
            "culture_duration_h",
        ),
        derived_keys=(
            "seed_per_well", "total_seed_count", "medium_volume_per_well_ml",
            "total_medium_ml", "expected_confluence_pct",
        ),
        consistency_keys=("plate_format", "seeding_density_cells_cm2", "wells"),
        field_map={
            "seed_per_well": "culture.seed_per_well",
            "total_seed_count": "culture.total_seed_count",
            "medium_volume_per_well_ml": "culture.medium_volume_per_well_ml",
            "total_medium_ml": "culture.total_medium_ml",
            "expected_confluence_pct": "culture.expected_confluence_pct",
        },
        sanity_bands={
            "culture.seed_per_well": Band(1e2, 1e9, 1.0, 1e10,
                "cells seeded per well", "cells"),
            "culture.total_seed_count": Band(1e2, 1e10, 1.0, 1e11,
                "total cells seeded across wells", "cells"),
            "culture.seeding_density_cells_cm2": Band(1e3, 1e7, 1.0, 1e9,
                "cell seeding density", "cells/cm^2"),
            "culture.medium_volume_per_well_ml": Band(0.01, 5.0, 1e-4, 100,
                "working medium volume per well", "mL"),
            "culture.total_medium_ml": Band(0.01, 1e3, 1e-4, 1e4,
                "total medium volume across wells", "mL"),
            "culture.expected_confluence_pct": Band(0.0, 100, 0.0, 1000,
                "predicted confluence at harvest (may exceed 100 % for over-confluent cultures)", "%"),
            "culture.doubling_time_h": Band(10, 200, 0.1, 1000,
                "population doubling time", "h"),
            "culture.culture_duration_h": Band(0.0, 2000, 0.0, 1e5,
                "culture duration", "h"),
        },
        canonical_units={
            "culture.seed_per_well": "cells",
            "culture.total_seed_count": "cells",
            "culture.medium_volume_per_well_ml": "mL",
            "culture.total_medium_ml": "mL",
            "culture.seeding_density_cells_cm2": "cells/cm^2",
            "culture.expected_confluence_pct": "%",
            "culture.viability_pct": "%",
            "culture.doubling_time_h": "h",
            "culture.culture_duration_h": "h",
        },
    )


def _spheroid() -> Block:
    return Block(
        name="spheroid",
        plan_field="spheroid",
        input_field="spheroid",
        calc=calc_spheroid,
        raw_keys=(
            "spheroid_format", "spheroid_count", "cells_per_spheroid",
            "cell_diameter_um", "doubling_time_h", "culture_duration_h",
        ),
        derived_keys=(
            "spheroid_volume_ul", "expected_diameter_um", "cells_total",
            "medium_volume_per_spheroid_ul", "total_medium_ml",
            "expected_cells_after_growth",
        ),
        consistency_keys=(
            "spheroid_format", "spheroid_count", "cells_per_spheroid",
            "cell_diameter_um",
        ),
        field_map={
            "spheroid_volume_ul": "spheroid.spheroid_volume_ul",
            "expected_diameter_um": "spheroid.expected_diameter_um",
            "cells_total": "spheroid.cells_total",
            "medium_volume_per_spheroid_ul": "spheroid.medium_volume_per_spheroid_ul",
            "total_medium_ml": "spheroid.total_medium_ml",
            "cells_per_spheroid": "spheroid.cells_per_spheroid",
            "spheroid_count": "spheroid.spheroid_count",
            "expected_cells_after_growth": "spheroid.expected_cells_after_growth",
        },
        sanity_bands={
            "spheroid.cells_per_spheroid": Band(100, 1e5, 1.0, 1e6,
                "cells seeded per spheroid", "cells"),
            "spheroid.expected_diameter_um": Band(30, 2000, 5, 1e4,
                "spheroid diameter (functional spheroids stay < ~400 µm to avoid necrotic cores)", "um"),
            "spheroid.spheroid_volume_ul": Band(1e-4, 1e2, 1e-6, 1e4,
                "spheroid volume (a 200 µm spheroid ≈ 4.2e-3 uL)", "uL"),
            "spheroid.cell_diameter_um": Band(5, 60, 1.0, 200,
                "mean single-cell diameter", "um"),
            "spheroid.medium_volume_per_spheroid_ul": Band(10, 300, 1.0, 2000,
                "working medium volume per spheroid", "uL"),
            "spheroid.total_medium_ml": Band(0.01, 1e3, 1e-4, 1e4,
                "total medium volume for the spheroid culture", "mL"),
            "spheroid.cells_total": Band(1e2, 1e10, 1.0, 1e11,
                "total cells for spheroid seeding", "cells"),
            "spheroid.spheroid_count": Band(1, 1e5, 1.0, 1e6,
                "number of spheroids", "n"),
            "spheroid.doubling_time_h": Band(10, 200, 0.1, 1000,
                "population doubling time", "h"),
            "spheroid.culture_duration_h": Band(0.0, 2000, 0.0, 1e5,
                "culture duration", "h"),
            "spheroid.expected_cells_after_growth": Band(10, 1e8, 1.0, 1e10,
                "predicted cells per spheroid at harvest", "cells"),
        },
        canonical_units={
            "spheroid.cells_per_spheroid": "cells",
            "spheroid.spheroid_count": "n",
            "spheroid.expected_diameter_um": "um",
            "spheroid.spheroid_volume_ul": "uL",
            "spheroid.cell_diameter_um": "um",
            "spheroid.medium_volume_per_spheroid_ul": "uL",
            "spheroid.total_medium_ml": "mL",
            "spheroid.cells_total": "cells",
            "spheroid.expected_cells_after_growth": "cells",
        },
    )


def _dosing() -> Block:
    return Block(
        name="dosing",
        plan_field="dosing",
        input_field="dosing",
        calc=calc_dosing,
        raw_keys=(
            "compound", "molecular_weight_g_mol", "stock_mM", "working_mM",
            "vehicle_control", "exposure_h",
        ),
        derived_keys=("dmso_fraction_vv",),
        consistency_keys=(),
        field_map={"dmso_fraction_vv": "dosing.dmso_fraction_vv"},
        sanity_bands={
            "dosing.stock_mM": Band(0.1, 1e4, 1e-4, 1e6,
                "compound stock concentration", "mM"),
            "dosing.working_mM": Band(1e-3, 100, 1e-6, 1e4,
                "compound working concentration", "mM"),
            "dosing.dmso_fraction_vv": Band(0.0, 0.005, 0.0, 0.14,
                "DMSO volume fraction in medium", "v/v"),
        },
        canonical_units={
            "dosing.stock_mM": "mM",
            "dosing.working_mM": "mM",
            "dosing.dmso_fraction_vv": "v/v",
            "dosing.molecular_weight_g_mol": "g/mol",
        },
    )


def _stats() -> Block:
    return Block(
        name="stats",
        plan_field="stats",
        input_field="stats",
        calc=calc_stats,
        raw_keys=("effect_size", "std_dev", "alpha", "power"),
        derived_keys=("n_per_group",),
        consistency_keys=(),
        field_map={"n_per_group": "stats.n_per_group"},
        sanity_bands={
            "stats.n_per_group": Band(3, 1000, 2, 1e6,
                "biological replicates per group", "n"),
        },
        canonical_units={"stats.n_per_group": "n"},
    )


BLOCKS: dict[str, Block] = {
    b.name: b
    for b in (_flow(), _cells(), _culture(), _spheroid(), _dosing(), _stats())
}


# ---------------------------------------------------------------------------
# Merged tables the consumers import (single source of truth).
# ---------------------------------------------------------------------------


def _merge() -> dict[str, str]:
    """Union of all blocks' field maps, first declaration wins.

    ``total_medium_ml`` is a derived key of both the culture and spheroid
    blocks (each mapped to its own verifier field, both canonical unit "mL");
    culture is declared first, so its mapping wins — matching the pre-registry
    ``_FIELD_MAP`` entry exactly.
    """
    out: dict[str, str] = {}
    for b in BLOCKS.values():
        for key, field_name in b.field_map.items():
            if key not in out:
                out[key] = field_name
    return out


ALL_FIELD_MAP: dict[str, str] = _merge()

ALL_DERIVED_KEYS: frozenset[str] = frozenset(
    key for b in BLOCKS.values() for key in b.derived_keys
)

ALL_RAW_KEYS: frozenset[str] = frozenset(
    key for b in BLOCKS.values() for key in b.raw_keys
)

ALL_SANITY_BANDS: dict[str, Band] = {
    field_name: band
    for b in BLOCKS.values()
    for field_name, band in b.sanity_bands.items()
}

ALL_CANONICAL_UNITS: dict[str, str] = {
    field_name: unit
    for b in BLOCKS.values()
    for field_name, unit in b.canonical_units.items()
}


def _validate() -> None:
    """Fail fast if a block's declaration is incomplete.

    Every declared derived field must have a field-map entry, a sanity band and
    a canonical unit; every field-map value must resolve to a canonical unit;
    no key may be both raw and derived in the same block. This runs at import
    time, so a future domain that forgets a band or unit breaks loudly instead
    of silently scoring.
    """
    for name, b in BLOCKS.items():
        dupes = set(b.raw_keys) & set(b.derived_keys)
        if dupes:
            raise ValueError(f"block {name!r}: keys are both raw and derived: {sorted(dupes)}")
        for dk in b.derived_keys:
            field_name = b.field_map.get(dk)
            if field_name is None:
                raise ValueError(f"block {name!r}: derived key {dk!r} has no field_map entry")
            if field_name not in b.sanity_bands:
                raise ValueError(
                    f"block {name!r}: derived field {field_name!r} has no sanity band"
                )
            if field_name not in b.canonical_units:
                raise ValueError(
                    f"block {name!r}: derived field {field_name!r} has no canonical unit"
                )
        for key, field_name in b.field_map.items():
            if field_name != key and not field_name.startswith(b.plan_field + "."):
                raise ValueError(
                    f"block {name!r}: field_map value {field_name!r} does not match "
                    f"its block prefix {b.plan_field + '.'!r}"
                )
            if field_name not in b.canonical_units:
                raise ValueError(
                    f"block {name!r}: field_map value {field_name!r} has no canonical unit"
                )


_validate()


__all__ = [
    "Band",
    "Block",
    "BLOCKS",
    "ALL_FIELD_MAP",
    "ALL_DERIVED_KEYS",
    "ALL_RAW_KEYS",
    "ALL_SANITY_BANDS",
    "ALL_CANONICAL_UNITS",
]
