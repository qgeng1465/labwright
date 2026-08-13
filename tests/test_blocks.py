"""Tests for the design-domain registry (:mod:`labwright.blocks`).

The Block spec promises that every design domain is declared exactly once and
that the design gate, the sanity bands, the canonical units and the benchmark
field map all derive from that one declaration. These tests pin the contract:

* every block is internally consistent (a key is not both raw and derived,
  every derived key has a field-map entry, a sanity band and a canonical unit),
* the consumers still agree with the registry (so the refactor changed nothing),
* the registry is wired into the schema (every ``plan_field`` exists on
  ``DesignPlan``, every ``input_field`` on ``DesignInput``).
"""

import pytest

from labwright import blocks
from labwright.blocks import ALL_DERIVED_KEYS, ALL_FIELD_MAP, Band, BLOCKS
from labwright.design import DesignInput, _DERIVED_FIELD_NAMES
from labwright.schema.design import DesignPlan
from labwright.verify.sanity import SANITY_BANDS
from labwright.verify.units import CANONICAL_UNITS

EXPECTED_BLOCKS = {
    "flow": "derived",
    "cells": "cells",
    "culture": "culture",
    "spheroid": "spheroid",
    "dosing": "dosing",
    "stats": "stats",
}


def test_expected_blocks_exist_with_plan_fields():
    assert set(BLOCKS) == set(EXPECTED_BLOCKS)
    for name, plan_field in EXPECTED_BLOCKS.items():
        assert BLOCKS[name].plan_field == plan_field
        # pydantic v2 fields live in model_fields, not as class attributes
        assert plan_field in DesignPlan.model_fields, (
            f"{name}.plan_field {plan_field!r} is not a DesignPlan field"
        )


def test_input_fields_exist_on_design_input():
    for name, b in BLOCKS.items():
        if b.input_field is not None:
            assert b.input_field in DesignInput.model_fields, (
                f"{name}.input_field {b.input_field!r} is not a DesignInput field"
            )


def test_no_key_is_both_raw_and_derived():
    for name, b in BLOCKS.items():
        dupes = set(b.raw_keys) & set(b.derived_keys)
        assert not dupes, f"{name}: {sorted(dupes)} both raw and derived"


def test_every_derived_key_has_field_map_band_and_unit():
    # The registry's own import-time _validate() already raises on this; this
    # test documents the contract explicitly rather than relying on the side
    # effect of import.
    for name, b in BLOCKS.items():
        for dk in b.derived_keys:
            field = b.field_map.get(dk)
            assert field is not None, f"{name}.{dk}: no field_map entry"
            assert field in b.sanity_bands, f"{name}.{dk} -> {field}: no sanity band"
            assert field in b.canonical_units, f"{name}.{dk} -> {field}: no canonical unit"


def test_field_map_values_resolve_and_are_prefix_consistent():
    for name, b in BLOCKS.items():
        for key, field in b.field_map.items():
            assert field in b.canonical_units, f"{name}: {key} -> {field} has no unit"
            if field != key:
                assert field.startswith(b.plan_field + "."), (
                    f"{name}: {key} -> {field} does not start with {b.plan_field!r}"
                )


def test_hard_band_contains_soft_band():
    for name, b in BLOCKS.items():
        for field, band in b.sanity_bands.items():
            assert band.hard_min is None or band.soft_min is None or band.hard_min <= band.soft_min
            assert band.hard_max is None or band.soft_max is None or band.hard_max >= band.soft_max
            assert band.units  # non-empty


def test_consumers_agree_with_registry():
    # The refactor moved the tables into the blocks; these must be the same
    # objects/values the consumers actually use.
    assert SANITY_BANDS == blocks.ALL_SANITY_BANDS
    assert CANONICAL_UNITS == blocks.ALL_CANONICAL_UNITS
    assert set(_DERIVED_FIELD_NAMES) == set(ALL_DERIVED_KEYS)


def test_benchmark_field_map_is_registry_union():
    import eval.benchmark as bench

    assert bench._FIELD_MAP == ALL_FIELD_MAP


def test_shared_total_medium_ml_maps_to_culture_first():
    # total_medium_ml is derived in both the culture and spheroid blocks; the
    # merged field map keeps the first declaration (culture), matching the
    # pre-registry precedence. Both map to a "mL" field, so it is behaviourally
    # neutral either way — this pins that we did not silently flip it.
    assert BLOCKS["culture"].field_map["total_medium_ml"] == "culture.total_medium_ml"
    assert BLOCKS["spheroid"].field_map["total_medium_ml"] == "spheroid.total_medium_ml"
    assert ALL_FIELD_MAP["total_medium_ml"] == "culture.total_medium_ml"


def test_culture_and_spheroid_shared_units_are_identical():
    # The two mappings of the shared key point at fields with the same canonical
    # unit, so classify_unit_misread cannot distinguish them.
    assert (
        BLOCKS["culture"].canonical_units["culture.total_medium_ml"]
        == BLOCKS["spheroid"].canonical_units["spheroid.total_medium_ml"]
    )


def test_bands_are_shared_band_class():
    # sanity.py must re-export the registry's Band, not define its own — two
    # dataclass types would silently break value equality everywhere.
    assert type(SANITY_BANDS["derived.shear_pa"]) is Band
