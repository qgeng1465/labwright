"""Tests for the unit & dimension layer.

The layer's whole job is turning "wrong number" into "right magnitude, wrong
unit". The alias table must be dimensionally correct against pint, and the
misread classifier must catch the real-world dyn/cm²-vs-Pa confusion without
flagging ordinary arithmetic errors.
"""

import pytest

from labwright.verify.units import (
    CANONICAL_UNITS,
    UNIT_ALIASES,
    canonical_unit,
    classify_unit_misread,
    convert,
)
from labwright.calc.units import Q


def test_dyn_vs_pa_is_tenfold():
    # 1 Pa = 10 dyn/cm² — the single most important unit pair in OOC.
    assert Q(1, "Pa").to("dyn/cm**2").magnitude == pytest.approx(10.0)


def test_convert_roundtrip():
    assert convert(0.05, "Pa", "dyn/cm**2") == pytest.approx(0.5)
    assert convert(0.5, "dyn/cm**2", "Pa") == pytest.approx(0.05)
    assert convert(1.0, "mL/min", "uL/min") == pytest.approx(1000.0)


def test_alias_table_matches_pint():
    # Every alias factor must agree with pint: factor = value_in_from/value_in_to.
    for name, from_u, to_u, factor in UNIT_ALIASES:
        try:
            measured = Q(1.0, to_u).to(from_u).magnitude
        except Exception:  # noqa: BLE001 - unsupported unit pair
            continue
        assert measured == pytest.approx(factor), name


def test_classify_kidney_unit_misread():
    # The real bug: PTEC target 0.2 dyn/cm² = 0.02 Pa, model reports 0.2 "Pa".
    m = classify_unit_misread(0.2, 0.02, "derived.shear_pa")
    assert m is not None
    assert m["alias"] == "dyn/cm^2 read as Pa"
    assert m["expected_in_from"] == pytest.approx(0.2)


def test_no_false_positive_on_arithmetic_error():
    # 0.03 vs 0.02 is a 1.5× error — not a known unit pair.
    assert classify_unit_misread(0.03, 0.02, "derived.shear_pa") is None
    # 3× error is not a unit pair either.
    assert classify_unit_misread(0.06, 0.02, "derived.shear_pa") is None


def test_ignores_wrong_field_canonical_unit():
    # 10× on a field whose canonical unit is not Pa (e.g. µL/min) is not a
    # dyn/cm² misread.
    assert classify_unit_misread(100, 10, "flow_rate_uLmin") is None


def test_canonical_units_declared_for_all_derived_fields():
    for field in (
        "derived.shear_pa", "derived.reynolds", "derived.pressure_drop_pa",
        "derived.residence_time_s", "derived.channel_volume_ul",
        "derived.mean_velocity_mms", "cells.seed_count",
        "culture.seed_per_well", "culture.total_seed_count",
        "culture.medium_volume_per_well_ml", "culture.total_medium_ml",
        "culture.expected_confluence_pct", "dosing.dmso_fraction_vv",
        "stats.n_per_group",
    ):
        assert canonical_unit(field), field
