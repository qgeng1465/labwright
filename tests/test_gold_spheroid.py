"""Self-consistency checks for the 3D-spheroid gold set.

Every number in ``eval/gold_spheroid.json`` must be reproducible by the
calculators from the raw inputs stated in its goal prose. This test pins that
property: if someone edits a gold ``expected`` value without re-deriving it
through :mod:`labwright.calc.spheroid`, the test fails — the gold set cannot
silently drift away from the code that defines the "correct" answer.
"""

import json
import os

import pytest

from labwright.calc import dosing
from labwright.calc import spheroid
from labwright.design import derive_spheroid

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(_HERE, "eval", "gold_spheroid.json")

#: SpheroidPlan fields a gold ``expected`` key is allowed to target, plus the
#: cross-domain dosing key used by the doxorubicin entry.
_ALLOWED_KEYS = {
    "spheroid_volume_ul", "expected_diameter_um", "cells_total",
    "medium_volume_per_spheroid_ul", "total_medium_ml",
    "expected_cells_after_growth", "cells_per_spheroid", "spheroid_count",
    "dmso_fraction_vv",
}


def _load():
    with open(GOLD_PATH) as fh:
        return json.load(fh)


def test_structure():
    gold = _load()
    assert len(gold) >= 14
    ids = [g["id"] for g in gold]
    assert len(ids) == len(set(ids)), "gold ids must be unique"
    for g in gold:
        assert g["goal"]
        assert g["expected"], f"{g['id']}: expected must be non-empty"
        assert set(g["expected"]) <= _ALLOWED_KEYS, (
            f"{g['id']}: expected keys must map onto SpheroidPlan fields, got "
            f"{set(g['expected']) - _ALLOWED_KEYS}"
        )
        for k, v in g["expected"].items():
            assert isinstance(v, (int, float)) and v > 0, (
                f"{g['id']}.{k}: expected must be a positive number, got {v!r}"
            )
        assert g.get("source", ""), f"{g['id']}: source is mandatory"
        strength = g.get("blind_strength")
        assert strength in (None, "cold", "prompt-backed"), f"{g['id']}: bad blind_strength"


# ---------------------------------------------------------------------------
# Per-entry recomputation: raw inputs stated in each goal prose, derived values
# reproduced with the calculators, compared to the gold file.
# ---------------------------------------------------------------------------


def _sph(spheroid_format, spheroid_count, cells_per_spheroid, cell_diameter_um, **extra):
    return derive_spheroid(
        dict(spheroid_format=spheroid_format, spheroid_count=spheroid_count,
             cells_per_spheroid=cells_per_spheroid, cell_diameter_um=cell_diameter_um, **extra)
    )


RECOMPUTE = {
    "spheroid-volume-from-diameter": lambda: (
        {"spheroid_volume_ul": spheroid.spheroid_volume_ul(200.0)},
        {"spheroid_volume_ul": "spheroid_volume_ul"},
    ),
    "spheroid-diameter-from-cells": lambda: (
        _sph("96-ula", 1, 1000.0, 20.0),
        {"expected_diameter_um": "expected_diameter_um"},
    ),
    "spheroid-200um-hypoxic": lambda: (
        {"cells_per_spheroid": spheroid.cells_per_spheroid_for_diameter(200.0, 20.0)},
        {"cells_per_spheroid": "cells_per_spheroid"},
    ),
    "spheroid-count-from-suspension": lambda: (
        {"spheroid_count": spheroid.spheroid_count_from_suspension(2.4e5, 1000.0)},
        {"spheroid_count": "spheroid_count"},
    ),
    "spheroid-96ula-medium": lambda: (
        _sph("96-ula", 1, 1000.0, 20.0),
        {"medium_volume_per_spheroid_ul": "medium_volume_per_spheroid_ul"},
    ),
    "spheroid-384ula-medium": lambda: (
        _sph("384-ula", 1, 1000.0, 20.0),
        {"medium_volume_per_spheroid_ul": "medium_volume_per_spheroid_ul"},
    ),
    "spheroid-hanging-drop-total": lambda: (
        {"total_medium_ml": spheroid.total_medium_volume(48, 20.0)},
        {"total_medium_ml": "total_medium_ml"},
    ),
    "spheroid-um-mm-unit-ambiguity": lambda: (
        {"expected_diameter_um": 200.0, "spheroid_volume_ul": spheroid.spheroid_volume_ul(200.0)},
        {"expected_diameter_um": "expected_diameter_um", "spheroid_volume_ul": "spheroid_volume_ul"},
    ),
    "spheroid-growth-72h": lambda: (
        _sph("96-ula", 1, 1000.0, 20.0, doubling_time_h=30.0, culture_duration_h=72.0),
        {"expected_cells_after_growth": "expected_cells_after_growth"},
    ),
    "spheroid-96well-total": lambda: (
        _sph("96-ula", 96, 1000.0, 20.0),
        {"cells_total": "cells_total", "total_medium_ml": "total_medium_ml"},
    ),
    "spheroid-doxorubicin-dosing": lambda: (
        {**_sph("96-ula", 96, 1000.0, 20.0),
         **{"dmso_fraction_vv": dosing.dmso_fraction(5.0, 0.005)}},
        {"cells_total": "cells_total", "total_medium_ml": "total_medium_ml",
         "dmso_fraction_vv": "dmso_fraction_vv"},
    ),
    "blind-spheroid-hepatocyte-formation": lambda: (
        {"cells_per_spheroid": 1000.0},
        {"cells_per_spheroid": "cells_per_spheroid"},
    ),
    "blind-spheroid-96ula-medium": lambda: (
        _sph("96-ula", 1, 1000.0, 20.0),
        {"medium_volume_per_spheroid_ul": "medium_volume_per_spheroid_ul"},
    ),
    "blind-spheroid-384ula-medium": lambda: (
        _sph("384-ula", 1, 1000.0, 20.0),
        {"medium_volume_per_spheroid_ul": "medium_volume_per_spheroid_ul"},
    ),
    "blind-spheroid-hanging-drop": lambda: (
        _sph("hanging-drop", 1, 1000.0, 20.0),
        {"medium_volume_per_spheroid_ul": "medium_volume_per_spheroid_ul"},
    ),
}


@pytest.mark.parametrize("entry", _load(), ids=lambda e: e["id"])
def test_gold_is_self_consistent(entry):
    spec = RECOMPUTE.get(entry["id"])
    assert spec is not None, f"{entry['id']}: no recompute spec — add one to RECOMPUTE"
    derived, mapping = spec()
    for gold_key, derived_key in mapping.items():
        expected = entry["expected"][gold_key]
        got = derived[derived_key]
        assert got == pytest.approx(expected, rel=1e-6), (
            f"{entry['id']}.{gold_key}: gold says {expected}, calculators give {got}"
        )
