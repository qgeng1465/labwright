"""Self-consistency checks for the plate-culture gold set.

Every number in ``eval/gold_cell_culture.json`` must be reproducible by the
calculators from the raw inputs stated in its goal prose. This test pins that
property: if someone edits a gold ``expected`` value without re-deriving it
through :mod:`labwright.calc.culture`, the test fails — the gold set cannot
silently drift away from the code that defines the "correct" answer.
"""

import json
import os

import pytest

from labwright.calc import culture
from labwright.design import derive_culture

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(_HERE, "eval", "gold_cell_culture.json")

#: CulturePlan fields a gold ``expected`` key is allowed to target.
_ALLOWED_KEYS = {
    "seed_per_well", "total_seed_count", "medium_volume_per_well_ml",
    "total_medium_ml", "expected_confluence_pct", "wells",
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
            f"{g['id']}: expected keys must map onto CulturePlan fields, got "
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


def _plate(plate_format, density, wells=1, **extra):
    return derive_culture(
        dict(plate_format=plate_format, wells=wells,
             seeding_density_cells_cm2=density, **extra)
    )


def _hemocytometer_wells(mean_cells, dilution, volume_ml, plate_format, density):
    conc = culture.hemocytometer_count(mean_cells, dilution)
    total = conc * volume_ml
    per_well = culture.cells_per_well(density, plate_format)
    return total / per_well


RECOMPUTE = {
    "plate-96well-hepg2-seed": lambda: (
        _plate("96", 1e4),
        {"seed_per_well": "seed_per_well", "medium_volume_per_well_ml": "medium_volume_per_well_ml"},
    ),
    "plate-6well-phh-seed": lambda: (
        _plate("6", 1.5e5, wells=6),
        {"seed_per_well": "seed_per_well", "total_medium_ml": "total_medium_ml"},
    ),
    "plate-hemocytometer-seed-96well": lambda: (
        {"wells": _hemocytometer_wells(32, 2, 0.45, "96", 1e4)},
        {"wells": "wells"},
    ),
    "plate-confluence-hepg2-72h": lambda: (
        _plate("96", 2e4, confluent_density_cells_cm2=1e6, doubling_time_h=30, culture_duration_h=72),
        {"expected_confluence_pct": "expected_confluence_pct"},
    ),
    "plate-24well-seed-from-count": lambda: (
        _plate("24", 8e4),
        {"seed_per_well": "seed_per_well", "medium_volume_per_well_ml": "medium_volume_per_well_ml"},
    ),
    "plate-split-replating": lambda: (
        _plate("6", 2e4, wells=4),
        {"total_seed_count": "total_seed_count"},
    ),
    "plate-48well-medium": lambda: (
        _plate("48", 1e4),
        {"medium_volume_per_well_ml": "medium_volume_per_well_ml"},
    ),
    "plate-thaw-viability-6well": lambda: (
        {"wells": 1e6 * 0.96 / culture.cells_per_well(1e4, "6")},
        {"wells": "wells"},
    ),
    "plate-12well-seed-hepg2": lambda: (
        _plate("12", 1e5),
        {"seed_per_well": "seed_per_well", "medium_volume_per_well_ml": "medium_volume_per_well_ml"},
    ),
    "plate-96well-total-medium": lambda: (
        _plate("96", 1e4, wells=96),
        {"total_medium_ml": "total_medium_ml"},
    ),
    "blind-6well-phh-sandwich-seed": lambda: (
        _plate("6", 1.5e5, wells=6),
        {"seed_per_well": "seed_per_well", "total_medium_ml": "total_medium_ml"},
    ),
    "blind-24well-phh-seed": lambda: (
        _plate("24", 1.5e5),
        {"seed_per_well": "seed_per_well"},
    ),
    "blind-96well-area-and-medium": lambda: (
        _plate("96", 1e4),
        {"medium_volume_per_well_ml": "medium_volume_per_well_ml"},
    ),
    "blind-hemocytometer-phh-96well": lambda: (
        {"wells": _hemocytometer_wells(40, 2, 0.36, "96", 1.5e5)},
        {"wells": "wells"},
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
