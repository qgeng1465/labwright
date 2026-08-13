"""Self-consistency checks for the perfused-system PK gold set.

Every number in ``eval/gold_pk.json`` must be reproducible by the PK
calculators from the raw inputs stated in its goal prose. This test pins that
property: if someone edits a gold ``expected`` value without re-deriving it
through :mod:`labwright.calc.pk`, the test fails — the gold set cannot
silently drift away from the code that defines the "correct" answer.
"""

import json
import os

import pytest

from labwright.calc import pk
from labwright.design import derive_pk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(_HERE, "eval", "gold_pk.json")

#: PkPlan fields a gold ``expected`` key is allowed to target.
_ALLOWED_KEYS = {
    "extraction_ratio", "clearance_uLmin", "half_life_h", "accumulation_ratio",
    "mass_cleared_ug_h", "inlet_concentration_uM", "outlet_concentration_uM",
    "flow_rate_uLmin",
}


def _load():
    with open(GOLD_PATH) as fh:
        return json.load(fh)


def test_structure():
    gold = _load()
    assert len(gold) >= 13
    ids = [g["id"] for g in gold]
    assert len(ids) == len(set(ids)), "gold ids must be unique"
    for g in gold:
        assert g["goal"]
        assert g["expected"], f"{g['id']}: expected must be non-empty"
        assert set(g["expected"]) <= _ALLOWED_KEYS, (
            f"{g['id']}: expected keys must map onto PkPlan fields, got "
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


def _pk(**raw):
    return derive_pk(raw)


RECOMPUTE = {
    "pk-extraction-ratio": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2),
        {"extraction_ratio": "extraction_ratio"},
    ),
    "pk-clearance": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2),
        {"clearance_uLmin": "clearance_uLmin"},
    ),
    "pk-half-life": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2,
            system_volume_uL=200),
        {"half_life_h": "half_life_h"},
    ),
    "pk-accumulation-ratio": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2,
            system_volume_uL=200, dose_interval_h=24),
        {"accumulation_ratio": "accumulation_ratio"},
    ),
    "pk-mass-cleared": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2,
            molecular_weight_g_mol=464),
        {"mass_cleared_ug_h": "mass_cleared_ug_h"},
    ),
    "pk-complete-clearance-panel": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2,
            system_volume_uL=200, molecular_weight_g_mol=464),
        {"extraction_ratio": "extraction_ratio", "clearance_uLmin": "clearance_uLmin",
         "half_life_h": "half_life_h", "mass_cleared_ug_h": "mass_cleared_ug_h"},
    ),
    "pk-mM-unit-trap": lambda: (
        _pk(inlet_concentration_uM=500, outlet_concentration_uM=350, flow_rate_uLmin=2),
        {"inlet_concentration_uM": "inlet_concentration_uM",
         "outlet_concentration_uM": "outlet_concentration_uM"},
    ),
    "pk-half-life-min-trap": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=7, flow_rate_uLmin=2,
            system_volume_uL=200),
        {"half_life_h": "half_life_h"},
    ),
    "pk-cell-free-subtraction": lambda: (
        {"clearance_uLmin": 0.45 - 0.05},
        {"clearance_uLmin": "clearance_uLmin"},
    ),
    "pk-repeat-dose-24h": lambda: (
        {"accumulation_ratio": pk.accumulation_ratio(12, 24)},
        {"accumulation_ratio": "accumulation_ratio"},
    ),
    "pk-high-extraction-clearance": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=2, flow_rate_uLmin=1),
        {"clearance_uLmin": "clearance_uLmin"},
    ),
    "pk-low-extraction-clearance": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=9, flow_rate_uLmin=2),
        {"clearance_uLmin": "clearance_uLmin"},
    ),
    "blind-pk-high-extraction-propranolol": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=2, flow_rate_uLmin=1),
        {"extraction_ratio": "extraction_ratio", "clearance_uLmin": "clearance_uLmin"},
    ),
    "blind-pk-low-extraction-antipyrine": lambda: (
        _pk(inlet_concentration_uM=10, outlet_concentration_uM=9, flow_rate_uLmin=2),
        {"extraction_ratio": "extraction_ratio", "clearance_uLmin": "clearance_uLmin"},
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
