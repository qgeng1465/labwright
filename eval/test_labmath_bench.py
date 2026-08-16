"""Tests for the LabMath-Bench dataset and generator.

The reviewer-facing benchmark asks for 500–1000 QA pairs across three levels
(L1 fluid/spatial, L2 biochemical stoichiometry, L3 pipeline parameterization),
each entry self-consistent with the deterministic calculators. These tests pin:

* the committed dataset: size ≥ 500, every level ≥ 140, valid tags, finite and
  strictly positive expected values, every expected key a bare derived key,
* benchmark routing: every new-domain entry resolves to its own block,
* generator determinism: the same seed reproduces the committed JSON byte-for-byte,
* the merged file (generated + tagged existing golds) keeps all three levels ≥ 140.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import GoldExperiment, _new_domain_block  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEW_DOMAINS = ("bioprinting", "coculture", "enzyme", "champ", "plink", "solvent")
_LEVEL_DERIVED = {
    "bioprinting": ("extrusion_volume_nl", "print_time_s", "extrusion_rate_nl_min",
                    "filament_mass_ug", "lines_to_cover"),
    "coculture": ("cells_per_well_a", "cells_per_well_b", "total_cells_a",
                  "total_cells_b", "seeding_ratio_ab"),
    "enzyme": ("fractional_activity", "percent_inhibition", "ic50_um",
               "apparent_km_um", "velocity_umol_min", "inhibitor_substrate_ratio"),
    "champ": ("n_arrays", "n_chips", "n_expected_failed_arrays"),
    "plink": ("bed_size_mb", "n_per_chr_files", "per_chr_bed_size_mb"),
    "solvent": ("evaporation_rate_ul_hr", "residual_volume_ul",
                "edge_evaporation_factor"),
}


def _load(path: str) -> list[dict]:
    with open(os.path.join(_HERE, path), encoding="utf-8") as fh:
        return json.load(fh)


def test_labmath_bench_dataset_shape():
    entries = _load("gold_labmath_bench.json")
    assert len(entries) >= 500
    from collections import Counter
    by_level = Counter(e["level"] for e in entries)
    for level in ("L1", "L2", "L3"):
        assert by_level[level] >= 140, f"{level}: {by_level[level]} < 140"
    for e in entries:
        assert e["level"] in ("L1", "L2", "L3")
        assert e["difficulty"] in ("easy", "medium", "hard")
        assert e["scenario"] == "complete-info"
        assert e["source"], e["id"]
        assert e["expected"], e["id"]
        for v in e["expected"].values():
            assert v == v and v > 0, f"{e['id']}: non-finite/positive expected {v}"


def test_labmath_bench_expected_keys_are_domain_derived():
    entries = _load("gold_labmath_bench.json")
    for e in entries:
        domain = e["id"].split("-")[1]
        if domain not in _LEVEL_DERIVED:
            continue  # flow entries use the derived block's keys
        allowed = set(_LEVEL_DERIVED[domain])
        assert set(e["expected"]) <= allowed, f"{e['id']}: {set(e['expected'])} !<= {allowed}"


def test_labmath_bench_routes_through_benchmark():
    golds = [GoldExperiment(**e) for e in _load("gold_labmath_bench.json")]
    for g in golds:
        domain = g.id.split("-")[1]
        if domain in _NEW_DOMAINS:
            assert _new_domain_block(g) == domain, f"{g.id} -> {_new_domain_block(g)}"


def test_labmath_generator_is_deterministic(tmp_path):
    """Same seed reproduces the committed dataset byte-for-byte."""
    import importlib
    from eval import make_labmath_bench as mg
    importlib.reload(mg)  # clear any prior module state

    out = tmp_path / "gold.json"
    old_argv = sys.argv
    sys.argv = ["make_labmath_bench", "--seed", "20260817", "--out", str(out)]
    try:
        assert mg.main() == 0
    finally:
        sys.argv = old_argv
    committed = json.dumps(_load("gold_labmath_bench.json"), indent=1, ensure_ascii=False)
    regenerated = json.dumps(json.load(open(out, encoding="utf-8")), indent=1,
                             ensure_ascii=False)
    assert regenerated == committed


def test_combined_dataset_keeps_levels():
    entries = _load("gold_labmath_combined.json")
    assert len(entries) >= 600
    from collections import Counter
    by_level = Counter(e["level"] for e in entries)
    for level in ("L1", "L2", "L3"):
        assert by_level[level] >= 140, f"{level}: {by_level[level]} < 140"
    # every entry loads as a GoldExperiment (no unknown tags)
    for e in entries:
        GoldExperiment(**e)
