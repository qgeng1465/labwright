"""Tag the pre-existing gold sets with LabMath-Bench levels and merge them in.

The ~100 committed golds (blind / cell-culture / cold-expansion / experiments /
new-domains / pk / spheroid) predate LabMath-Bench and carry no ``level``. This
script assigns each a level from the reviewer's three difficulty axes — L1
fluid & spatial engineering, L2 biochemical stoichiometry, L3 pipeline
parameterization — by the derived keys the entry targets, tags a ``difficulty``
the same way the generator does, and writes the merged dataset
``eval/gold_labmath_combined.json`` (generated 510 + existing ~100).

The committed gold files are left untouched — this only *reads* them, so a
LabMath-Bench run can use the combined file without disturbing the gold that
backs the committed result JSONs.

Usage::

    python -m eval.tag_existing_levels --out eval/gold_labmath_combined.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Derived-key → level buckets (see make_labmath_bench.py for the same split).
_L1_KEYS = {
    # microfluidic channel transport (the reviewer's shear-stress example)
    "shear_pa", "reynolds", "pressure_drop_pa", "residence_time_s",
    "channel_volume_ul", "mean_velocity_mms", "flow_rate_uLmin",
    # spatial/gravity organ-chip fluidics
    "hydrostatic_head_pa", "peak_wall_shear_pa", "oscillatory_shear_index",
    "volume_per_half_cycle_ul", "cycles_per_hour", "womersley_number",
    "peak_shear_pa", "pulsatility_index", "driven_flow_rate_uLmin",
    "shear_ratio_to_target",
    # gradient / oxygen / breathing spatial-transport fields
    "steepness_um_per_mm", "midpoint_conc_um", "relaxation_time_s",
    "flux_mol_m2s", "dissolved_o2_mM", "penetration_depth_um",
    "necrotic_fraction", "demand_umol_min", "breaths_per_minute",
    "cyclic_displacement_um", "strain_rate_per_s", "total_cycles",
    "stretch_duty_fraction", "ali_liquid_film_um",
    # organ-scale flow scaling
    "organ_flow_fraction", "organ_flow_rate_mlmin", "cells_in_organ",
    "transit_time_s", "residence_time_match_error_s", "allometric_scale",
}
_L2_KEYS = {
    # seeding / stoichiometry
    "seed_count", "seed_per_well", "total_seed_count",
    "medium_volume_per_well_ml", "total_medium_ml", "expected_confluence_pct",
    "wells", "spheroid_volume_ul", "expected_diameter_um",
    "cells_per_spheroid", "spheroid_count", "medium_volume_per_spheroid_ul",
    "cells_total", "expected_cells_after_growth", "dmso_fraction_vv",
    # barrier / PK / enzyme / coculture
    "teer_ohm_cm2", "papp_cm_s", "clearance_mL_min",
    "extraction_ratio", "clearance_uLmin", "half_life_h", "accumulation_ratio",
    "mass_cleared_ug_h", "inlet_concentration_uM", "outlet_concentration_uM",
    "fractional_activity", "percent_inhibition", "ic50_um", "apparent_km_um",
    "velocity_umol_min", "inhibitor_substrate_ratio",
    "cells_per_well_a", "cells_per_well_b", "total_cells_a", "total_cells_b",
    "seeding_ratio_ab",
}
_L3_KEYS = {
    "n_per_group", "n_arrays", "n_chips", "n_expected_failed_arrays",
    "bed_size_mb", "n_per_chr_files", "per_chr_bed_size_mb",
    "evaporation_rate_ul_hr", "residual_volume_ul", "edge_evaporation_factor",
}

_GOLD_FILES = (
    "gold_blind.json", "gold_cell_culture.json", "gold_cold_expansion.json",
    "gold_experiments.json", "gold_new_domains.json", "gold_pk.json",
    "gold_spheroid.json",
)


def _level_for(expected: dict) -> str:
    """Pick the level with the most overlapping derived keys (ties → higher L)."""
    scores = {"L1": 0, "L2": 0, "L3": 0}
    for key in expected:
        if key in _L1_KEYS:
            scores["L1"] += 1
        if key in _L2_KEYS:
            scores["L2"] += 1
        if key in _L3_KEYS:
            scores["L3"] += 1
    # highest score wins; ties resolved L3 > L2 > L1 (pipeline > stoichiometry >
    # fluid), mirroring the generator's per-domain single-level assignment.
    best = max(scores, key=lambda lv: (scores[lv], "L3L2L1".find(lv)))
    return best


def _difficulty_for(n_targets: int) -> str:
    if n_targets <= 2:
        return "easy"
    if n_targets <= 4:
        return "medium"
    return "hard"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "gold_labmath_combined.json"))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    generated = json.load(open(os.path.join(here, "gold_labmath_bench.json"), encoding="utf-8"))

    tagged: list[dict] = []
    for name in _GOLD_FILES:
        path = os.path.join(here, name)
        for e in json.load(open(path, encoding="utf-8")):
            if e.get("level"):
                continue  # already tagged
            tagged.append({**e, "level": _level_for(e.get("expected", {})),
                           "difficulty": _difficulty_for(len(e.get("expected", {})))})

    combined = generated + tagged
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    from collections import Counter
    by_level = Counter(e["level"] for e in combined)
    print(f"tagged {len(tagged)} existing golds")
    print(f"combined -> {args.out}  ({len(combined)} entries)")
    print("by level :", dict(sorted(by_level.items())))
    if any(by_level[lv] < 140 for lv in ("L1", "L2", "L3")):
        print("[warn] a level has < 140 entries", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
