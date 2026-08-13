"""Render the multi-seed Wilson-CI table for all five gold sets.

The single-run benchmark files are one Bernoulli trial per gold×system. The
seed sweeps (``eval/run_seed_benchmark.py``) pool *seeds* × gold into
successes/trials per system, and ``eval/ci.py`` gives every rate a Wilson 95 %
interval — including the honest 0 %/100 % cases a naive ``k/n`` collapses to a
false 0-width interval. This script renders the combined table for the README.

Usage::

    python -m eval.report_seed_ci
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: (set label, gold n, seed result file)
SETS = [
    ("24-reading", "eval_seed_benchmark.json"),
    ("15-blind", "eval_seed_blind.json"),
    ("15-3D-spheroid", "eval_seed_spheroid.json"),
    ("14-plate-culture", "eval_seed_culture.json"),
    ("14-perfused-PK", "eval_seed_pk.json"),
]

SYSTEMS = ("bare", "soft_gate", "self_verify", "labwright")


def main() -> int:
    print(f"{'set':<17}{'model':<22}{'seed n':<9}{'system':<13}"
          f"{'usable [95% CI]':<26}{'self-cons [95% CI]'}")
    for set_label, fname in SETS:
        path = RESULTS / fname
        if not path.exists():
            print(f"--- {set_label}: {fname} NOT YET PRESENT ---")
            continue
        with open(path) as fh:
            d = json.load(fh)
        seeds = d.get("seeds", "?")
        for model in d.get("models", []):
            pooled = d["pooled"].get(model, {})
            for sysk in SYSTEMS:
                if sysk not in pooled:
                    continue
                p = pooled[sysk]
                u = p["usable_design_rate"]
                sc = p["self_consistent_rate"]
                print(
                    f"{set_label:<17}{model:<22}{seeds:<9}{sysk:<13}"
                    f"{100 * u:5.1f}% {p['usable_ci_str']:<20}"
                    f"{100 * sc:5.1f}% {p['self_consistent_ci_str']}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
