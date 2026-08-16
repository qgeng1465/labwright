"""Deterministic stratified ~100-entry subset of gold_labmath_combined.json.

The full LabMath-Bench ablation (bare / code_interpreter / labwright) runs on
all 610 entries; the *extended* ablation (soft_gate, self_verify, tool_no_gate,
labwright_iter) and the Thoth-8B cross-model row are run on a representative
slice to bound API spend, as stated in the README. This script draws a fixed,
level-stratified sample (≈34 per L1/L2/L3, evenly spaced across each level's
pool) so the subset always covers every difficulty axis and both the generated
entries and the tagged historical golds.

Usage::

    python -m eval.make_labmath_subset --seed 20260817 --out eval/gold_labmath_subset.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PER_LEVEL = 34


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=os.path.join(_HERE, "gold_labmath_subset.json"))
    args = ap.parse_args()

    with open(os.path.join(_HERE, "gold_labmath_combined.json"), encoding="utf-8") as fh:
        entries = json.load(fh)

    by_level: dict[str, list[dict]] = {}
    for e in entries:
        by_level.setdefault(e["level"], []).append(e)

    subset: list[dict] = []
    for level in ("L1", "L2", "L3"):
        pool = sorted(by_level[level], key=lambda e: e["id"])
        # Deterministic even stride across the level's pool (independent of the
        # rng, so the selection is auditable by inspection).
        step = max(1, len(pool) // _PER_LEVEL)
        subset.extend(pool[::step][:_PER_LEVEL])

    subset.sort(key=lambda e: e["id"])
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(subset, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    from collections import Counter
    print(f"subset -> {args.out}  ({len(subset)} entries)")
    print("by level   :", dict(sorted(Counter(e["level"] for e in subset).items())))
    # show generated-vs-tagged provenance
    gen = sum(1 for e in subset if e["id"].startswith("lmb-"))
    print(f"generated: {gen}   tagged-existing: {len(subset) - gen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
