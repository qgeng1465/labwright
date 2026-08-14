"""Bucket the fine-tuned extractor's spheroid/pk benchmark results honestly.

The spheroid and pk benchmark golds partly double as *training* instances: the
LoRA fine-tune folds 8/15 spheroid and 6/14 pk gold goals into its gold pairs
(labwright/extract/gold_pairs.py). After that retrain, those columns would
measure memorization as much as generalization. This script re-derives the three
headline metrics separately for:

- **folded-in** — benchmark gold whose goal appears in a training gold pair;
- **excluded**  — spheroid/pk gold deliberately NOT used in training (the
  honest in-domain test);
- **blind**     — golds prefixed ``blind-`` (out-of-distribution framing).

The metrics are recomputed from ``per_entry`` with the *exact* rules in
eval/benchmark.py (usable = fraction with ``valid``; self-consistent = fraction
with zero verifier errors; hallucination = mean per-entry hallucination).

Usage::

    python -m eval.split_finetuned_report \
        --results results/eval_finetuned_spheroid.json results/eval_finetuned_pk.json \
        --gold-dir eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: result basename → benchmark gold file, mirroring eval/run_finetuned_benchmark.py
_GOLD_FILES = {
    "reading": "gold_experiments.json",
    "culture": "gold_cell_culture.json",
    "spheroid": "gold_spheroid.json",
    "pk": "gold_pk.json",
    "blind": "gold_blind.json",
}


def _canon(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, help="eval_finetuned_*.json files")
    ap.add_argument("--gold-pairs", default="results/extractor/gold_pairs.jsonl")
    ap.add_argument("--gold-dir", default="eval", help="directory with eval/gold_*.json")
    args = ap.parse_args()

    with open(args.gold_pairs, encoding="utf-8") as fh:
        pair_goals = [_canon(json.loads(line)["goal"]) for line in fh]

    for path in sorted(args.results):
        with open(path, encoding="utf-8") as fh:
            result = json.load(fh)
        dom = os.path.basename(path).replace("eval_finetuned_", "").replace(".json", "")
        gold_file = _GOLD_FILES.get(dom)
        gold_goal: dict[str, str] = {}
        if gold_file:
            with open(os.path.join(args.gold_dir, gold_file), encoding="utf-8") as fh:
                for g in json.load(fh):
                    gold_goal[g["id"]] = _canon(g["goal"])

        def bucket_for(gid: str) -> str:
            if gid.startswith("blind-"):
                return "blind"
            goal = gold_goal.get(gid, "")
            if any(goal in pg or pg in goal for pg in pair_goals):
                return "folded-in"
            return "excluded"

        def metrics(entries: list[dict]) -> dict:
            if not entries:
                return {"n": 0, "usable": float("nan"),
                        "self_consistent": float("nan"), "hallucination": float("nan")}
            halls = [e["finetuned"]["hallucination_rate"] for e in entries]
            return {
                "n": len(entries),
                "usable": sum(1.0 for e in entries if e["finetuned"].get("valid")) / len(entries),
                "self_consistent": sum(1.0 for h in halls if h == 0.0) / len(entries),
                "hallucination": sum(halls) / len(entries),
            }

        buckets: dict[str, list[dict]] = {"folded-in": [], "excluded": [], "blind": []}
        for e in result["per_entry"]:
            buckets[bucket_for(e["id"])].append(e)
        m = metrics(result["per_entry"])
        print(f"== {dom}  (n={m['n']}, usable {m['usable']:.3f}, hall {m['hallucination']:.3f}) ==")
        for bucket in ("folded-in", "excluded", "blind"):
            b = metrics(buckets[bucket])
            print(f"  {bucket:<10} n={b['n']:>2}  usable={b['usable']:.3f}  "
                  f"self-cons={b['self_consistent']:.3f}  hall={b['hallucination']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
