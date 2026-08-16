"""Gold-pair supervision split of the six benchmark sets, from committed JSONs.

A gold goal is *supervised* ("seen") when its id appears in the training
gold pairs (`results/extractor_11dom_v4/gold_pairs.jsonl`): that means a
source-pinned answer for the goal sits verbatim in the training data, so the
goal tests memorization more than transfer. The 46 pairs split as
24 reading + 8 spheroid + 8 culture + 6 PK; the blind and new-domain sets
have no gold pairs by design (their goals are withheld).

This reproduces the numbers the README / paper docx use when they split each
set's usable rate into "supervised" vs "never-seen" recoveries. It reads only
committed result files — nothing here is re-typed.

Usage::

    python eval/supervised_split.py            # human table
    python eval/supervised_split.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (set name, v6 per-entry file)
SETS = {
    "reading": "results/eval_finetuned_reading_lora_v6.json",
    "blind": "results/eval_finetuned_blind_lora_v6.json",
    "spheroid": "results/eval_finetuned_spheroid_lora_v6.json",
    "culture": "results/eval_finetuned_culture_lora_v6.json",
    "pk": "results/eval_finetuned_pk_lora_v6.json",
    "newdomains": "results/eval_finetuned_newdomains_lora_v6.json",
}


def load_gold_pair_ids() -> set[str]:
    """The goal ids pinned by the 46 gold-pair supervision rows."""
    ids: set[str] = set()
    for line in (ROOT / "results/extractor_11dom_v4/gold_pairs.jsonl").open():
        obj = json.loads(line)
        if obj.get("gold"):
            ids.add(obj["gold"].strip())
    return ids


def is_usable(e: dict) -> bool:
    """The derive() usable predicate: no hallucination, recovery non-empty,
    every recovery residual within ±5 %."""
    f = e.get("finetuned", {})
    return (
        f.get("hallucination_rate") == 0.0
        and bool(f.get("recovery"))
        and all(err <= 0.05 for err in f["recovery"].values())
    )


def is_sc(e: dict) -> bool:
    return e.get("finetuned", {}).get("hallucination_rate") == 0.0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    pair_ids = load_gold_pair_ids()
    out: dict[str, dict] = {}
    rows: list[tuple[str, list[str], list[str]]] = []
    for name, rel in SETS.items():
        data = json.load((ROOT / rel).open())
        entries = data["per_entry"]
        seen = [e for e in entries if e["id"] in pair_ids]
        novel = [e for e in entries if e["id"] not in pair_ids]
        seen_ok = [e["id"] for e in seen if is_usable(e)]
        novel_ok = [e["id"] for e in novel if is_usable(e)]
        out[name] = {
            "n": len(entries),
            "supervised": len(seen),
            "supervised_usable": len(seen_ok),
            "never_seen": len(novel),
            "never_seen_usable": len(novel_ok),
            "usable": sum(1 for e in entries if is_usable(e)),
            "self_consistent": sum(1 for e in entries if is_sc(e)),
            "supervised_usable_ids": seen_ok,
            "never_seen_usable_ids": novel_ok,
        }
        rows.append((name, seen_ok, novel_ok))

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"gold-pair ids: {len(pair_ids)}")
    print("supervised  = goal id pinned in gold_pairs.jsonl (memorization test)")
    print("never-seen  = no gold pair (transfer test)\n")
    hdr = (f"{'set':10s} {'n':>3s} {'sup':>4s} {'sup_ok':>6s} "
           f"{'novel':>6s} {'novel_ok':>8s} {'usable':>6s} {'sc':>3s}")
    print(hdr)
    print("-" * len(hdr))
    for name, seen_ok, novel_ok in rows:
        r = out[name]
        print(f"{name:10s} {r['n']:3d} {r['supervised']:4d} {r['supervised_usable']:6d} "
              f"{r['never_seen']:6d} {r['never_seen_usable']:8d} "
              f"{r['usable']:6d} {r['self_consistent']:3d}")
        print(f"  supervised_ok: {', '.join(seen_ok) or '—'}")
        print(f"  never_seen_ok: {', '.join(novel_ok) or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
