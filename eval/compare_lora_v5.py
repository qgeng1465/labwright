"""Compare lora_v4 (committed) vs lora_v5 (hand-written-register retrain) per entry.

The lora_v5 experiment retrains the extractor on results/extractor_11dom_v3:
the seven v2 domains are regenerated with natural-register (hand-written)
prose variants, the working domains stay byte-identical. The headline question
is whether the new register lifts the newdomains set from v4's 0/14 usable
(the schema-prompt A/B proved the gap is data, not prompt).

This script diffs, per entry:

  v4        results/eval_finetuned_{set}_lora_v4.json         (committed)
  v5        results/eval_finetuned_{set}_lora_v5.json         (plain)
  v5+repair results/eval_finetuned_{set}_lora_v5_repair.json  (+ schema repair)

and writes ``results/lora_v5_summary.json`` with aggregate metrics and a
per-entry class for v4->v5 and v4->v5+repair. Greedy decoding is
deterministic, so a ``recovered`` entry is a real fix, not re-sampling noise.

Usage::

    python -m eval.compare_lora_v5                # all six sets
    python -m eval.compare_lora_v5 newdomains pk  # named sets only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.compare_repair import _usable, _metrics, _classify

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "results"

SETS = ["reading", "blind", "spheroid", "culture", "pk", "newdomains"]
V4 = "lora_v4"
V5 = "lora_v5"


def _entries(set_name: str, adapter: str, repair: bool) -> dict[str, dict | None]:
    suffix = "_repair" if repair else ""
    p = RESULTS / f"eval_finetuned_{set_name}_{adapter}{suffix}.json"
    if not p.exists():
        return {}
    return {e["id"]: e.get("finetuned") for e in json.load(open(p))["per_entry"]}


def compare(set_name: str) -> dict:
    v4 = _entries(set_name, V4, False)
    v5 = _entries(set_name, V5, False)
    v5r = _entries(set_name, V5, True)
    out: dict = {"set": set_name}
    if not v4:
        return {"set": set_name, "missing": "lora_v4 baseline"}
    ids = list(v4)
    out["n"] = len(ids)
    out["lora_v4"] = _metrics([{"finetuned": v4[i]} for i in ids])
    if v5:
        out["lora_v5"] = _metrics([{"finetuned": v5[i]} for i in ids])
    if v5r:
        out["lora_v5_repair"] = _metrics([{"finetuned": v5r[i]} for i in ids])
    # per-entry classes relative to v4
    detail = []
    for i in ids:
        row = {"id": i}
        if v5:
            row["v4_to_v5"] = _classify(v4[i], v5[i])
        if v5r:
            row["v4_to_v5repair"] = _classify(v4[i], v5r[i])
        detail.append(row)
    out["detail"] = detail
    return out


def main(argv: list[str]) -> int:
    sets = argv or SETS
    summary: dict = {}
    print(f"{'set':<12}{'v4':>24}{'v5':>24}{'v5+rep':>24}"
          f"{'recov':>7}{'regr':>7}  (v4->v5+repair)")
    for s in sets:
        r = compare(s)
        summary[s] = r
        if "missing" in r:
            print(f"{s:<12} MISSING {r['missing']}")
            continue
        row = f"{s:<12}"
        for k in ("lora_v4", "lora_v5", "lora_v5_repair"):
            m = r.get(k)
            row += ("MISSING".rjust(24) if m is None else
                    f"{m['usable_rate']:.0%}/{m['self_consistent_rate']:.0%}/{m['hallucination_rate']:.3f}".rjust(24))
        classes = {}
        for d in r["detail"]:
            c = d.get("v4_to_v5repair")
            if c:
                classes[c] = classes.get(c, 0) + 1
        row += f"{classes.get('recovered', 0):>7}{classes.get('regressed', 0):>7}"
        print(row)
    summary_path = RESULTS / "lora_v5_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {summary_path}  (columns: usable / self-cons / hallucination)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
