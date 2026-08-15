"""Compare lora_v5 (committed) vs lora_v6 (natural-register core expansion) per entry.

The lora_v6 experiment appends natural-register (hand-written prose) templates
to the four core generators (flow/culture/spheroid/pk; composite inherits)
while keeping every v3 row byte-identical and retrains on the v4 split
(results/extractor_11dom_v4, 61k rows). The headline question is whether the
new register lifts the core-domain failures that survived v5 (the 400x100-shear
reading regression, pk accumulation-ratio, plate-culture novel recovery) and
whether it holds the v5 newdomains 4/14 gain without regressing the working
domains.

This script diffs, per entry:

  v5        results/eval_finetuned_{set}_lora_v5.json         (committed)
  v6        results/eval_finetuned_{set}_lora_v6.json         (plain)
  v6+repair results/eval_finetuned_{set}_lora_v6_repair.json  (+ schema repair)

and writes ``results/lora_v6_summary.json`` with aggregate metrics and a
per-entry class for v5->v6 and v5->v6+repair. Greedy decoding is
deterministic, so a ``recovered`` entry is a real fix, not re-sampling noise.

Usage::

    python -m eval.compare_lora_v6                # all six sets
    python -m eval.compare_lora_v6 reading pk     # named sets only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.compare_repair import _usable, _metrics, _classify

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "results"

SETS = ["reading", "blind", "spheroid", "culture", "pk", "newdomains"]
V5 = "lora_v5"
V6 = "lora_v6"


def _entries(set_name: str, adapter: str, repair: bool) -> dict[str, dict | None]:
    suffix = "_repair" if repair else ""
    p = RESULTS / f"eval_finetuned_{set_name}_{adapter}{suffix}.json"
    if not p.exists():
        return {}
    return {e["id"]: e.get("finetuned") for e in json.load(open(p))["per_entry"]}


def compare(set_name: str) -> dict:
    v5 = _entries(set_name, V5, False)
    v6 = _entries(set_name, V6, False)
    v6r = _entries(set_name, V6, True)
    out: dict = {"set": set_name}
    if not v5:
        return {"set": set_name, "missing": "lora_v5 baseline"}
    ids = list(v5)
    out["n"] = len(ids)
    out["lora_v5"] = _metrics([{"finetuned": v5[i]} for i in ids])
    if v6:
        out["lora_v6"] = _metrics([{"finetuned": v6[i]} for i in ids])
    if v6r:
        out["lora_v6_repair"] = _metrics([{"finetuned": v6r[i]} for i in ids])
    # per-entry classes relative to v5
    detail = []
    for i in ids:
        row = {"id": i}
        if v6:
            row["v5_to_v6"] = _classify(v5[i], v6[i])
        if v6r:
            row["v5_to_v6repair"] = _classify(v5[i], v6r[i])
        detail.append(row)
    out["detail"] = detail
    return out


def main(argv: list[str]) -> int:
    sets = argv or SETS
    summary: dict = {}
    print(f"{'set':<12}{'v5':>24}{'v6':>24}{'v6+rep':>24}"
          f"{'recov':>7}{'regr':>7}  (v5->v6+repair)")
    for s in sets:
        r = compare(s)
        summary[s] = r
        if "missing" in r:
            print(f"{s:<12} MISSING {r['missing']}")
            continue
        row = f"{s:<12}"
        for k in ("lora_v5", "lora_v6", "lora_v6_repair"):
            m = r.get(k)
            row += ("MISSING".rjust(24) if m is None else
                    f"{m['usable_rate']:.0%}/{m['self_consistent_rate']:.0%}/{m['hallucination_rate']:.3f}".rjust(24))
        classes = {}
        for d in r["detail"]:
            c = d.get("v5_to_v6repair")
            if c:
                classes[c] = classes.get(c, 0) + 1
        row += f"{classes.get('recovered', 0):>7}{classes.get('regressed', 0):>7}"
        print(row)
    summary_path = RESULTS / "lora_v6_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {summary_path}  (columns: usable / self-cons / hallucination)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
