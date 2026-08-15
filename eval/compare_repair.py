"""Compare lora_v4 baseline vs lora_v4+repair per entry and in aggregate.

The schema-repair experiment (``eval/run_lora_v4_repair.sh``) re-runs the
extractor with up to 2 extra attempts that re-feed the validator error. This
script diffs the two per-entry records, reports the aggregate usable /
self-consistent / hallucination rates for both variants, classifies what the
repair changed, and writes ``results/repair_summary.json``.

Greedy decoding is deterministic, so entries the repair does not touch are
byte-identical — a ``recovered`` entry is a genuine schema-error fix, not
re-sampling noise.

Usage::

    python -m eval.compare_repair                # all six sets
    python -m eval.compare_repair newdomains pk  # named sets only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "results"
ADAPTER = "lora_v4"

SETS = ["reading", "blind", "spheroid", "culture", "pk", "newdomains"]


def _usable(f) -> bool:
    """A design is usable iff it produced a valid plan hitting every target."""
    if not f:
        return False
    if not f.get("plan") or not f.get("valid"):
        return False
    rec = f.get("recovery") or {}
    return bool(rec) and all(abs(v) <= 0.05 for v in rec.values())


def _metrics(per_entry):
    n = len(per_entry)
    usable = sum(1 for e in per_entry if _usable(e.get("finetuned"))) / n
    hall = sum(1 for e in per_entry if e.get("finetuned", {}).get("hallucination_rate", 1.0) > 0) / n
    self_cons = 1.0 - hall
    return {"n": n, "usable_rate": round(usable, 4), "hallucination_rate": round(hall, 4),
            "self_consistent_rate": round(self_cons, 4)}


def _classify(base_f, rep_f):
    base_ok, rep_ok = _usable(base_f), _usable(rep_f)
    if base_ok and rep_ok:
        return "unchanged_ok"
    if not base_ok and rep_ok:
        return "recovered"
    if base_ok and not rep_ok:
        return "regressed"
    base_err = ((base_f or {}).get("error") or "").split(":")[0]
    rep_err = ((rep_f or {}).get("error") or "").split(":")[0]
    if base_err == rep_err and base_err:
        return f"still_{base_err}"
    if base_err == "schema_error" and rep_err == "schema_error":
        return "schema_to_schema"
    return f"{base_err or 'fail'}->{rep_err or 'fail'}"


def compare(set_name: str) -> dict:
    base_p = RESULTS / f"eval_finetuned_{set_name}_{ADAPTER}.json"
    rep_p = RESULTS / f"eval_finetuned_{set_name}_{ADAPTER}_repair.json"
    if not base_p.exists() or not rep_p.exists():
        return {"set": set_name, "missing": [p.name for p in (base_p, rep_p) if not p.exists()]}
    base = {e["id"]: e.get("finetuned") for e in json.load(open(base_p))["per_entry"]}
    rep = {e["id"]: e.get("finetuned") for e in json.load(open(rep_p))["per_entry"]}
    ids = list(base)
    classes: dict[str, int] = {}
    detail = []
    for i in ids:
        c = _classify(base[i], rep[i])
        classes[c] = classes.get(c, 0) + 1
        if c != "unchanged_ok":
            detail.append({"id": i, "change": c,
                           "base": (base[i] or {}).get("error", ""),
                           "repair": (rep[i] or {}).get("error", "")})
    b_entries = [{"finetuned": base[i]} for i in ids]
    r_entries = [{"finetuned": rep[i]} for i in ids]
    return {"set": set_name, "baseline": _metrics(b_entries), "repair": _metrics(r_entries),
            "classes": classes, "detail": detail}


def main(argv: list[str]) -> int:
    sets = argv or SETS
    out = {}
    print(f"{'set':<12}{'baseline':>26}{'repair':>26}{'recovered':>10}{'regressed':>10}")
    for s in sets:
        r = compare(s)
        out[s] = r
        if "missing" in r:
            print(f"{s:<12} MISSING {r['missing']}")
            continue
        b, rep = r["baseline"], r["repair"]
        d = r["classes"]
        print(f"{s:<12}"
              f"  {b['usable_rate']:.0%}/{b['self_consistent_rate']:.0%}/{b['hallucination_rate']:.3f}"
              f"  {rep['usable_rate']:.0%}/{rep['self_consistent_rate']:.0%}/{rep['hallucination_rate']:.3f}"
              f"  {d.get('recovered', 0):>7}{d.get('regressed', 0):>9}")
    summary_path = RESULTS / "repair_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {summary_path} (columns: usable / self-cons / hallucination)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
