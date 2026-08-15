"""Four-way comparison for the schema-prompt A/B (newdomains, pk).

Reuses :mod:`eval.compare_repair`'s usable/classify rules to diff, per set:

  baseline            results/eval_finetuned_{set}_lora_v4.json
  + repair            results/eval_finetuned_{set}_lora_v4_repair.json
  + schema-prompt     results/eval_finetuned_{set}_lora_v4_schemaprompt.json
  + both              results/eval_finetuned_{set}_lora_v4_schemaprompt_repair.json

Writes ``results/schemaprompt_summary.json`` with per-set metrics and a
per-entry class relative to baseline (recovered / regressed / still_<err>).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.compare_repair import _usable, _classify, _metrics

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "results"
ADAPTER = "lora_v4"

VARIANT_FILES = {
    "baseline": "eval_finetuned_{s}_{a}.json",
    "repair": "eval_finetuned_{s}_{a}_repair.json",
    "schemaprompt": "eval_finetuned_{s}_{a}_schemaprompt.json",
    "schemaprompt_repair": "eval_finetuned_{s}_{a}_schemaprompt_repair.json",
}


def _entries(set_name: str, variant: str) -> dict[str, dict | None]:
    p = RESULTS / VARIANT_FILES[variant].format(s=set_name, a=ADAPTER)
    if not p.exists():
        return {}
    return {e["id"]: e.get("finetuned") for e in json.load(open(p))["per_entry"]}


def compare(set_name: str) -> dict:
    base = _entries(set_name, "baseline")
    out: dict = {"set": set_name}
    ids = list(base)
    for variant in VARIANT_FILES:
        entries = _entries(set_name, variant)
        if not entries:
            out[variant] = {"missing": True}
            continue
        out[variant] = _metrics([{"finetuned": entries[i]} for i in ids])
    # per-entry classes relative to baseline, for each non-baseline variant
    out["detail"] = []
    for i in ids:
        row = {"id": i}
        for variant in ("repair", "schemaprompt", "schemaprompt_repair"):
            entries = _entries(set_name, variant)
            if entries:
                row[variant] = _classify(base[i], entries[i])
        out["detail"].append(row)
    return out


def main(argv: list[str]) -> int:
    sets = argv or ["newdomains", "pk"]
    summary: dict = {}
    print(f"{'set':<12}{'base':>22}{'repair':>22}{'schema':>22}{'schema+rep':>22}")
    for s in sets:
        r = compare(s)
        summary[s] = r
        row = [f"{s:<12}"]
        for variant in ("baseline", "repair", "schemaprompt", "schemaprompt_repair"):
            m = r[variant]
            if m.get("missing"):
                row.append(f"{'MISSING':>22}")
            else:
                row.append(f"{m['usable_rate']:.0%}/{m['self_consistent_rate']:.0%}/{m['hallucination_rate']:.3f}".rjust(22))
        print("".join(row))
    summary_path = RESULTS / "schemaprompt_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {summary_path}  (columns: usable / self-cons / hallucination)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
