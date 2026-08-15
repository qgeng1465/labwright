"""Build the lora_v6 training split (results/extractor_11dom_v4).

lora_v5 fixed the register gap on the seven post-v1 domains (new-domains 0/14 ->
4/14) by regenerating them with hand-written-register prose, but left the five
working domains on their v2 calculator-register templates. The failures that
remain in v5 sit partly in those working domains too: the 400x100-shear reading
regression, the pk-accumulation-ratio regression, and the plate-culture 1/6
novel-goal recovery. The hypothesis is the same one v5 proved for the new
domains: the benchmark's hand-written wording is far enough from the
calculator templates to drop those rows.

The v6 change is therefore *register, not volume*: natural-register prose
templates were appended to the four core generators (flow, culture, spheroid,
pk; composite inherits them). This script keeps every v3 row byte-identical
(so no memorized pattern is lost) and appends ~50 % fresh rows per core domain,
which now sample the mixed register.

Honesty gates
-------------
- Appended rows sample the same source-pinned ranges and build verifier-clean
  designs (the generators' coupling rule, unchanged from v3/v5).
- No training goal is verbatim in any benchmark gold set (checked below), so
  the benchmark stays a *transfer* test, not a memorization test.
- The eval.jsonl and gold_pairs.jsonl are copied byte-identical from v3, so the
  extract eval-report stays comparable.

Usage::

    python -m labwright.extract.make_lora_v6_data
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from labwright.extract.synthetic import (
    generate_culture,
    generate_flow,
    generate_pk,
    generate_spheroid,
)

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent.parent / "results"
V3 = RESULTS / "extractor_11dom_v3"
V4 = RESULTS / "extractor_11dom_v4"

#: Core domains get natural-register rows appended (the seven post-v1 domains
#: already carry hand-written register from v3 and stay byte-identical).
_CORE = {
    "flow": generate_flow,
    "culture": generate_culture,
    "spheroid": generate_spheroid,
    "pk": generate_pk,
}

#: Fraction of each core domain's v3 train count to append.
_APPEND_FRACTION = 0.5

_GOLD_FILES = [
    "eval/gold_experiments.json", "eval/gold_cell_culture.json",
    "eval/gold_spheroid.json", "eval/gold_pk.json", "eval/gold_blind.json",
    "eval/gold_new_domains.json",
]


def _load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def _write(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _benchmark_goals() -> set[str]:
    goals: set[str] = set()
    repo = _HERE.parent.parent
    for rel in _GOLD_FILES:
        path = repo / rel
        data = json.load(open(path, encoding="utf-8"))
        entries = data.get("goals", data) if isinstance(data, dict) else data
        for e in entries:
            g = e.get("goal", e.get("text", ""))
            if g:
                goals.add(g)
    return goals


def main() -> int:
    v3_train = _load(V3 / "train.jsonl")
    core_counts: dict[str, int] = {}
    for r in v3_train:
        dom = r["domain"].split(":")[0]
        if dom in _CORE:
            core_counts[dom] = core_counts.get(dom, 0) + 1
    print("v3 core-domain train counts:", dict(sorted(core_counts.items())))

    rng = random.Random(20260816)
    extra: list[dict] = []
    for dom in sorted(_CORE):
        n = int(core_counts.get(dom, 0) * _APPEND_FRACTION)
        gen = _CORE[dom]
        extra += [gen(rng) for _ in range(n)]
        print(f"  append {n:>5} natural-register {dom} rows")
    rng.shuffle(extra)

    train = v3_train + extra
    V4.mkdir(parents=True, exist_ok=True)
    _write(V4 / "train.jsonl", train)
    shutil.copy2(V3 / "eval.jsonl", V4 / "eval.jsonl")
    shutil.copy2(V3 / "gold_pairs.jsonl", V4 / "gold_pairs.jsonl")

    # honesty gate: no training goal verbatim in any benchmark gold set
    bench_goals = _benchmark_goals()
    overlap = {r["goal"] for r in train} & bench_goals
    print(f"train {len(train)} (v3 {len(v3_train)} + appended {len(extra)}) / "
          f"eval {sum(1 for _ in open(V4/'eval.jsonl'))} (byte-identical)")
    print(f"benchmark-verbatim overlap: {len(overlap)}")
    if overlap:
        for g in list(overlap)[:3]:
            print("  OVERLAP:", g[:120])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
