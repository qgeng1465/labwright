"""Build the lora_v5 training split (results/extractor_11dom_v3).

The lora_v4 model selects the right block type on the seven v2 domains but
drops REQUIRED fields on hand-written benchmark prose (0/14 new-domains
usable). Root cause: the v2 synthetic goals phrase those domains in a
calculator register ("total resistance X against a Y blank") that never taught
the benchmark's paraphrases ("cell-free blank reads 150 Ω; the seeded
monolayer totals 1900 Ω"). The fix is *data*: regenerate the seven new-domain
rows with hand-written-register prose variants (added to
``labwright.extract.synthetic`` for lora_v5) while keeping the working
flow/culture/spheroid/pk/composite rows byte-identical from v2.

Honesty gates
-------------
- Every new-domain raw still samples from the same source-pinned ranges and
  builds a verifier-clean design (the generators' coupling rule).
- No training goal is verbatim in any benchmark gold set (checked below), so
  the benchmark stays a *transfer* test, not a memorization test.
- The v2 eval rows for the seven new domains are replaced by fresh held-out
  rows (write_split dedups eval against train by goal and raw string).

Usage::

    python -m labwright.extract.make_lora_v5_data
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from labwright.extract.synthetic import (
    generate_barrier, generate_breathing, generate_gradient, generate_oxygen,
    generate_pulsatile, generate_pumpless, generate_scaling, write_split,
)

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent.parent / "results"
V2 = RESULTS / "extractor_11dom_v2"
V3 = RESULTS / "extractor_11dom_v3"

_NEW_DOMAIN_BLOCKS = {
    "barrier", "oxygen", "pumpless", "breathing", "pulsatile", "scaling", "gradient",
}
_GENERATORS = {
    "barrier": generate_barrier,
    "oxygen": generate_oxygen,
    "pumpless": generate_pumpless,
    "breathing": generate_breathing,
    "pulsatile": generate_pulsatile,
    "scaling": generate_scaling,
    "gradient": generate_gradient,
}

#: Benchmark gold files whose goals must never appear verbatim in training.
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
    v2_train = _load(V2 / "train.jsonl")
    v2_eval = _load(V2 / "eval.jsonl")

    # per-domain counts from v2 train keep the new-domain share stable
    counts: dict[str, int] = {}
    for r in v2_train:
        for b in set(r["raw"].keys()) & _NEW_DOMAIN_BLOCKS:
            counts[b] = counts.get(b, 0) + 1
    print("v2 new-domain train counts:", dict(sorted(counts.items())))

    # keep the working-domain rows byte-identical from v2
    kept_train = [r for r in v2_train if not (set(r["raw"].keys()) & _NEW_DOMAIN_BLOCKS)]
    kept_eval = [r for r in v2_eval if not (set(r["raw"].keys()) & _NEW_DOMAIN_BLOCKS)]
    print(f"kept v2 rows: train {len(kept_train)} / eval {len(kept_eval)}")

    # regenerate the seven new-domain rows (natural-register prose included)
    rng = random.Random(20260815)
    new_rows: list[dict] = []
    for dom in sorted(counts):
        gen = _GENERATORS[dom]
        new_rows += [gen(rng) for _ in range(counts[dom])]
    print(f"generated {len(new_rows)} new-domain rows ->", dict(sorted(counts.items())))

    # write_split does not shuffle (caller's job) and new_rows is grouped by
    # domain above — a domain-grouped split would dump one whole domain into the
    # eval half, so shuffle first.
    rng.shuffle(new_rows)
    # split the fresh rows with the same train/eval dedup as v2
    tmp = V3 / "_new"
    write_split(new_rows, tmp, split=0.9)
    new_train = _load(tmp / "train.jsonl")
    new_eval = _load(tmp / "eval.jsonl")

    train = kept_train + new_train
    eval_ = kept_eval + new_eval
    V3.mkdir(parents=True, exist_ok=True)
    _write(V3 / "train.jsonl", train)
    _write(V3 / "eval.jsonl", eval_)

    # gold pairs append at training time; copy the committed set
    import shutil
    shutil.copy2(V2 / "gold_pairs.jsonl", V3 / "gold_pairs.jsonl")

    # honesty gate: no training goal verbatim in any benchmark gold set
    bench_goals = _benchmark_goals()
    train_goals = {r["goal"] for r in train}
    overlap = train_goals & bench_goals
    print(f"train {len(train)} / eval {len(eval_)} (kept {len(kept_train)}/{len(kept_eval)} "
          f"+ new {len(new_train)}/{len(new_eval)})")
    print(f"benchmark-verbatim overlap: {len(overlap)}")
    if overlap:
        for g in list(overlap)[:3]:
            print("  OVERLAP:", g[:120])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
