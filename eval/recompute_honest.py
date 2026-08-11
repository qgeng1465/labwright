"""Recompute bare-LLM metrics honestly from a committed results JSON.

Why this exists
---------------
The first committed benchmark runs scored a bare answer as *self-consistent*
(hallucination 0.0) whenever the model reported chip geometry + flow but *no
derived flow numbers at all* — "nothing claimed → nothing to contradict". That
was the metric's documented intent ("recovery catches silence"), but it inflated
the bare self-consistent rate: in both committed runs, 10/24 bare entries were
scored consistent purely because their numbers could not be cross-checked.

Labwright's own convention scores a run that never submits a plan as
hallucination 1.0 ("numbers you type are not trusted"). This script applies the
same convention to bare: a bare answer whose numbers cannot be re-derived from
its own reported inputs is *unverifiable*, and unverifiable = 1.0.

The script re-derives every bare field from the **stored per-entry `reported`
records** — no API calls, fully auditable. Labwright fields are untouched (they
were already honest). Run it, then commit the corrected JSONs.

Usage::

    python -m eval.recompute_honest results/eval_flash.json
    python -m eval.recompute_honest results/eval_pro.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.benchmark import bare_checkable, bare_hallucination  # noqa: E402


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def recompute(result: dict) -> dict:
    """Update the bare metrics in-place from the per-entry reported records."""
    for entry in result["per_entry"]:
        reported = entry["bare"]["reported"]
        h = bare_hallucination(reported)
        entry["bare"]["verifiable"] = bare_checkable(reported)
        entry["bare"]["hallucination_rate"] = round(h, 6)
        entry["bare"]["valid"] = h == 0.0 and all(
            err <= 0.05 for err in entry["bare"]["recovery"].values()
        )

    bare = result["bare"]
    rates = [e["bare"]["hallucination_rate"] for e in result["per_entry"]]
    bare["hallucination_rate"] = _mean(rates)
    bare["self_consistent_rate"] = _mean([1.0 if r == 0.0 else 0.0 for r in rates])
    bare["usable_design_rate"] = _mean(
        [1.0 if e["bare"]["valid"] else 0.0 for e in result["per_entry"]]
    )
    # recovery means are already aggregated from the same per-entry values.
    return result


def main(argv: list[str]) -> int:
    for path in argv:
        p = Path(path)
        with open(p) as fh:
            result = json.load(fh)
        recompute(result)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        b = result["bare"]
        print(
            f"{p}: bare self-consistent={100*b['self_consistent_rate']:.1f}%  "
            f"usable={100*b['usable_design_rate']:.1f}%  hallucination={b['hallucination_rate']:.3f}  "
            f"verifiable={sum(e['bare']['verifiable'] for e in result['per_entry'])}/{result['n_gold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
