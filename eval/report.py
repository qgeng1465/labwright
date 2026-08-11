"""Render a human/paper-ready comparison from a benchmark results JSON.

Reads a ``results/eval_*.json`` produced by ``eval.run_benchmark`` and prints
the headline table — self-consistent rate, usable rate, parameter recovery and
hallucination rate, for bare-LLM vs Labwright. The metrics are *derived* from
the per-entry records, so the same raw JSON can be re-rendered (or re-styled)
without re-running the benchmark.

Usage::

    python -m eval.report results/eval_flash.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100.0 * x:.0f}%"


def derive(result: dict) -> dict:
    """Recompute headline metrics from per-entry records.

    - **self_consistent_rate** — fraction of entries with hallucination rate 0
      (every derived number the system reported agrees with its own inputs).
    - **usable_rate** — fraction of entries that are self-consistent *and*
      recover every gold target within ±5 %. A clean but off-target design is
      not usable.

    Applied uniformly to both systems: a Labwright entry counts as usable only
    when the plan exists, has zero verifier errors and matches the gold within
    tolerance (in practice machine precision).
    """
    out: dict = {}
    for system in ("bare", "labwright"):
        entries = [e[system] for e in result["per_entry"]]
        hall = [e["hallucination_rate"] for e in entries]
        self_consistent = [h == 0.0 for h in hall]
        usable = [
            (e["hallucination_rate"] == 0.0)
            and bool(e.get("recovery"))
            and all(err <= 0.05 for err in e["recovery"].values())
            for e in entries
        ]
        recovery: dict[str, list[float]] = {}
        for e in entries:
            for key, err in e["recovery"].items():
                recovery.setdefault(key, []).append(err)
        out[system] = {
            "self_consistent_rate": _mean([1.0 if s else 0.0 for s in self_consistent]),
            "usable_rate": _mean([1.0 if u else 0.0 for u in usable]),
            "hallucination_rate": _mean(hall),
            "recovery": {k: _mean(v) for k, v in recovery.items()},
        }
    # Bare answers that report geometry+flow and at least one derived number can
    # be cross-checked; the rest are unverifiable and scored hallucination 1.0.
    out["bare"]["verifiable_rate"] = _mean(
        [1.0 if e["bare"].get("verifiable") else 0.0 for e in result["per_entry"]]
    )
    out["n_gold"] = result["n_gold"]
    out["model"] = result.get("model")
    return out


def render(result: dict) -> str:
    d = derive(result)
    lines = []
    model = d.get("model") or "unknown"
    lines.append(f"Benchmark: bare-LLM vs Labwright on {d['n_gold']} gold entries (model {model})")
    lines.append("")
    header = f"{'metric':<26}{'bare-LLM':>14}{'Labwright':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    rows = [
        ("self-consistent rate", "self_consistent_rate"),
        ("usable rate", "usable_rate"),
        ("hallucination rate", "hallucination_rate"),
    ]
    for label, key in rows:
        b = _pct(d["bare"][key]) if key != "hallucination_rate" else f"{d['bare'][key]:.3f}"
        l = _pct(d["labwright"][key]) if key != "hallucination_rate" else f"{d['labwright'][key]:.3f}"
        lines.append(f"{label:<26}{b:>14}{l:>14}")
    lines.append(f"{'bare answers verifiable':<26}{_pct(d['bare']['verifiable_rate']):>14}")
    lines.append("")
    lines.append("Parameter recovery (mean relative error):")
    for key in sorted(set(d["bare"]["recovery"]) | set(d["labwright"]["recovery"])):
        b = d["bare"]["recovery"].get(key, float("nan"))
        l = d["labwright"]["recovery"].get(key, float("nan"))
        b = "n/a" if b != b else f"{b:.4g}"
        l = "n/a" if l != l else f"{l:.4g}"
        lines.append(f"  {key:<26}{b:>14}{l:>14}")
    lines.append("")
    lines.append("Per entry:")
    for e in result["per_entry"]:
        b = e["bare"]
        l = e["labwright"]
        verif = "ver" if b.get("verifiable") else "n/a"
        lines.append(
            f"  {e['id']:<26} bare {verif} h={b['hallucination_rate']:.2f} "
            f"valid={str(b['valid']).lower():<5} | lw h={l['hallucination_rate']:.2f} "
            f"valid={str(l['valid']).lower():<5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    path = Path(argv[0])
    with open(path) as fh:
        result = json.load(fh)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
