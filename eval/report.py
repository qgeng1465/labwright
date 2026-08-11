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

    Applied uniformly to every system present in the records: bare, soft-gate,
    self-verify and Labwright. A Labwright entry counts as usable only when the
    plan exists, has zero verifier errors and matches the gold within tolerance
    (in practice machine precision); the LLM-memory systems are usable only when
    their reported numbers are self-consistent *and* recover the gold target.
    """
    per_entry = result["per_entry"]
    systems = [s for s in ("bare", "soft_gate", "self_verify", "labwright")
               if per_entry and s in per_entry[0]]
    out: dict = {}
    for system in systems:
        entries = [e[system] for e in per_entry]
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
        # Systems that report numbers from memory (bare, soft-gate, self-verify)
        # carry a verifiable_rate: answers that report geometry+flow and at least
        # one derived number can be cross-checked; the rest are unverifiable and
        # scored hallucination 1.0.
        if system in ("bare", "soft_gate", "self_verify"):
            out[system]["verifiable_rate"] = _mean(
                [1.0 if e[system].get("verifiable") else 0.0 for e in per_entry]
            )
    out["n_gold"] = result["n_gold"]
    out["model"] = result.get("model")
    out["systems"] = systems
    return out


_LABELS = {
    "bare": "bare-LLM",
    "soft_gate": "soft-gate",
    "self_verify": "self-verify",
    "labwright": "Labwright",
}


def render(result: dict) -> str:
    d = derive(result)
    systems = d["systems"]
    lines = []
    model = d.get("model") or "unknown"
    lines.append(
        f"Benchmark on {d['n_gold']} gold entries (model {model}): "
        + " vs ".join(_LABELS[s] for s in systems)
    )
    lines.append("")
    header = f"{'metric':<26}" + "".join(f"{_LABELS[s]:>14}" for s in systems)
    lines.append(header)
    lines.append("-" * len(header))
    rows = [
        ("self-consistent rate", "self_consistent_rate"),
        ("usable rate", "usable_rate"),
        ("hallucination rate", "hallucination_rate"),
    ]
    for label, key in rows:
        cells = []
        for s in systems:
            v = d[s][key]
            cells.append(_pct(v) if key != "hallucination_rate" else f"{v:.3f}")
        lines.append(f"{label:<26}" + "".join(f"{c:>14}" for c in cells))
    if "bare" in systems:
        lines.append(f"{'bare answers verifiable':<26}" + "".join(
            f"{_pct(d[s]['verifiable_rate']):>14}" for s in ("bare",) if s in systems))
    lines.append("")
    lines.append("Parameter recovery (mean relative error):")
    all_keys = set()
    for s in systems:
        all_keys |= set(d[s]["recovery"])
    for key in sorted(all_keys):
        cells = []
        for s in systems:
            v = d[s]["recovery"].get(key, float("nan"))
            cells.append("n/a" if v != v else f"{v:.4g}")
        lines.append(f"  {key:<26}" + "".join(f"{c:>14}" for c in cells))
    lines.append("")
    lines.append("Per entry:")
    for e in result["per_entry"]:
        parts = []
        for s in systems:
            rec = e[s]
            tag = "ver" if rec.get("verifiable") else "n/a"
            parts.append(f"{_LABELS[s]} {tag} h={rec['hallucination_rate']:.2f} "
                         f"valid={str(rec['valid']).lower():<5}")
        lines.append(f"  {e['id']:<26} " + " | ".join(parts))
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
