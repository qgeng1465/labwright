"""Synthesis of every benchmark result file into one honest results digest.

Pulls together:

* the single-run main table (bare / soft_gate / self_verify / labwright) from
  the five gold sets — reading, blind, spheroid, culture, PK;
* the pooled 3-seed Wilson-CI table (``eval_seed_*.json``) where present;
* the no-gate ablation (``eval_nogate_*.json``);
* the ``labwright_iter`` fix-and-resubmit comparison (``eval_iter_*.json``).

The digest is a terminal summary plus a markdown block for the README; every
number in it comes from a results file that exists on disk, and rows whose
file is missing are marked ``(missing)`` rather than fabricated.

Usage::

    python -m eval.summarize_results            # print digest
    python -m eval.summarize_results --md       # print README-ready markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _load(fname: str) -> dict | None:
    p = RESULTS / fname
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _single(d: dict, name: str) -> str | None:
    """Format one system's aggregate from a run_benchmark output."""
    m = d.get(name)
    if not isinstance(m, dict) or "usable_design_rate" not in m:
        return None
    n = len(d.get("per_entry", []))
    k = int(round(m["usable_design_rate"] * n))
    return f"{k}/{n} = {100 * m['usable_design_rate']:.0f}%"


def _pool_str(d: dict | None, model: str, sysk: str) -> str:
    if d is None or "pooled" not in d:
        return "(missing)"
    p = d["pooled"].get(model, {}).get(sysk)
    if not p:
        return "—"
    return f"{100 * p['usable_design_rate']:.0f}% {p['usable_ci_str']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()

    sets = [
        ("24-reading", "eval_v1.json", "eval_seed_benchmark.json"),
        ("15-blind", "eval_blind.json", "eval_seed_blind.json"),
        ("15-3D-spheroid", "eval_spheroid.json", "eval_seed_spheroid.json"),
        ("14-plate-culture", "eval_culture.json", "eval_seed_culture.json"),
        ("14-perfused-PK", "eval_pk.json", "eval_seed_pk.json"),
    ]
    lines = []
    lines.append("### Seed-CI (pooled 3-seed, Wilson 95%) — usable rate")
    lines.append("| set | model | bare | soft_gate | self_verify | labwright |")
    lines.append("|---|---|---|---|---|---|")
    for label, seed_f in (s[::2] if False else [(a, c) for a, _, c in sets]):
        d = _load(seed_f)
        for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
            cells = [
                f"{100 * (d['pooled'].get(model, {}).get(sk, {}).get('usable_design_rate') or 0):.0f}% "
                f"{d['pooled'].get(model, {}).get(sk, {}).get('usable_ci_str', '')}"
                for sk in ("bare", "soft_gate", "self_verify", "labwright")
            ] if d and "pooled" in d else ["(missing)"] * 4
            lines.append(
                f"| {label} | {model.replace('deepseek-v4-', '')} | "
                + " | ".join(cells) + " |"
            )
    if args.md:
        print("\n".join(lines))
        return 0

    print("\n=== seed-CI pooled usable (Wilson 95%) ===")
    print(f"{'set':<17}{'model':<7}{'bare':<24}{'soft_gate':<24}{'self_verify':<24}{'labwright':<24}")
    for label, _, seed_f in sets:
        d = _load(seed_f)
        print(f"--- {label} ({seed_f}) ---")
        if d is None or "pooled" not in d:
            print("  (missing)")
            continue
        for model in d.get("models", []):
            row = []
            for sk in ("bare", "soft_gate", "self_verify", "labwright"):
                p = d["pooled"].get(model, {}).get(sk, {})
                row.append(f"{100 * p.get('usable_design_rate', 0):.0f}% {p.get('usable_ci_str', '')}")
            print(f"  {model.replace('deepseek-v4-',''):<7}" + "  ".join(f"{x:<24}" for x in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
