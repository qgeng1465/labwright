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
    systems = [s for s in ("bare", "soft_gate", "self_verify", "labwright", "finetuned")
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
        # New metrics from per-entry records produced by the current harness
        # (absent from older result JSONs — then simply omitted).
        if any("failure" in e for e in entries):
            from collections import Counter

            counts = Counter(e.get("failure") for e in entries)
            counts.pop(None, None)
            out[system]["failure_counts"] = dict(counts)
            out[system]["unit_misread_rate"] = _mean(
                [1.0 if e.get("unit_misread") else 0.0 for e in entries]
            )
            out[system]["target_selection_accuracy"] = _mean(
                [1.0 if e.get("target_selected") else 0.0 for e in entries]
            )
    out["n_gold"] = result["n_gold"]
    out["model"] = result.get("model")
    out["systems"] = systems
    # Cold vs prompt-backed blind split (only when the gold metadata is present).
    strengths = sorted({
        e["gold"]["blind_strength"] for e in per_entry
        if e.get("gold") and e["gold"].get("blind_strength")
    })
    if strengths:
        out["by_blind_strength"] = {}
        for s in strengths:
            subs = [e for e in per_entry if e.get("gold") and e["gold"].get("blind_strength") == s]
            out["by_blind_strength"][s] = {}
            for system in systems:
                rows = [e[system] for e in subs]
                out["by_blind_strength"][s][system] = {
                    "n": len(rows),
                    "usable_rate": _mean([1.0 if r["valid"] else 0.0 for r in rows]),
                    "hallucination_rate": _mean([r["hallucination_rate"] for r in rows]),
                }
    return out


def scirecipe_derive(report: dict) -> dict:
    """Derive the SciRecipe funnel + consistency metrics from an audit JSON.

    The denominator story is the whole point: of N protocol summaries, only a
    fraction carry numbers, of those a fraction route to a domain we can check,
    of those a fraction *state a derived number*, of those a fraction are
    checkable. ``n_ok`` counts only rows that asserted a derived number which
    re-computed within tolerance; rows with no derived claims are unverifiable
    (``no_derived_claims``), never vacuously "ok". The consistency rate is
    reported over the checkable set, never over the corpus.
    """
    verdicts = report.get("verdict_counts", {})
    n_ok = verdicts.get("ok", 0)
    n_review = verdicts.get("review_required", 0)
    n_checkable = n_ok + n_review
    n_claimed = report.get(
        "n_stated_derived",
        sum(1 for r in report.get("rows", []) if r.get("has_claims")),
    )
    return {
        "n_total": report["n_total"],
        "n_numeric": report["n_numeric"],
        "n_culture": report["n_culture"],
        "n_flow": report["n_flow"],
        "n_audited": report["n_audited"],
        "verdict_counts": verdicts,
        "n_ok": n_ok,
        "n_review_required": n_review,
        "n_unverifiable": verdicts.get("unverifiable", 0),
        "n_stated_derived": n_claimed,
        "numeric_pct": report["n_numeric"] / report["n_total"] if report["n_total"] else 0,
        "domain_rate": (report["n_culture"] + report["n_flow"]) / report["n_numeric"] if report["n_numeric"] else 0,
        "claimed_rate": n_claimed / report["n_audited"] if report["n_audited"] else 0,
        "checkable_rate": n_checkable / n_claimed if n_claimed else 0,
        "consistency_among_checkable": n_ok / n_checkable if n_checkable else 0,
    }


def render_scirecipe(report: dict) -> str:
    d = scirecipe_derive(report)
    pct = _pct
    lines = [
        f"SciRecipe reverse-verification audit ({d['n_total']} protocols)",
        f"  funnel: {d['n_numeric']} numeric ({pct(d['numeric_pct'])}) "
        f"-> {d['n_culture']} culture + {d['n_flow']} flow "
        f"({pct(d['domain_rate'])} of numeric) -> {d['n_audited']} audited "
        f"-> {d['n_stated_derived']} stated a derived number "
        f"({pct(d['claimed_rate'])} of audited) -> "
        f"{d['n_ok'] + d['n_review_required']} checkable ({pct(d['checkable_rate'])} of stated)",
        f"  consistency among checkable: {pct(d['consistency_among_checkable'])} "
        f"({d['n_ok']} ok / {d['n_review_required']} review_required) — "
        f"ok counts only rows that stated a derived number (no-derived-claims rows are unverifiable)",
        f"  verdicts: {d['verdict_counts']}",
        f"  runtime: {report.get('runtime_s')}s",
    ]
    return "\n".join(lines)


_LABELS = {
    "bare": "bare-LLM",
    "soft_gate": "soft-gate",
    "self_verify": "self-verify",
    "labwright": "Labwright",
    "finetuned": "finetuned-ext",
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
    # failure-reason breakdown / unit-misread / target-selection (new harness)
    if systems and "failure_counts" in d[systems[0]]:
        labels = {k: v for k, v in _LABELS.items() if k in systems}
        lines.append("")
        lines.append("Failure reasons (ok / silence / calculation_error / wrong_target):")
        for key in ("ok", "silence", "calculation_error", "wrong_target"):
            cells = []
            for s in systems:
                cells.append(f"{d[s]['failure_counts'].get(key, 0)}/{d['n_gold']}")
            lines.append(f"  {key:<26}" + "".join(f"{c:>14}" for c in cells))
        lines.append(f"{'unit-misread rate':<26}" + "".join(
            f"{_pct(d[s]['unit_misread_rate']):>14}" for s in systems))
        lines.append(f"{'target-selection accuracy':<26}" + "".join(
            f"{_pct(d[s]['target_selection_accuracy']):>14}" for s in systems))
    # cold vs prompt-backed blind split
    if d.get("by_blind_strength"):
        lines.append("")
        lines.append("Blind split by hint strength (usable rate / n):")
        for strength, subs in d["by_blind_strength"].items():
            cells = []
            for s in systems:
                sub = subs.get(s)
                cells.append(f"{_pct(sub['usable_rate'])}/{sub['n']}")
            lines.append(f"  {strength:<26}" + "".join(f"{c:>14}" for c in cells))
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
    if "verdict_counts" in result and "n_numeric" in result:
        print(render_scirecipe(result))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
