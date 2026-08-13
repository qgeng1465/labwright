"""Analyse the ``tool_no_gate`` ablation vs ``labwright``.

The ablation answers the "circular verification" criticism: give the same model
the same calculator tools, but switch the verifier off (no ``verify_design`` in
``submit_design``, no verification discipline in the prompt), then score the
resulting plans post-hoc with the *identical* rules. If the verifier were
circular — just recomputing what the calculators already computed — turning it
off would change nothing. The gap between ``labwright`` and ``tool_no_gate`` is
the verifier's independent value.

``hallucination`` for ``tool_no_gate`` is the *would-have-been-caught* rate: the
post-hoc verifier flags plans the no-gate submit accepted. ``usable`` requires
both verifier-clean *and* target recovery, exactly as for ``labwright``.

Input files are the raw ``run_benchmark`` outputs (top-level aggregate dicts +
``per_entry``); they are used as-is, not passed through ``derive`` (which would
reshape them away).

Usage::

    python -m eval.analyze_nogate results/eval_nogate_flash.json results/eval_nogate_pro.json
"""

from __future__ import annotations

import json
import sys

from eval.ci import format_ci


def _per_system_rates(d: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    n = len(d["per_entry"])
    for name in ("labwright", "tool_no_gate"):
        m = d.get(name)
        if not isinstance(m, dict) or "usable_design_rate" not in m:
            continue
        k = int(round(m["usable_design_rate"] * n))
        out[name] = {
            "usable": m["usable_design_rate"],
            "usable_ci": format_ci(k, n),
            "self_consistent": m["self_consistent_rate"],
            "hallucination": m["hallucination_rate"],
        }
    return out


def _nogate_gate_caught(d: dict) -> dict[str, int]:
    """What the no-gate agent submitted and what the post-hoc verifier rejects."""
    submitted = dirty = no_plan = 0
    for e in d["per_entry"]:
        rec = e.get("tool_no_gate", {})
        if rec.get("plan"):
            submitted += 1
            if rec.get("hallucination_rate", 1.0) > 0.0:
                dirty += 1
        else:
            no_plan += 1
    return {"submitted": submitted, "verifier_dirty": dirty, "no_plan": no_plan}


def _tool_trace(d: dict, system: str) -> str:
    """Mean calculator calls per goal, comparing a system to itself across gates."""
    calls = [
        e.get(system, {}).get("tool_calls")
        for e in d["per_entry"]
        if e.get(system, {}).get("tool_calls") is not None
    ]
    refusals = [
        e.get(system, {}).get("prose_refusals", 0)
        for e in d["per_entry"]
        if system in e and "prose_refusals" in e.get(system, {})
    ]
    if not calls:
        return "  (no tool_calls recorded — pre-diagnostic checkpoint)"
    return (
        f"  {system:14s} mean {sum(calls) / len(calls):.1f} calculator calls / goal"
        f"  ({len(calls)} goals; total {sum(refusals)} prose refusals)"
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for path in argv:
        with open(path) as fh:
            d = json.load(fh)
        print(f"\n=== {path} ===")
        rates = _per_system_rates(d)
        for name in ("labwright", "tool_no_gate"):
            if name in rates:
                r = rates[name]
                print(
                    f"  {name:14s} usable {r['usable']:6.1%} {r['usable_ci']:>22s}"
                    f"   self-consistent {r['self_consistent']:6.1%}"
                    f"   hallucination {r['hallucination']:.3f}"
                )
        caught = _nogate_gate_caught(d)
        print(
            f"  tool_no_gate post-hoc: {caught['submitted']}/{len(d['per_entry'])} plans "
            f"submitted; verifier would have rejected {caught['verifier_dirty']} "
            f"({caught['verifier_dirty'] / max(1, caught['submitted']):.0%} of submitted); "
            f"{caught['no_plan']} no plan"
        )
        print(_tool_trace(d, "labwright"))
        print(_tool_trace(d, "tool_no_gate"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
