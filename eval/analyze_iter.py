"""Analyse the ``labwright_iter`` fix-and-resubmit comparison.

The paper's "agent attempt": when the verifier returns ``review_required``, the
iterating agent reads the verification report, fixes ONLY the flagged raw
inputs, and resubmits (up to ``max_submission_attempts``). ``labwright``
(first-submit) treats the first ``review_required`` as terminal and is scored on
the dirty plan; ``labwright_iter`` keeps the honest verdict but gets to fix it.

For each run file (paired ``labwright`` / ``labwright_iter`` columns) this
reports, per system:

* usable rate (verifier-clean AND all gold targets recovered);
* self-consistent rate (``hallucination_rate == 0``);
* for iter: the ``fix_rounds`` distribution and how many entries ended on a
  recovered ``ok`` after a ``review_required`` — i.e. the *recoverable* cases.

An honest reading: on entries where the verifier never fires, ``labwright_iter``
cannot differ from ``labwright`` (same calculator discipline, same prompt). The
system's value is concentrated exactly where the verifier does fire — those are
the plans a first-submit pipeline would have lost.

Usage::

    python -m eval.analyze_iter results/eval_iter_blind_flash.json
"""

from __future__ import annotations

import json
import sys

from eval.ci import format_ci


def _per_system(d: dict, name: str) -> dict | None:
    m = d.get(name)
    if not isinstance(m, dict) or "usable_design_rate" not in m:
        return None
    n = len(d["per_entry"])
    k = int(round(m["usable_design_rate"] * n))
    kc = int(round(m["self_consistent_rate"] * n))
    return {
        "usable": m["usable_design_rate"],
        "usable_ci": format_ci(k, n),
        "self_consistent": m["self_consistent_rate"],
        "self_consistent_ci": format_ci(kc, n),
        "hallucination": m["hallucination_rate"],
        "n": n,
    }


def _iter_verdicts(d: dict) -> dict:
    """Where did the verifier fire, and did the iter loop recover the entry?"""
    fired = recovered = exhausted = 0
    rounds = []
    first_review: dict = {"submitted_dirty_as_ok": 0}
    for e in d["per_entry"]:
        rec = e.get("labwright_iter", {})
        fr = rec.get("fix_rounds", 0)
        rounds.append(fr)
        if fr > 0:
            fired += 1
            # Recovered = the final plan is verifier-clean after >=1 review.
            if rec.get("hallucination_rate", 1.0) == 0.0 and rec.get("plan"):
                recovered += 1
            else:
                exhausted += 1
    return {
        "verifier_fired": fired,
        "recovered_to_ok": recovered,
        "exhausted_budget": exhausted,
        "mean_fix_rounds": (sum(rounds) / len(rounds)) if rounds else 0.0,
        "max_fix_rounds": max(rounds) if rounds else 0,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for path in argv:
        with open(path) as fh:
            d = json.load(fh)
        print(f"\n=== {path} ===")
        for name in ("labwright", "labwright_iter"):
            r = _per_system(d, name)
            if r is None:
                continue
            print(
                f"  {name:16s} usable {r['usable']:6.1%} {r['usable_ci']:>22s}"
                f"   selfcons {r['self_consistent']:6.1%} {r['self_consistent_ci']:>18s}"
                f"   hallu {r['hallucination']:.3f}   n={r['n']}"
            )
        v = _iter_verdicts(d)
        print(
            f"  verifier fired on {v['verifier_fired']} entries; "
            f"{v['recovered_to_ok']} recovered to ok, {v['exhausted_budget']} exhausted the "
            f"attempt budget; mean fix_rounds {v['mean_fix_rounds']:.2f} "
            f"(max {v['max_fix_rounds']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
