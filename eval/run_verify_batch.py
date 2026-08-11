"""Run the batch literature reverse-verification over a set of published protocols.

Each JSON file in ``eval/published_protocols/`` describes a reported protocol:
the paper's channel geometry, flow, and the derived numbers it claims, plus a
mandatory ``reference`` and a ``kind`` (``published`` or ``control``). For every
protocol the checker recomputes the claimed numbers from the paper's *own*
inputs and classifies the protocol as:

* ``ok`` — every claim is internally consistent with the reported inputs;
* ``review_required`` — at least one claimed number does not follow;
* ``unverifiable`` — the paper does not report enough inputs to recompute.

Real published protocols with full reported inputs are expected to verify
(positive controls). The ``control`` entries are explicitly labelled synthetic
tests that demonstrate the flagging machinery on constructed errors; they are
*not* presented as published findings.

Usage::

    python -m eval.run_verify_batch
    python -m eval.run_verify_batch --out results/eval_verify_batch.json

Exit code is nonzero if any protocol's outcome differs from its ``expected``
label, so the batch doubles as a regression test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from labwright.published import verify_published_protocol

_HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOLS_DIR = os.path.join(_HERE, "published_protocols")
DEFAULT_OUT = os.path.join(_HERE, "..", "results", "eval_verify_batch.json")


def classify(result: dict) -> str:
    """Map a verify result to ok / review_required / unverifiable."""
    if result["status"] == "validation_error":
        return "unverifiable"
    return result["status"]  # ok | review_required


def run_batch(protocols_dir: str = PROTOCOLS_DIR) -> dict:
    records = []
    for path in sorted(os.listdir(protocols_dir)):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(protocols_dir, path)) as fh:
            entry = json.load(fh)
        result = verify_published_protocol(
            chip=entry.get("chip", {}),
            flow=entry.get("flow", {}),
            claimed=entry.get("claimed", {}),
            reference=entry.get("reference", ""),
        )
        records.append(
            {
                "id": entry.get("id", path),
                "kind": entry.get("kind", "unknown"),
                "reference": entry.get("reference", ""),
                "note": entry.get("note", ""),
                "expected": entry.get("expected"),
                "actual": classify(result),
                "n_discrepancies": result.get("n_discrepancies", 0),
                "discrepancy_fields": [
                    c["field"] for c in result.get("checks", []) if c["verdict"] == "discrepancy"
                ],
                "detail": result,
            }
        )
    return {
        "n_protocols": len(records),
        "n_ok": sum(1 for r in records if r["actual"] == "ok"),
        "n_review_required": sum(1 for r in records if r["actual"] == "review_required"),
        "n_unverifiable": sum(1 for r in records if r["actual"] == "unverifiable"),
        "protocols": records,
    }


def render(batch: dict) -> str:
    lines = [
        f"Published-protocol reverse verification ({batch['n_protocols']} protocols)",
        f"  ok: {batch['n_ok']}   review_required: {batch['n_review_required']}   "
        f"unverifiable: {batch['n_unverifiable']}",
        "",
        f"{'protocol':<26}{'kind':<10}{'expected':<14}{'actual':<16}flags",
        "-" * 84,
    ]
    for r in batch["protocols"]:
        flags = ",".join(r["discrepancy_fields"]) if r["discrepancy_fields"] else "-"
        passmark = "OK" if r["expected"] is None or r["expected"] == r["actual"] else "MISMATCH"
        lines.append(
            f"{r['id']:<26}{r['kind']:<10}{str(r['expected']):<14}{r['actual']:<16}"
            f"{flags:<10}{passmark}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON output path")
    parser.add_argument("--protocols-dir", default=PROTOCOLS_DIR)
    args = parser.parse_args(argv)

    batch = run_batch(args.protocols_dir)
    print(render(batch))

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(batch, fh, indent=2, default=str)
        print(f"\nsaved -> {out_path}")

    mismatches = [
        r["id"] for r in batch["protocols"]
        if r["expected"] is not None and r["expected"] != r["actual"]
    ]
    if mismatches:
        print(f"\nFAIL: expected/actual mismatch on: {', '.join(mismatches)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
