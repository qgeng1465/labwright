"""Regression test for the batch published-protocol reverse verification.

Every protocol in ``eval/published_protocols/`` declares an ``expected``
outcome (ok / review_required / unverifiable). This test runs the batch and
asserts each protocol lands where it should. The ``control`` entries are
explicitly synthetic and labelled as such; the ``published`` entries are real
papers whose reported inputs are on record.
"""

from __future__ import annotations

import json
import os

from eval.run_verify_batch import PROTOCOLS_DIR, run_batch


def test_batch_outcomes_match_expectations():
    batch = run_batch(PROTOCOLS_DIR)
    assert batch["n_protocols"] >= 5
    for record in batch["protocols"]:
        assert record["actual"] == record["expected"], (
            f"{record['id']}: expected {record['expected']}, got {record['actual']}"
        )
    # The synthetic controls must include at least one discrepancy flag.
    flagged = [r for r in batch["protocols"] if r["actual"] == "review_required"]
    assert flagged, "batch must contain at least one discrepancy case"


def test_every_protocol_has_a_reference():
    for path in sorted(os.listdir(PROTOCOLS_DIR)):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(PROTOCOLS_DIR, path)) as fh:
            entry = json.load(fh)
        assert entry.get("reference"), f"{path} is missing its provenance reference"
        assert entry.get("kind") in {"published", "control"}
