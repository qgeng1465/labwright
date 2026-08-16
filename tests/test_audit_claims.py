"""The headline-number reproducibility guard runs as part of the suite.

Every number in README.md / eval/README.md is recomputed from the committed
``results/*.json`` and asserted equal to the displayed value (see
``eval/audit_claims.py``). Running it here means a regenerated JSON that is not
synced to the docs — or a doc edit that is not synced to the JSON — fails CI.
"""

from eval import audit_claims


def test_audit_claims_all_pass():
    assert audit_claims.main() == 0
