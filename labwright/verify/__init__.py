"""Verification — Labwright's anti-hallucination core.

Every number the agent emits is recomputed here from first principles (the
raw inputs the agent provided) using :mod:`labwright.calc`. A number that
cannot be reproduced is rejected, not displayed.
"""

from labwright.verify.checker import Issue, verify_design

__all__ = ["Issue", "verify_design"]
