"""Tests for the physiology registry — the single source of cell reference data.

The registry is the contract between the safety layer, the agent system prompt,
the tools and the (future) O2/barrier calculators. These tests pin the lookup
behaviour and the key defensible values so a later edit can't silently change
what the agent is told.
"""

from labwright.physiology import (
    PROFILES,
    biosafety_for,
    lookup_cell,
    physiology_anchor_text,
)
from labwright.tools import REGISTRY


def test_lookup_exact_key_and_alias():
    assert lookup_cell("HepG2").key == "hepg2"
    assert lookup_cell("hepg2 cells").key == "hepg2"
    assert lookup_cell("PHH").key == "phh"
    assert lookup_cell("Caco2").key == "caco-2"
    assert lookup_cell("hCMEC/D3").key == "hcmec-d3"


def test_lookup_fuzzy_substring():
    # "HepG2 spheroids" must resolve to HepG2, "primary human hepatocytes" to PHH.
    assert lookup_cell("HepG2 spheroids").key == "hepg2"
    assert lookup_cell("primary human hepatocytes").key == "phh"
    assert lookup_cell("   primary mouse hepatocytes  ").key == "primary-mouse-hepatocytes"


def test_lookup_unknown_returns_none():
    assert lookup_cell("my-custom-line-xyz") is None
    assert lookup_cell(None) is None
    assert lookup_cell("") is None


def test_biosafety_for_uses_registry():
    assert biosafety_for("HeLa") == (2, "HeLa is classed BSL-2 at ATCC")
    assert biosafety_for("HepG2")[0] == 1
    assert biosafety_for("PHH")[0] == 2
    assert biosafety_for("primary mouse hepatocytes")[0] == 2
    # unregistered "primary" material conservatively BSL-2
    assert biosafety_for("primary rat neurons")[0] == 2
    # unknown non-primary falls back to BSL-1
    assert biosafety_for("made-up-line") == (1, None)


def test_hepg2_doubling_is_the_corrected_catalog_value():
    """The registry must not repeat the old ~30-40 h prose claim (DSMZ/ENCODE say 48-60 h)."""
    prof = lookup_cell("HepG2")
    lo, hi = prof.doubling_time_h
    assert lo >= 40, "old 30-40 h figure contradicted by DSMZ ACC-180 / ENCODE4 SOP"
    assert hi >= 55
    assert "30-40" not in physiology_anchor_text()


def test_registry_covers_gold_cell_types():
    for name in ("HepG2", "primary human hepatocytes", "primary mouse hepatocytes",
                 "hepatocyte spheroid", "Caco-2"):
        assert lookup_cell(name) is not None, name


def test_cell_physiology_tool_registered():
    assert "cell_physiology" in REGISTRY
    out = REGISTRY["cell_physiology"].func("HepG2")
    assert out["resolved"] == "hepg2"
    assert out["doubling_time_h"] == [48, 60]
    assert out["shear_range_pa"] == [0.05, 0.15]
    missing = REGISTRY["cell_physiology"].func("unknown-line-xyz")
    assert missing["resolved"] is None


def test_anchor_text_lists_every_profile():
    rendered = physiology_anchor_text()
    for key in PROFILES:
        assert key in rendered, key
