"""CPU-only tests for the extractor package (no GPU, no model download)."""

from __future__ import annotations

import json

import pytest

from labwright.design import DesignInput, build_design
from labwright.extract.data import encode_example, raw_to_json
from labwright.extract.eval import field_errors, errors_all_within, score_batch, score_one
from labwright.extract.gold_pairs import gold_pairs
from labwright.extract.pipeline import parse_json
from labwright.extract.synthetic import generate
from labwright.verify.checker import has_errors, verify_design


class _StubTokenizer:
    """Minimal tokenizer stand-in: one token per character, ChatML rendering."""

    def apply_chat_template(self, messages, tokenize=False):
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        assert tokenize is False
        return rendered

    def __call__(self, text, return_offsets_mapping=False, max_length=None, truncation=False):
        chars = list(text)
        if truncation and max_length:
            chars = chars[:max_length]
        ids = [ord(c) for c in chars]
        offsets = [(i, i + 1) for i in range(len(ids))]
        return {"input_ids": ids, "offset_mapping": offsets}


# ---------------------------------------------------------------------------
# Loss masking
# ---------------------------------------------------------------------------


def test_mask_covers_only_assistant_json():
    tok = _StubTokenizer()
    goal = "Seed HepG2 in a 96-well plate at 10000 cells/cm2 across 12 wells."
    raw = {"culture": {"plate_format": "96-well", "wells": 12, "cell_type": "HepG2",
                       "seeding_density_cells_cm2": 10000.0}}
    enc = encode_example(tok, goal, raw_to_json(raw))
    assert enc is not None
    ids, labels = enc["input_ids"], enc["labels"]
    assert len(ids) == len(labels)
    # exactly the JSON (as chars) plus the closing <|im_end|> tag is unmasked
    masked = "".join(chr(i) for i, l in zip(ids, labels) if l != -100)
    assert masked == raw_to_json(raw) + "<|im_end|>\n"
    # everything before the assistant turn (system prompt + user goal) is masked,
    # and the assistant JSON itself is never masked
    prefix = "".join(chr(i) for i, l in zip(ids, labels) if l == -100)
    assert goal in prefix
    assert raw_to_json(raw) not in prefix


def test_encode_skips_truncated_assistant():
    tok = _StubTokenizer()
    raw = {"culture": {"plate_format": "96-well", "wells": 1, "cell_type": "HepG2",
                       "seeding_density_cells_cm2": 10000.0}}
    # max_len smaller than the whole chat → assistant content truncated away
    enc = encode_example(tok, "a" * 5000, raw_to_json(raw), max_len=64)
    assert enc is None


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------


def test_synthetic_is_deterministic_and_consistent():
    rows = generate(20, 20, seed=7)
    again = generate(20, 20, seed=7)
    assert rows == again
    for row in rows:
        inp = DesignInput(goal=row["goal"], rationale="test", **row["raw"])
        assert not has_errors(verify_design(build_design(inp)))


def test_synthetic_optional_fields_are_coupled_to_prose():
    """A raw optional field must be stated in the prose (never underdetermined)."""
    rows = generate(60, 60, seed=3)
    for row in rows:
        goal = row["goal"].lower()
        if row["domain"] == "flow":
            cells = row["raw"]["cells"]
            if "doubling_time_h" in cells:
                assert "double" in goal
            assert "flow rate" in goal
        else:
            culture = row["raw"]["culture"]
            assert culture["plate_format"] in row["goal"]
            if "doubling_time_h" in culture:
                assert "double" in goal
            if "viability_pct" in culture:
                assert "viability" in goal


# ---------------------------------------------------------------------------
# Gold pairs
# ---------------------------------------------------------------------------


def test_gold_pairs_all_consistent():
    pairs, skipped = gold_pairs()
    assert len(pairs) == 32
    assert sorted(skipped) == ["plate-hemocytometer-seed-96well", "plate-thaw-viability-6well"]
    for p in pairs:
        inp = DesignInput(goal=p["goal"], rationale="gold", **p["raw"])
        assert not has_errors(verify_design(build_design(inp))), p["gold"]


def test_gold_pairs_reproduce_expected_numbers():
    pairs, _ = gold_pairs()
    by_gold = {p["gold"]: p for p in pairs}
    # plate-6well-phh-seed → seed_per_well 1.44e6, total medium 16.2
    inp = DesignInput(goal="x", rationale="x", **by_gold["plate-6well-phh-seed"]["raw"])
    c = build_design(inp).culture
    assert c.seed_per_well == pytest.approx(1.44e6, rel=1e-6)
    assert c.total_medium_ml == pytest.approx(16.2, rel=1e-6)
    # liver-sinusoid-shear canonical → 0.05 Pa
    inp = DesignInput(goal="x", rationale="x", **by_gold["liver-sinusoid-shear"]["raw"])
    assert build_design(inp).derived.shear_pa == pytest.approx(0.05, rel=1e-4)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_json_handles_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here it is: {"a": 1, "b": {"c": 2}} trailing') == {"a": 1, "b": {"c": 2}}
    assert parse_json("") is None
    assert parse_json("no json here") is None


# ---------------------------------------------------------------------------
# Eval scoring
# ---------------------------------------------------------------------------


def test_field_errors_and_recovery():
    gold = {"chip": {"width_um": 400, "height_um": 100}}
    assert field_errors({"chip": {"width_um": 420, "height_um": 100}}, gold) == {
        "chip.width_um": 0.05, "chip.height_um": 0.0}
    assert errors_all_within({"chip.width_um": 0.05, "chip.height_um": 0.0})
    assert not errors_all_within({"chip.width_um": 0.2, "chip.height_um": 0.0})
    assert not errors_all_within({"chip.width_um": None})
    # missing key
    assert field_errors({"chip": {"width_um": 400}}, gold)["chip.height_um"] is None


def test_score_batch_with_stub_extractor():
    rows = generate(6, 4, seed=11)
    blind = [{"goal": r["goal"], "expected": {"shear_pa": 0.05}} for r in rows[:2]]

    def perfect(goal: str):
        for r in rows:
            if r["goal"] == goal:
                return r["raw"]
        return None

    rep = score_batch(perfect, rows, blind)
    assert rep["json_parse_rate"] == 1.0
    assert rep["consistency_rate"] == 1.0
    assert rep["field_recovery_ok_rate"] == 1.0

    def none_extract(_goal):
        return None

    rep = score_batch(none_extract, rows, blind)
    assert rep["json_parse_rate"] == 0.0
