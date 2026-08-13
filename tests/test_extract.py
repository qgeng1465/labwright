"""CPU-only tests for the extractor package (no GPU, no model download)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from labwright.design import DesignInput, _reject_derived_fields, build_design
from labwright.extract.data import encode_example, raw_to_json
from labwright.extract.eval import build_from_raw, field_errors, errors_all_within, score_batch, score_one
from labwright.extract.gold_pairs import gold_pairs
from labwright.extract.pipeline import Extractor, configure_tokenizer, parse_json
from labwright.extract.synthetic import generate
from labwright.verify.checker import has_errors, verify_design


class _StubTokenizer:
    """Minimal tokenizer stand-in: one token per character, ChatML rendering.

    Mimics the Qwen2.5 tokenizer's dangerous default: right padding and no
    pad token (pad falls back to eos), which is what corrupts batch decode
    for a decoder-only model.
    """

    padding_side = "right"
    pad_token = None
    eos_token = "<|endoftext|>"

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
# Decoder-only tokenizer configuration (regression: the 19.9 % parse rate)
# ---------------------------------------------------------------------------


def test_configure_tokenizer_forces_left_padding():
    """Right padding corrupts decoder-only batch decode — must be left."""
    tok = _StubTokenizer()
    assert tok.padding_side == "right"  # the dangerous default we're guarding
    configure_tokenizer(tok)
    assert tok.padding_side == "left"


def test_configure_tokenizer_falls_back_to_eos_pad():
    tok = _StubTokenizer()
    assert tok.pad_token is None
    configure_tokenizer(tok)
    assert tok.pad_token == "<|endoftext|>"
    # and it never overwrites an explicit pad token
    tok = _StubTokenizer()
    tok.pad_token = "<|pad|>"
    configure_tokenizer(tok)
    assert tok.pad_token == "<|pad|>"


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


def test_partial_cells_block_is_schema_error_not_crash():
    """A cells block missing seeding density/area must not KeyError in build_design."""
    goal = "Seed cells onto a chip."
    # cells present but partial -> schema_error, parsed counts as not-ok
    rec = score_one(goal, {"cells": {"cell_type": "HepG2"}}, None)
    assert rec["parsed"] is True
    assert rec["schema_ok"] is False
    assert rec["consistent"] is False
    # build_from_raw surfaces the schema_error reason
    _plan, _issues, err = build_from_raw(goal, {"cells": {"cell_type": "HepG2"}})
    assert err == "schema_error"
    # and a blind-gold-shaped flow row with no cells works fine
    plan, issues, err = build_from_raw(
        "Design a 400 um channel at 0.05 Pa.",
        {"chip": {"width_um": 400, "height_um": 100, "length_mm": 20},
         "flow": {"flow_rate_uLmin": 2, "viscosity_pas": 0.001}},
    )
    assert err is None and plan is not None


# ---------------------------------------------------------------------------
# The extractor gate: raw inputs cross the same derived-field gate and schema
# strictness as the agent's submit_design
# ---------------------------------------------------------------------------


def test_gate_rejects_top_level_derived_field():
    with pytest.raises(ValueError, match="derived field"):
        _reject_derived_fields({"shear_pa": 0.05})


def test_gate_rejects_nested_derived_field():
    with pytest.raises(ValueError, match="culture.seed_per_well"):
        _reject_derived_fields({"culture": {"plate_format": "96-well",
                                            "seeding_density_cells_cm2": 1e5,
                                            "seed_per_well": 3200}})


def test_design_input_rejects_unknown_top_level_key():
    # A hallucinated field (e.g. cell_count) is a loud schema error, not silently dropped.
    with pytest.raises(ValidationError, match="Extra inputs"):
        DesignInput(goal="g", rationale="r", cell_count=5000)


def test_block_models_reject_unknown_nested_key():
    # DesignInput blocks are dict[str, Any]; the real strictness sits on the
    # block models built in build_design.
    from labwright.schema.design import CellPlan, ChipGeometry

    with pytest.raises(ValidationError):
        ChipGeometry(width_um=400, height_um=100, length_mm=20, width_mm=0.4)
    with pytest.raises(ValidationError):
        CellPlan(cell_type="HepG2", seeding_density_cells_cm2=1e5,
                 culture_area_cm2=0.08, seed_count=8000, viability=90)


def _mock_extractor(raw: dict | None):
    """An Extractor that skips model loading and returns ``raw`` from extract()."""
    ext = object.__new__(Extractor)
    ext.extract = lambda goal, max_new_tokens=384: raw
    return ext


def test_extract_plan_rejects_derived_field_via_gate():
    ext = _mock_extractor({"cells": {"cell_type": "HepG2", "seed_count": 8000,
                                     "seeding_density_cells_cm2": 1e5,
                                     "culture_area_cm2": 0.08}})
    plan, _issues, error = ext.extract_plan("goal")
    assert plan is None
    assert error is not None and error.startswith("derived_field_rejected")


def test_extract_plan_rejects_unknown_key():
    ext = _mock_extractor({"chip": {"width_um": 400, "height_um": 100,
                                    "length_mm": 20, "width_mm": 0.4},
                           "flow": {"flow_rate_uLmin": 10, "viscosity_pas": 1e-3}})
    plan, _issues, error = ext.extract_plan("goal")
    assert plan is None
    assert error is not None and error.startswith("schema_error")


def test_extract_plan_catches_duplicate_keyword_typeerror():
    # The extractor emitting a "goal" key collides with the injected goal
    # keyword — must be a schema_error, never a crash.
    ext = _mock_extractor({"goal": "the model tried to restate the goal",
                           "chip": {"width_um": 400, "height_um": 100, "length_mm": 20}})
    plan, _issues, error = ext.extract_plan("original goal")
    assert plan is None
    assert error is not None and error.startswith("schema_error")


def test_extract_plan_happy_path():
    ext = _mock_extractor({"chip": {"width_um": 400, "height_um": 100, "length_mm": 20},
                           "flow": {"flow_rate_uLmin": 10, "viscosity_pas": 1e-3}})
    plan, issues, error = ext.extract_plan("goal")
    assert error is None
    assert plan is not None
    assert plan.derived is not None
    assert not has_errors(verify_design(plan))


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
