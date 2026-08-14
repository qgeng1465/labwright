"""CPU-only tests for the extractor package (no GPU, no model download)."""

from __future__ import annotations

import json
import random

import pytest
import torch
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


def test_synthetic_spheroid_and_pk_coupled_and_calc_legal():
    """Spheroid/pk raw fields are stated in the prose *or* are the canonical
    defaults the gold set forces the model to infer, and every raw builds a
    design that passes the verifier (no invented physiology)."""
    rows = generate(30, 30, 200, 200, seed=9)
    domains = {r["domain"] for r in rows}
    assert {"spheroid", "pk"} <= domains
    for row in rows:
        goal = row["goal"].lower()
        if row["domain"] == "spheroid":
            s = row["raw"]["spheroid"]
            # prose uses the display name ("96-well ultra-low-attachment (ULA)")
            # not the raw key ("96-ula"), so check the canonical tokens.
            fmt = s["spheroid_format"]
            if fmt == "hanging-drop":
                assert "hanging" in goal
            elif "ula" in goal:
                assert fmt.split("-")[0] in goal
            else:
                # default-bearing geometry row: no plate stated, raw falls back
                # to the canonical 96-ula.
                assert fmt == "96-ula"
                assert "assuming a solid sphere" in goal and "in diameter" in goal
            # cps/count coupling by pattern:
            if "how many cells per spheroid" in goal:
                assert s["spheroid_count"] == 1  # inverse: cps is the answer
            elif "assuming a solid sphere" in goal and "in diameter" in goal:
                assert s["spheroid_count"] == 1  # inverse geometry, volume ask
            elif "medium volume per spheroid" in goal or "total medium volume in ml" in goal:
                if "cells/spheroid" in goal:
                    assert str(s["cells_per_spheroid"]) in goal  # multi-target D
                else:
                    assert s["cells_per_spheroid"] == 1000        # partial-info default
            else:
                assert str(s["cells_per_spheroid"]) in goal        # forward A / D
            # cd coupling: stated when the prose names a cell size, else the
            # canonical 20 µm default.
            if "mean cell diameter" in goal or "mean diameter of" in goal:
                assert str(int(s["cell_diameter_um"])) in goal
            else:
                assert s["cell_diameter_um"] == 20.0
            if "doubling_time_h" in s:
                assert "double" in goal
        elif row["domain"] == "pk":
            p = row["raw"]["pk"]
            assert p["compound"] in row["goal"]
            if "mM" in row["goal"]:
                # mM → µM unit-trap row: the raw is in µM, the prose in mM.
                assert f"{p['inlet_concentration_uM'] / 1000:g}" in goal
                assert f"{p['outlet_concentration_uM'] / 1000:g}" in goal
            else:
                assert f"{p['inlet_concentration_uM']:g}" in goal
                assert f"{p['outlet_concentration_uM']:g}" in goal
            assert f"{p['flow_rate_uLmin']:g}" in goal
            if "molecular_weight_g_mol" in p:
                assert "molecular weight" in goal
            if "system_volume_uL" in p:
                assert "system volume" in goal
            if "dose_interval_h" in p:
                assert "every" in goal
        inp = DesignInput(goal=row["goal"], rationale="test", **row["raw"])
        assert not has_errors(verify_design(build_design(inp))), row["domain"]


# ---------------------------------------------------------------------------
# Gold pairs
# ---------------------------------------------------------------------------


def test_gold_pairs_all_consistent():
    pairs, skipped = gold_pairs()
    assert len(pairs) == 46  # 24 flow + 8 culture + 8 spheroid + 6 pk
    assert sorted(skipped) == [
        "pk-cell-free-subtraction",
        "pk-complete-clearance-panel",
        "pk-high-extraction-clearance",
        "pk-low-extraction-clearance",
        "pk-mass-cleared",
        "pk-repeat-dose-24h",
        "plate-hemocytometer-seed-96well",
        "plate-thaw-viability-6well",
        "spheroid-count-from-suspension",
        "spheroid-doxorubicin-dosing",
        "spheroid-um-mm-unit-ambiguity",
    ]
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
    # spheroid-96well-total → cells_total 96000, total medium 9.6 mL
    inp = DesignInput(goal="x", rationale="x", **by_gold["spheroid-96well-total"]["raw"])
    s = build_design(inp).spheroid
    assert s.cells_total == pytest.approx(96000.0, rel=1e-6)
    assert s.total_medium_ml == pytest.approx(9.6, rel=1e-6)
    # spheroid-growth-72h → expected_cells_after_growth 5278
    inp = DesignInput(goal="x", rationale="x", **by_gold["spheroid-growth-72h"]["raw"])
    assert build_design(inp).spheroid.expected_cells_after_growth == pytest.approx(5278.03, rel=1e-4)
    # pk-accumulation-ratio → E 0.3, Cl 0.6, t½ 3.851 h, R 1.0135
    inp = DesignInput(goal="x", rationale="x", **by_gold["pk-accumulation-ratio"]["raw"])
    p = build_design(inp).pk
    assert p.extraction_ratio == pytest.approx(0.3, rel=1e-6)
    assert p.clearance_uLmin == pytest.approx(0.6, rel=1e-6)
    assert p.half_life_h == pytest.approx(3.8508176697774745, rel=1e-6)
    assert p.accumulation_ratio == pytest.approx(1.0134791547306115, rel=1e-6)


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
    assert rep["consistency_rate"] == 0.0


# ---------------------------------------------------------------------------
# Diversity generators (negatives + cross-domain composites)
# ---------------------------------------------------------------------------


def test_composite_rows_carry_two_blocks_and_verify_clean():
    """A composite goal merges two single-domain rows; the raw carries two
    top-level blocks and still builds a design that passes the verifier."""
    from labwright.extract.synthetic import generate_composite, _COMPOSITE_PAIRS

    rng = random.Random(4)
    for _ in range(20):
        row = generate_composite(rng)
        assert len(row["raw"]) == 2, f"composite must merge two blocks: {row['domain']}"
        assert row["domain"].startswith("composite:")
        inp = DesignInput(goal=row["goal"], rationale="test", **row["raw"])
        assert not has_errors(verify_design(build_design(inp)))
    # composites only ever merge a defined pair (never an arbitrary two-block raw)
    seen = {generate_composite(rng)["domain"] for _ in range(200)}
    pairs = {f"composite:{a}+{b}" for a, b in _COMPOSITE_PAIRS}
    assert seen <= pairs


def test_multi_block_prompt_is_versioned():
    """The multi-block prompt is a separate constant: lora_v3 (single-block)
    is never evaluated under a prompt it did not train on, and the multi-block
    variant renders a composite row's two-block target without error."""
    from labwright.extract.data import SYSTEM_PROMPT, SYSTEM_PROMPT_MULTI, SCHEMA_PROMPT, SCHEMA_PROMPT_MULTI
    from labwright.extract.synthetic import generate_composite

    assert "never two" in SCHEMA_PROMPT
    assert "two blocks when the goal describes two subsystems" in SCHEMA_PROMPT_MULTI
    assert "raw input block:" in SYSTEM_PROMPT and "raw input block(s):" in SYSTEM_PROMPT_MULTI
    # composite row encodes under the multi-block prompt (masked loss survives)
    row = generate_composite(random.Random(21))
    stub = _StubTokenizer()
    enc = encode_example(stub, row["goal"], raw_to_json(row["raw"]), max_len=2048,
                         system_prompt=SYSTEM_PROMPT_MULTI)
    assert enc is not None and len(enc["input_ids"]) > 0


def test_negative_sample_perturbs_embedded_approx_only():
    """A negative sample flips one '≈value unit' derived claim; the raw block is
    untouched (the target stays correct), and only ≈-bearing goals change."""
    from labwright.extract.synthetic import _maybe_perturb_approx, generate_gradient

    rng = random.Random(11)
    changed = 0
    for seed in range(80):
        row = generate_gradient(random.Random(seed))
        if "≈" not in row["goal"]:
            continue
        new_goal = _maybe_perturb_approx(rng, row["goal"], p=1.0)
        if new_goal != row["goal"]:
            changed += 1
            assert "≈" in new_goal
    assert changed > 0
    # a goal with no embedded ≈ is returned untouched
    plain = "Culture cells at 1e5 cells/cm² in a 96-well plate."
    assert _maybe_perturb_approx(rng, plain, p=1.0) == plain


def test_negative_sample_changes_only_the_approx_value():
    """A perturbed goal is byte-identical to the original outside the ≈-value
    span: prose, units, and every non-≈ number are untouched, so the raw block
    (the training target) stays correct. Normalising ≈-values to a sentinel must
    make the two goals equal — any collateral edit breaks the invariant."""
    import re as _re

    from labwright.extract.synthetic import _maybe_perturb_approx, generate_gradient

    rng = random.Random(9)
    norm = lambda s: _re.sub(r"≈\s*[0-9]+(?:\.[0-9]+)?", "≈V", s)
    checked = 0
    for seed in range(80):
        row = generate_gradient(random.Random(seed))
        if "≈" not in row["goal"]:
            continue
        new_goal = _maybe_perturb_approx(rng, row["goal"], p=1.0)
        if new_goal == row["goal"]:
            continue
        checked += 1
        assert norm(new_goal) == norm(row["goal"])
        # the ≈-value actually changed (the sentinel-normalised form differs
        # from the plain goal only when a number was flipped, not a no-op)
        assert "≈V" in norm(new_goal)
    assert checked > 0


def test_negative_sample_generate_never_corrupts_raw():
    """With neg_frac active, every produced row still carries a parseable raw
    block: the negative-sample pass rewrites only the goal prose and leaves the
    training target intact for every domain that embeds ≈ claims."""
    from labwright.extract.synthetic import generate

    rows = generate(
        n_flow=40, n_culture=40, n_gradient=120, n_breathing=120,
        n_pulsatile=120, n_scaling=120, n_composite=40, neg_frac=1.0, seed=3,
    )
    assert rows
    # the hook actually fired: with p=1.0 every ≈-bearing candidate got flipped
    assert any("≈" in r["goal"] for r in rows)
    for r in rows:
        assert isinstance(r["raw"], dict) and r["raw"], r["goal"][:60]


# ---------------------------------------------------------------------------
# extract_batch: empty-goal fast path + parse-back ordering
# ---------------------------------------------------------------------------


class _BatchStubTokenizer:
    """Minimal left-padded tokenizer for the batch decode path (one token per
    character, ``padding_side='left'`` as :func:`configure_tokenizer` enforces
    for decoder-only generation)."""

    padding_side = "left"
    pad_token_id = 0
    eos_token = "<|endoftext|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        assert tokenize is False
        return rendered

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=1024, return_token_type_ids=False):
        rows = [[ord(c) for c in p] for p in prompts]
        width = max(len(r) for r in rows)
        padded = [[self.pad_token_id] * (width - len(r)) + r for r in rows]  # left pad
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["".join(chr(int(t)) for t in row) for row in token_ids.tolist()]


class _BatchStubModel:
    """Fake causal model: emits one fixed JSON per row after the prompt width,
    matching the real batch-generate shape (out[:, prompt_width:] is output)."""

    def __init__(self, outputs: list[str]):
        self._outputs = outputs

    def generate(self, input_ids, max_new_tokens=384, do_sample=False,
                 pad_token_id=None, **kwargs):
        rows = []
        gen_len = max(len(o) for o in self._outputs)
        for i, row in enumerate(input_ids):
            gen = [ord(c) for c in self._outputs[i]]
            gen = gen + [pad_token_id or 0] * (gen_len - len(gen))  # fixed width, like HF
            rows.append(torch.cat([row, torch.tensor(gen, dtype=row.dtype)]))
        return torch.stack(rows)


def _stub_extractor():
    """An Extractor with stubbed tokenizer/model/device (no weights on disk)."""
    ex = Extractor.__new__(Extractor)
    ex.system_prompt = "system"
    ex.device = "cpu"
    ex.tokenizer = _BatchStubTokenizer()
    ex.model = _BatchStubModel([
        '{"culture": {"plate_format": "96-well", "wells": 12}}',
        '{"flow": {"width_um": 400, "height_um": 100}}',
        '{"flow": {"width_um": 800, "height_um": 200}}',
    ])
    return ex


def test_extract_batch_empty_goals_returns_empty_list():
    """extract_batch([]) short-circuits to [] without touching the model."""
    ex = _stub_extractor()
    assert ex.extract_batch([]) == []


def test_extract_batch_preserves_row_order_and_parses_json():
    """Left-padded batch decode returns parsed JSON in input order — the
    un-padding must not scramble rows when goals differ in length."""
    ex = _stub_extractor()
    goals = ["a short goal", "a considerably longer second goal with more words", "third"]
    out = ex.extract_batch(goals)
    assert out[0] == {"culture": {"plate_format": "96-well", "wells": 12}}
    assert out[1] == {"flow": {"width_um": 400, "height_um": 100}}
    assert out[2] == {"flow": {"width_um": 800, "height_um": 200}}
    assert len(out) == len(goals)


def test_generate_with_composites_and_negatives():
    """--n-composite / --neg-frac wiring: composite rows appear and negative
    rows keep their raw identical to the parent single-domain row."""
    from labwright.extract.synthetic import _APPROX_RE, generate

    rows = generate(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                    n_composite=10, neg_frac=0.2, seed=13)
    comps = [r for r in rows if r["domain"].startswith("composite:")]
    assert len(comps) == 10
    negs = [r for r in rows if _APPROX_RE.search(r["goal"])]
    # negatives only exist on ≈-bearing domains (breathing/scaling/gradient)
    assert negs  # at least one appeared
    # determinism
    again = generate(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                     n_composite=10, neg_frac=0.2, seed=13)
    assert rows == again
