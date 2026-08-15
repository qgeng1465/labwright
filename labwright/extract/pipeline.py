"""Inference pipeline for the fine-tuned raw-input extractor.

:class:`Extractor` loads the LoRA-tuned model, greedily decodes a raw input
block for a goal, then runs it through the real Labwright pipeline
(:func:`labwright.design.build_design` + :func:`labwright.verify.checker.verify_design`).
The returned plan is the *same* object the full agent would produce — derived
numbers recomputed by the calculators, then re-proven by the verifier — so a
wrong flow rate or seed density is caught exactly as it would be in the
tool-using agent, not silently accepted.
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from peft import PeftModel
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from labwright.design import DesignInput, _reject_derived_fields, build_design
from labwright.extract.data import SYSTEM_PROMPT, SYSTEM_PROMPT_MULTI
from labwright.verify.checker import Issue, format_issues, verify_design

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_ADAPTER = "results/extractor/lora"


def configure_tokenizer(tokenizer) -> None:
    """Normalize a tokenizer for decoder-only generation, in place.

    Decoder-only models must be **left-padded**: with right padding the pad
    tokens of a shorter prompt sit between the prompt and the new tokens, the
    model attends over them, and the decoded output is corrupted. This was the
    cause of the 19.9 % JSON parse rate in the first extractor eval run.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def parse_json(text: str) -> dict[str, Any] | None:
    """Robust JSON extraction from model output (fences + balanced braces)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for candidate in (cleaned, text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class Extractor:
    """A LoRA-tuned goal → raw-inputs model, wired into the Labwright pipeline."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        adapter_path: str | None = _DEFAULT_ADAPTER,
        device: str | None = None,
        multi_block: bool = False,
        repair_retries: int = 0,
    ):
        self.multi_block = multi_block
        self.system_prompt = SYSTEM_PROMPT_MULTI if multi_block else SYSTEM_PROMPT
        #: schema-repair retries per goal (see :meth:`extract_plan`). 0 = the
        #: baseline behaviour: a schema error is final and the entry fails.
        self.repair_retries = repair_retries
        #: cumulative count of repair re-prompts issued across this instance's
        #: lifetime (for reporting; a repair is only counted when it actually runs).
        self.repairs = 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        configure_tokenizer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map=self.device
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def extract(self, goal: str, max_new_tokens: int = 384) -> dict[str, Any] | None:
        """Return the parsed raw input block for ``goal``, or ``None``."""
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": goal},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_json(text)

    def _repair(self, goal: str, raw: dict[str, Any] | None, error: str) -> dict[str, Any] | None:
        """Re-run the extractor with the rejection reason appended to the goal.

        The extractor is an instruct model, so a follow-up prompt that quotes
        the rejected block and the validator's error lets it drop an extra
        field, fix an enum value, or add a missing key. Greedy decoding is
        deterministic, so a repair is a *different* single attempt, not a
        stochastic retry — an A/B with ``repair_retries=0`` cleanly isolates
        the effect.
        """
        self.repairs += 1
        block = json.dumps(raw, ensure_ascii=False) if raw is not None else "no JSON object parsed"
        user = (
            "Your previous raw-input block was rejected. Fix the fields the "
            "validator names and output only the corrected raw-input block as JSON.\n\n"
            f"Goal: {goal}\n\nRejected block:\n{block}\n\n"
            f"Validator error:\n{error}"
        )
        return self.extract(user)

    def extract_plan(self, goal: str, repair_retries: int | None = None) -> tuple[object | None, list[Issue] | None, str | None]:
        """Extract → build → verify. Returns ``(plan, issues, error)``.

        ``plan`` is a :class:`~labwright.schema.design.DesignPlan` (or ``None``
        on parse/schema failure), ``issues`` the verifier findings, and
        ``error`` a short reason when the pipeline could not run. With
        ``repair_retries`` (defaults to the instance setting) a failed parse or
        schema gate is re-attempted by re-prompting the model with the error —
        up to ``repair_retries`` extra attempts. A successful plan still crosses
        the *same* derived-field / schema / build gates on every attempt.
        """
        if repair_retries is None:
            repair_retries = self.repair_retries
        raw = self.extract(goal)
        for attempt in range(repair_retries + 1):
            if raw is None:
                if attempt < repair_retries:
                    raw = self._repair(goal, None, "unparseable_json: no valid JSON object in the output")
                    continue
                return None, None, "unparseable_json"
            # The extracted raw inputs cross the *same* gate as the agent's
            # submit_design: a derived field (seed_count, expected_diameter_um, ...)
            # invented by the extractor is rejected, never silently overwritten.
            try:
                _reject_derived_fields(raw)
            except ValueError as exc:
                if attempt < repair_retries:
                    raw = self._repair(goal, raw, f"derived_field_rejected: {exc}")
                    continue
                return None, None, f"derived_field_rejected: {exc}"
            try:
                inp = DesignInput(goal=goal, rationale="Auto-extracted raw inputs", **raw)
            except (ValidationError, TypeError) as exc:
                if attempt < repair_retries:
                    raw = self._repair(goal, raw, f"schema_error: {exc}")
                    continue
                return None, None, f"schema_error: {exc}"
            try:
                plan = build_design(inp)
            except (ValueError, KeyError, TypeError) as exc:
                if attempt < repair_retries:
                    raw = self._repair(goal, raw, f"build_error: {exc}")
                    continue
                return None, None, f"schema_error: {exc}"
            issues = verify_design(plan)
            return plan, issues, None
        return None, None, "repair_exhausted"

    def extract_batch(self, goals: list[str], max_new_tokens: int = 384) -> list[dict[str, Any] | None]:
        """Decode a batch of goals in one generate call (left-padded).

        The single-row :meth:`extract` dominates wall time on the V100 (a 1.5B
        fp16 decode of ~300 tokens is ~3 s/row); batching raises GPU utilization
        so the SciRecipe audit over thousands of rows stays tractable. Outputs
        are parsed back to their row order after un-padding.
        """
        if not goals:
            return []
        msgs = [[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": g},
        ] for g in goals]
        prompts = [self.tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024, return_token_type_ids=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Generated tokens always begin at the batch prompt width (generate
        # extends the full padded batch), so that slice drops every prompt and
        # keeps only the decoded output — independent of padding side.
        prompt_width = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts = self.tokenizer.batch_decode(
            out[:, prompt_width:], skip_special_tokens=True)
        return [parse_json(t) for t in texts]


def format_audit(goal: str, plan, issues: list[Issue] | None, error: str | None) -> str:
    """Human-readable ``labwright audit`` output."""
    lines = [f"Goal: {goal}", ""]
    if error:
        lines.append(f"[error] {error}")
        return "\n".join(lines)
    assert plan is not None and issues is not None
    lines.append("Extracted raw inputs:")
    for block in ("chip", "flow", "cells", "culture", "dosing", "stats"):
        obj = getattr(plan, block, None)
        if obj is None:
            continue
        lines.append(f"  {block}: {json.dumps(obj.model_dump(mode='json'), ensure_ascii=False)}")
    lines.append("")
    lines.append("Derived numbers (computed by the calculators):")
    if plan.derived is not None:
        d = plan.derived
        lines.append(
            f"  shear {d.shear_pa:.4f} Pa | Re {d.reynolds:.2f} | ΔP {d.pressure_drop_pa:.1f} Pa | "
            f"V {d.channel_volume_ul:.3f} µL | τ_res {d.residence_time_s:.1f} s"
        )
    if plan.culture is not None:
        c = plan.culture
        lines.append(
            f"  seed {c.seed_per_well:g}/well × {c.wells} = {c.total_seed_count:g} | "
            f"med {c.medium_volume_per_well_ml:g} mL/well = {c.total_medium_ml:g} mL total"
            + (f" | confluence {c.expected_confluence_pct:.1f}%" if c.expected_confluence_pct is not None else "")
        )
    lines.append("")
    lines.append("Verifier:")
    lines.append("  " + format_issues(issues).replace("\n", "\n  "))
    return "\n".join(lines)


__all__ = ["Extractor", "parse_json", "format_audit"]
