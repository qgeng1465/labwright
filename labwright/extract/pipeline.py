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

from labwright.design import DesignInput, build_design
from labwright.extract.data import SYSTEM_PROMPT
from labwright.verify.checker import Issue, format_issues, verify_design

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_ADAPTER = "results/extractor/lora"


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
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map=self.device
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def extract(self, goal: str, max_new_tokens: int = 384) -> dict[str, Any] | None:
        """Return the parsed raw input block for ``goal``, or ``None``."""
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
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

    def extract_plan(self, goal: str) -> tuple[object | None, list[Issue] | None, str | None]:
        """Extract → build → verify. Returns ``(plan, issues, error)``.

        ``plan`` is a :class:`~labwright.schema.design.DesignPlan` (or ``None``
        on parse/schema failure), ``issues`` the verifier findings, and
        ``error`` a short reason when the pipeline could not run.
        """
        raw = self.extract(goal)
        if raw is None:
            return None, None, "unparseable_json"
        try:
            inp = DesignInput(goal=goal, rationale="Auto-extracted raw inputs", **raw)
        except ValidationError as exc:
            return None, None, f"schema_error: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}"
        try:
            plan = build_design(inp)
        except (ValueError, KeyError) as exc:
            return None, None, f"schema_error: {exc}"
        issues = verify_design(plan)
        return plan, issues, None

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": g},
        ] for g in goals]
        prompts = [self.tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024, return_token_type_ids=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Left-padding makes per-row prompt lengths unequal to the batch width;
        # the mask sum recovers each row's own prompt length for un-padding.
        prompt_lens = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts = [
            self.tokenizer.decode(out[i, prompt_lens[i]:], skip_special_tokens=True)
            for i in range(len(goals))
        ]
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
