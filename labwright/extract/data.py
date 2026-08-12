"""Data plumbing for the extractor fine-tune — chat rendering and loss masking.

The extractor maps an experimental *goal* (prose) to the *raw* design inputs
only — never the derived numbers, which always come from the calculators. We
fine-tune a chat-instruct model with a masked LM objective: loss is computed
**only on the assistant's JSON turn**, so everything the model says is scored
against the exact raw-JSON rendering rather than the instruction or the goal.
"""

from __future__ import annotations

import json

#: System instruction for the extractor. Deliberately mirrors Labwright's
#: raw/derived split: the model reports raw inputs and never touches a number
#: the calculators own.
SYSTEM_PROMPT = (
    "You extract raw wet-lab design inputs from an experimental goal. "
    "Return a single JSON object with ONLY the raw input blocks: chip, flow "
    "and cells for a microfluidic channel design, or culture for a "
    "plate-culture design. Do NOT compute or report derived numbers such as "
    "wall shear stress, Reynolds number, seed counts, medium volumes or "
    "confluence — those are always calculated deterministically after you "
    "return. Return ONLY the JSON object."
)

#: ChatML assistant-turn opener, as rendered by the Qwen2.5 template. The loss
#: mask boundary is found by locating this marker in the rendered text.
_ASSISTANT_MARKER = "<|im_start|>assistant\n"


def render_chat(goal: str, raw_json_str: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Build the three-turn chat for one (goal → raw) example."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
        {"role": "assistant", "content": raw_json_str},
    ]


def encode_example(
    tokenizer,
    goal: str,
    raw_json_str: str,
    max_len: int = 1024,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict | None:
    """Tokenize one example and build ``input_ids``/``labels`` with masked loss.

    Labels carry ``-100`` for everything before the assistant turn, so the
    model only learns from its own JSON output. Returns ``None`` when the
    assistant content is truncated away entirely (a skip, not a bug).
    """
    msgs = render_chat(goal, raw_json_str, system_prompt=system_prompt)
    full = tokenizer.apply_chat_template(msgs, tokenize=False)
    start = full.find(_ASSISTANT_MARKER)
    if start < 0:
        raise ValueError(
            "assistant-turn marker not found in rendered chat — is the "
            "tokenizer's chat template ChatML-style?"
        )
    start += len(_ASSISTANT_MARKER)

    enc = tokenizer(full, return_offsets_mapping=True, max_length=max_len, truncation=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    cut = len(input_ids)
    for i, (s, _e) in enumerate(offsets):
        if s >= start:
            cut = i
            break
    labels = [-100] * cut + input_ids[cut:]
    if not any(l != -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def raw_to_json(raw: dict) -> str:
    """Deterministic, compact JSON rendering of a raw input block."""
    return json.dumps(raw, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
