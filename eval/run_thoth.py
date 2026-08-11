"""Run the Labwright benchmark with Thoth (ICLR 2026) as the model.

Thoth (``manglu3935/Thoth``, cc-by-4.0) is the closest published LLM
competitor: an 8B Qwen3-based model trained to generate biological protocol
text with a structured component-based reward. It is benchmarked through the
*identical* harness as the DeepSeek models — same gold goals, same bare prompt,
same JSON extraction, same bare-LLM scoring rules — so the only difference from
the paper's Table 1 rows is the model weights.

Thoth's native output is protocol prose (``<think>``/``<key>``/``<orc>``/
``<note>`` segments), so a prompt-only comparison is the honest one: if the
trained protocol generator cannot emit a self-consistent design when asked the
same question through the same schema, that is a finding, not a rigging.

Usage::

    python -m eval.run_thoth --gold eval/gold_blind.json --systems bare,soft_gate,self_verify \\
        --out results/eval_blind_thoth.json --model-dir /data/hf_models/manglu3935/Thoth

Defaults to the reading set and ``bare``. Model loads in BF16 on GPU when
available, else CPU (slow — intended for the queued GPU run).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import evaluate, load_gold


def _build_chat(model_dir: str, max_new_tokens: int = 2048):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_thoth] loading {model_dir} on {dev}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        device_map="auto" if dev == "cuda" else None,
    )
    if dev == "cpu":
        model = model.to("cpu")
    model.eval()

    def chat(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if dev == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return chat


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Labwright eval with Thoth")
    ap.add_argument("--gold", default=None, help="Path to gold JSON (default eval/gold_experiments.json)")
    ap.add_argument("--systems", default="bare", help="Comma-separated: bare,soft_gate,self_verify")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--model-dir", default="/data/hf_models/manglu3935/Thoth")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    for name in systems:
        if name not in ("bare", "soft_gate", "self_verify"):
            print(f"unknown system: {name}", file=sys.stderr)
            return 2

    gold = load_gold(args.gold) if args.gold else load_gold()
    print(f"gold entries: {len(gold)}   model: thoth-8b   systems: {','.join(systems)}", flush=True)

    chat = _build_chat(args.model_dir, args.max_new_tokens)

    def agent_factory():
        raise AssertionError("labwright system is not available for Thoth (prompt-only comparison)")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def checkpoint(partial: dict) -> None:
        partial["model"] = "thoth-8b"
        partial["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(partial, fh, indent=2, ensure_ascii=False)

    summary = evaluate(gold, agent_factory, chat, print, checkpoint=checkpoint, systems=systems)
    summary["model"] = "thoth-8b"
    summary["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nsaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
