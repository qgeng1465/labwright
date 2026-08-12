"""Fine-tune the raw-input extractor with LoRA on the V100.

Trains a small chat-instruct model (default Qwen2.5-1.5B-Instruct) to map a
wet-lab goal (prose) to the *raw* design inputs. Loss is computed only on the
assistant JSON turn (see :mod:`labwright.extract.data`), so the model is never
credited for echoing the instruction — it must produce the exact raw block.

Hardware notes
--------------
- fp16, not bf16: the V100 (SM70) has no BF16 tensor cores.
- No bitsandbytes: keep the model in fp16 (1.5B ≈ 3 GB) + LoRA, no 4-bit.
- Run through the resource arbitrator, e.g.::

    python3 ~/.claude/resources/arbitrate.py run --name extract-train \
        --gpu-mem 12 --cpu 4 --ram 12 --detach -- \\
        python3 -m labwright.extract.train --data results/extractor
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from labwright.extract.data import encode_example, raw_to_json, SYSTEM_PROMPT

#: Qwen2.5-style dense LoRA targets — all linear projections.
_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_rows(data_dir: Path) -> list[dict]:
    """Load train.jsonl and gold_pairs.jsonl; gold pairs are appended."""
    rows: list[dict] = []
    for name in ("train.jsonl", "gold_pairs.jsonl"):
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    return rows


def tokenize_rows(tokenizer, rows: list[dict], max_len: int) -> list[dict]:
    examples: list[dict] = []
    for row in rows:
        enc = encode_example(tokenizer, row["goal"], raw_to_json(row["raw"]), max_len=max_len)
        if enc is not None:
            examples.append(enc)
    return examples


def make_collator(tokenizer):
    return DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/hf_models/Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="results/extractor")
    parser.add_argument("--out", default="results/extractor/lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", type=int, default=0, help="use N rows only (CPU smoke)")
    parser.add_argument("--log", default="results/extractor/train.log")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="mixed precision (default: auto-on when CUDA is available)")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data)
    rows = load_rows(data_dir)
    if not rows:
        print("[error] no rows found in", data_dir, file=os.sys.stderr)
        return 2
    if args.smoke:
        rows = rows[: args.smoke]
    random.Random(args.seed).shuffle(rows)
    n_train = int(len(rows) * 0.9) or 1
    train_rows, eval_rows = rows[:n_train], rows[n_train:]

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ex = tokenize_rows(tokenizer, train_rows, args.max_len)
    eval_ex = tokenize_rows(tokenizer, eval_rows, args.max_len)
    train_ds = Dataset.from_list(train_ex)
    eval_ds = Dataset.from_list(eval_ex) if eval_ex else None
    print(f"train {len(train_ex)} / eval {len(eval_ex)} (of {len(rows)} rows)")

    use_fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if use_fp16 else torch.float32, use_cache=False
    )
    peft_cfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    log_dir = Path(args.log).resolve()
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out).resolve()
    steps_per_epoch = math.ceil(len(train_ex) / (args.batch * args.grad_accum))
    warmup_steps = max(1, int(0.03 * steps_per_epoch * args.epochs))
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if eval_ds is not None else "no",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=make_collator(tokenizer),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    final = {
        "n_train": len(train_ex),
        "n_eval": len(eval_ex),
        "final_train_loss": trainer.state.log_history[-1].get("loss"),
        "eval_loss": getattr(trainer.state, "best_metric", None),
    }
    with open(log_dir.parent / "train_report.json", "w") as fh:
        json.dump(final, fh, indent=2)
    print("saved adapter ->", out_dir)
    print("report ->", log_dir.parent / "train_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
