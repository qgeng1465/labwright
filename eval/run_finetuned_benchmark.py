"""Run the fine-tuned extractor as a design-benchmark system (the fast path).

The main benchmark's four systems (bare / soft_gate / self_verify / labwright)
all run against a remote API model. The fine-tuned raw-input extractor
(Qwen2.5-1.5B LoRA) is a *local* model, so it never had a slot in that harness.
This script gives it one: for every gold goal it runs
``extractor.extract_plan(goal)`` — goal → raw → derive → verify — and scores the
result with the *same* usable / hallucination / target-recovery rules as
Labwright (``eval.benchmark._score_design``). The extractor proposes raw inputs
only; every number is produced and re-proven by the calculators, exactly like
the agent path.

Honesty: the extractor was fine-tuned on synthetic flow/culture instances whose
shear targets are reused from the benchmark gold sets (``labwright/extract/
synthetic.py``). On those domains its numbers are *in-distribution* and must be
labelled as such at report time; on domains it never trained on (spheroid, and
the new PK domain) it is a clean out-of-distribution test.

Usage::

    python -m eval.run_finetuned_benchmark --gold eval/gold_experiments.json \\
        --out results/eval_finetuned_reading.json --device cuda
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import evaluate, load_gold
from labwright.extract.pipeline import Extractor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="eval/gold_experiments.json", help="Gold set to run")
    ap.add_argument("--out", default="results/eval_finetuned_reading.json", help="Output JSON path")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="results/extractor/lora")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--multi-block", action="store_true",
                    help="use SYSTEM_PROMPT_MULTI (lora_v4 composite extraction)")
    ap.add_argument("--repair-retries", type=int, default=0,
                    help="on schema/build failure, re-prompt the model with the validator "
                         "error up to N extra attempts (0 = baseline, schema error is final)")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    print(f"gold entries: {len(gold)}   adapter: {args.adapter}   repair-retries: {args.repair_retries}", flush=True)

    ext = Extractor(model_path=args.model, adapter_path=args.adapter, device=args.device,
                    multi_block=args.multi_block, repair_retries=args.repair_retries)
    print(f"extractor on {ext.device}", flush=True)

    def progress(msg: str) -> None:
        print(f"  {msg}", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def checkpoint(partial: dict) -> None:
        partial["system"] = "finetuned"
        partial["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(partial, fh, indent=2, ensure_ascii=False)

    # agent_factory and chat are unused by the finetuned branch of _run_system.
    summary = evaluate(
        gold, agent_factory=None, chat=None,
        progress=progress, checkpoint=checkpoint, systems=("finetuned",), extractor=ext,
    )

    summary["system"] = "finetuned"
    summary["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_entry"}, indent=2))
    print(f"repairs issued: {ext.repairs}")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
