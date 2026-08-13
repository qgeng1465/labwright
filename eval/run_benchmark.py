"""Run the Labwright benchmark against a real model (DeepSeek by default).

Usage
-----
    python -m eval.run_benchmark                      # all gold entries
    python -m eval.run_benchmark --limit 3 --out /tmp/eval.json
    python -m eval.run_benchmark --model deepseek-chat
    python -m eval.run_benchmark --systems bare,soft_gate,self_verify,labwright

Each gold experiment is run once per requested system:

* **bare-LLM** — asked once, with the full JSON schema, to produce a complete
  design and *compute the numbers itself* (no calculators).
* **soft-gate** — bare + a "check yourself" prompt: re-derive your own derived
  numbers before finalising. No calculators, no verifier.
* **self-verify** — two LLM passes: propose, then hand the model back its own
  raw inputs and ask it to recompute the derived numbers itself. The naive
  "use the LLM as the verifier" alternative.
* **Labwright** — the ReAct agent runs its normal tool loop and must end in a
  verified design.

The default ``--systems`` is ``bare,labwright`` (the historical comparison);
add ``soft_gate`` and ``self_verify`` for the competitor baselines.

Each gold experiment is run twice:

* **bare LLM** — asked once, with the full JSON schema, to produce a complete
  design and *compute the numbers itself* (no calculators).
* **Labwright** — the ReAct agent runs its normal tool loop and must end in a
  verified design.

Output is written as JSON (results/<name>.json) and printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labwright.agent import DesignAgent, LLMClient
from eval.benchmark import evaluate, load_gold


def _make_chat(model: str, base_url: str | None, max_tokens: int = 8192):
    client = LLMClient(model=model, base_url=base_url)

    def chat(prompt: str) -> str:
        return client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens).content

    return chat


def _make_agent_factory(
    model: str,
    base_url: str | None,
    max_iterations: int = 12,
    verify_gate: bool = True,
    max_submission_attempts: int = 1,
):
    def factory() -> DesignAgent:
        return DesignAgent(
            LLMClient(model=model, base_url=base_url),
            max_iterations=max_iterations,
            verify_gate=verify_gate,
            max_submission_attempts=max_submission_attempts,
        )

    return factory


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Labwright eval against a live model")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N gold entries (0 = all)")
    ap.add_argument("--out", default="results/eval_v1.json", help="Output JSON path")
    ap.add_argument("--model", default=os.environ.get("LABWRIGHT_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--gold", default=None, help="Path to a gold JSON (default: eval/gold_experiments.json)")
    ap.add_argument(
        "--systems", default="bare,labwright",
        help="Comma-separated systems to run (bare, soft_gate, self_verify, labwright, "
             "tool_no_gate, labwright_iter)",
    )
    ap.add_argument(
        "--max-submission-attempts", type=int, default=3,
        help="Fix-and-resubmit attempts for labwright_iter (verifier's review_required "
             "feeds back into the loop). labwright itself stays first-submit.",
    )
    args = ap.parse_args()

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    for name in systems:
        if name not in ("bare", "soft_gate", "self_verify", "labwright", "tool_no_gate", "labwright_iter"):
            print(f"unknown system: {name}", file=sys.stderr)
            return 2

    gold = load_gold(args.gold) if args.gold else load_gold()
    if args.limit:
        gold = gold[: args.limit]
    print(f"gold entries: {len(gold)}   model: {args.model}   systems: {','.join(systems)}")

    chat = _make_chat(args.model, args.base_url)
    agent_factory = _make_agent_factory(args.model, args.base_url, args.max_iterations)
    # The no-gate ablation reuses the same model/loop but with the verifier off.
    agent_factory_nogate = _make_agent_factory(args.model, args.base_url, args.max_iterations, verify_gate=False)
    # The iterative agent honours the prompt's review_required fix loop: the
    # verifier's report feeds back and the agent resubmits (default 3 attempts).
    agent_factory_iter = _make_agent_factory(
        args.model, args.base_url, args.max_iterations,
        max_submission_attempts=args.max_submission_attempts,
    )

    def progress(msg: str) -> None:
        print(f"  {msg}", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def checkpoint(partial: dict) -> None:
        # Save after every entry so a mid-run failure never loses the API spend.
        partial["model"] = args.model
        partial["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(partial, fh, indent=2, ensure_ascii=False)

    summary = evaluate(
        gold, agent_factory, chat, progress,
        checkpoint=checkpoint, systems=systems, agent_factory_nogate=agent_factory_nogate,
        agent_factory_iter=agent_factory_iter,
    )

    summary["model"] = args.model
    summary["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
