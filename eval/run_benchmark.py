"""Run the Labwright benchmark against a real model (DeepSeek by default).

Usage
-----
    python -m eval.run_benchmark                      # all gold entries
    python -m eval.run_benchmark --limit 3 --out /tmp/eval.json
    python -m eval.run_benchmark --model deepseek-chat

Each gold experiment is run twice:

* **bare LLM** — asked once, with the full JSON schema, to produce a complete
  design and *compute the numbers itself* (no calculators).
* **Labwright** — the ReAct agent runs its normal tool loop and must end in a
  verified design.

Output is written as JSON (results/<name>.json) and printed.
"""

from __future__ import annotations

import argparse
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


def _make_agent_factory(model: str, base_url: str | None, max_iterations: int = 12):
    def factory() -> DesignAgent:
        return DesignAgent(LLMClient(model=model, base_url=base_url), max_iterations=max_iterations)

    return factory


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Labwright eval against a live model")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N gold entries (0 = all)")
    ap.add_argument("--out", default="results/eval_v1.json", help="Output JSON path")
    ap.add_argument("--model", default=os.environ.get("LABWRIGHT_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-iterations", type=int, default=12)
    args = ap.parse_args()

    gold = load_gold()
    if args.limit:
        gold = gold[: args.limit]
    print(f"gold entries: {len(gold)}   model: {args.model}")

    chat = _make_chat(args.model, args.base_url)
    agent_factory = _make_agent_factory(args.model, args.base_url, args.max_iterations)

    def progress(msg: str) -> None:
        print(f"  {msg}", flush=True)

    summary = evaluate(gold, agent_factory, chat, progress)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    summary["model"] = args.model
    summary["generated_at"] = "2026-08-11"  # stamped after run; Date.now unavailable in some sandboxes
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
