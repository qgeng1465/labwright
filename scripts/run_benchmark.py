#!/usr/bin/env python
"""Run the Labwright vs bare-LLM benchmark.

Usage:
    python scripts/run_benchmark.py [--gold path] [--max-iterations N]

Requires LABWRIGHT_API_KEY / DEEPSEEK_API_KEY (reserved experiment window).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from eval.benchmark import evaluate, load_gold
    from labwright.agent import DesignAgent, LLMClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=None)
    parser.add_argument("--max-iterations", type=int, default=12)
    args = parser.parse_args()

    gold = load_gold(args.gold) if args.gold else load_gold()

    def chat(prompt: str) -> str:
        client = LLMClient()
        msg = client.chat([{"role": "user", "content": prompt}], tools=None, max_tokens=1024)
        return msg.content or ""

    def agent_factory():
        return DesignAgent(LLMClient(), max_iterations=args.max_iterations)

    def progress(msg: str) -> None:
        print("  " + msg, flush=True)

    print(f"Benchmarking {len(gold)} gold experiments (reserved experiment window)...")
    results = evaluate(gold, agent_factory, chat, progress)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
