"""Multi-seed benchmark rerun with honest Wilson confidence intervals.

The paper's Table-1-style numbers are single runs. The mentor's experiment ②
asks for 5-seed statistics so the proportions stop reading as exact: every rate
gets a Wilson score interval (``eval/ci.py``), including the honest 0/100 %
cases that naive ``k/n`` collapses to a false 0-width.

Run::

    DEEPSEEK_API_KEY="$(cat /home/qiushuogeng/deepseek_key.txt)" \
        .venv/bin/python -m eval.run_seed_benchmark \
        --seeds 5 --systems bare,soft_gate,self_verify,labwright \
        --models deepseek-v4-flash,deepseek-v4-pro \
        --out results/eval_seed_benchmark.json

Output aggregates successes/trials per system across seeds and reports, for
each system × model: ``usable_design_rate`` and ``self_consistent_rate`` with
Wilson intervals, plus the pooled n.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import evaluate, load_gold
from eval.ci import format_ci, wilson_ci
from labwright.agent import DesignAgent, LLMClient


def _make_chat(model: str):
    client = LLMClient(model=model)

    def chat(prompt: str) -> str:
        return client.chat([{"role": "user", "content": prompt}], max_tokens=8192).content

    return chat


def _make_agent_factory(model: str):
    def factory() -> DesignAgent:
        return DesignAgent(LLMClient(model=model), max_iterations=12)

    return factory


def _pool(seed_runs: list[dict], systems: tuple) -> dict:
    """Pool per-entry valid flags across seeds into successes/trials per system.

    Every gold×system×seed is one Bernoulli trial; the Wilson interval on the
    pooled successes/trials is the honest headline rate. Pooling across seeds
    (rather than averaging per-seed rates) keeps the interval well-defined when
    a system is 0/N or N/N in some seeds.
    """
    pooled: dict = {}
    for name in systems:
        ok = consistent = trials = 0
        for summary in seed_runs:
            for entry in summary.get("per_entry", []):
                rec = entry.get(name)
                if rec is None:
                    continue
                trials += 1
                if rec.get("valid"):
                    ok += 1
                if rec.get("hallucination_rate") == 0.0:
                    consistent += 1
        usable = ok / trials if trials else float("nan")
        consistent_rate = consistent / trials if trials else float("nan")
        pooled[name] = {
            "trials": trials,
            "usable_ok": ok,
            "usable_design_rate": round(usable, 4),
            "usable_ci_str": format_ci(ok, trials) if trials else "n/a",
            "self_consistent_ok": consistent,
            "self_consistent_rate": round(consistent_rate, 4),
            "self_consistent_ci_str": format_ci(consistent, trials) if trials else "n/a",
        }
    return pooled


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="eval/gold_experiments.json")
    ap.add_argument("--systems", default="bare,labwright")
    ap.add_argument("--models", default="deepseek-v4-flash")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/eval_seed_benchmark.json")
    args = ap.parse_args(argv)

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    gold = load_gold(args.gold)
    print(f"gold: {len(gold)}   seeds: {args.seeds}   models: {models}   systems: {systems}", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    results: dict = {
        "gold": args.gold,
        "seeds": args.seeds,
        "systems": list(systems),
        "models": models,
        "per_seed": {},
        "pooled": {},
    }

    for model in models:
        chat = _make_chat(model)
        agent_factory = _make_agent_factory(model)
        seed_runs: list[dict] = []
        for seed in range(args.seeds):
            print(f"[{model}] seed {seed}/{args.seeds} ...", flush=True)
            summary = evaluate(
                gold, agent_factory, chat,
                progress=lambda msg: print("  " + msg, flush=True),
                systems=systems,
            )
            seed_runs.append(summary)
            results["per_seed"][f"{model}#{seed}"] = summary
            # Checkpoint after every seed so a mid-run failure never loses spend.
            results["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
        results["pooled"][model] = _pool(seed_runs, systems)

    results["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("\n=== pooled (Wilson CI) ===", flush=True)
    for model in models:
        print(f"--- {model} ---", flush=True)
        print(json.dumps(results["pooled"][model], indent=2), flush=True)
    print(f"\nsaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
