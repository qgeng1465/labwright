"""Command-line interface.

    labwright design "Liver-chip model of DILI at sinusoidal shear" --output sop
    labwright tools            # list available calculators
"""

from __future__ import annotations

import argparse
import json
import sys

from labwright import __version__
from labwright.tools import list_tools


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="Model name (default: env LABWRIGHT_MODEL or deepseek-chat)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: env or DeepSeek)")
    p.add_argument("--api-key", help="API key (default: env LABWRIGHT_API_KEY / DEEPSEEK_API_KEY)")
    p.add_argument("--max-iterations", type=int, default=12)


def _make_agent(args):
    from labwright.agent import DesignAgent, LLMClient

    llm = LLMClient(
        api_key=getattr(args, "api_key", None),
        base_url=getattr(args, "base_url", None),
        model=getattr(args, "model", None),
    )
    return DesignAgent(llm, max_iterations=args.max_iterations)


def cmd_design(args: argparse.Namespace) -> int:
    from labwright.sop import design_to_sop

    try:
        result = _make_agent(args).run(args.goal)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print("Set LABWRIGHT_API_KEY / DEEPSEEK_API_KEY, or pass --api-key.", file=sys.stderr)
        return 2

    if result.status == "error":
        print(f"[error] {result.error}", file=sys.stderr)
        return 1

    print(result.verification_summary)
    print()
    if args.output == "json":
        print(json.dumps(result.design.model_dump(mode="json"), indent=2, ensure_ascii=False))
    elif args.output == "sop":
        print(design_to_sop(result.design))
    else:
        print(design_to_sop(result.design))
        print("\n--- design (json) ---")
        print(json.dumps(result.design.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    print(f"{'category':<14} {'tool':<28} description")
    for t in list_tools():
        print(f"{t.category:<14} {t.name:<28} {t.description[:64]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labwright", description=f"Labwright v{__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("design", help="Design an experiment from a natural-language goal")
    p.add_argument("goal", help="Experimental goal, e.g. 'liver-chip model of DILI at sinusoidal shear'")
    p.add_argument("--output", choices=["sop", "json", "all"], default="all")
    _add_llm_args(p)
    p.set_defaults(func=cmd_design)

    p = sub.add_parser("tools", help="List available calculator tools")
    p.set_defaults(func=cmd_tools)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
