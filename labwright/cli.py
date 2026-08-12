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


def cmd_verify_protocol(args: argparse.Namespace) -> int:
    """Recompute a paper's claimed numbers from a JSON file.

    The file must contain ``chip``, ``flow``, ``claimed`` and ``reference``,
    e.g.::

        {
          "chip": {"width_um": 800, "height_um": 100, "length_mm": 20},
          "flow": {"flow_rate_uLmin": 8, "viscosity_pas": 0.001},
          "claimed": {"shear_pa": 0.05, "channel_volume_ul": 0.16},
          "reference": "10.xxxx/journal.yyyy"
        }
    """
    from labwright.published import verify_published_protocol

    try:
        with open(args.file, encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        print(f"[error] cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[error] invalid JSON in {args.file}: {exc}", file=sys.stderr)
        return 2

    required = {"chip", "flow", "claimed", "reference"}
    missing = required - set(payload)
    if missing:
        print(f"[error] missing keys {sorted(missing)} in {args.file}", file=sys.stderr)
        return 2

    result = verify_published_protocol(
        chip=payload["chip"], flow=payload["flow"], claimed=payload["claimed"], reference=payload["reference"]
    )
    if result["status"] == "validation_error":
        print(f"[error] {result['error']}", file=sys.stderr)
        return 1

    print(f"reference : {result['reference']}  (tolerance ±{result['tolerance_pct']:.0f}%)")
    print(f"{'field':<22} {'computed':>12} {'claimed':>12} {'rel.err':>9}  verdict")
    for c in result["checks"]:
        claimed = "—" if c["claimed"] is None else f"{c['claimed']:.6g}"
        rel = "—" if c["relative_error"] is None else f"{c['relative_error']:.3f}"
        print(f"{c['field']:<22} {c['computed']:>12.6g} {claimed:>12} {rel:>9}  {c['verdict']}")
    if result["n_discrepancies"]:
        print(f"\n{result['n_discrepancies']} claimed value(s) do not follow from the reported inputs.")
        return 0
    print("\nAll claimed values are internally consistent.")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    print(f"{'category':<14} {'tool':<28} description")
    for t in list_tools():
        print(f"{t.category:<14} {t.name:<28} {t.description[:64]}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Extract raw inputs from a goal prose with the fine-tuned extractor, then
    derive and verify. This is the local, calculator-backed fast path."""
    from labwright.extract.pipeline import Extractor, format_audit
    from labwright.verify.checker import has_errors

    try:
        ext = Extractor(model_path=args.model, adapter_path=args.adapter)
    except Exception as exc:  # noqa: BLE001 - surface model/adapter load failure
        print(f"[error] cannot load extractor: {exc}", file=sys.stderr)
        return 2
    plan, issues, error = ext.extract_plan(args.goal)
    print(format_audit(args.goal, plan, issues, error))
    if error is not None or (issues is not None and has_errors(issues)):
        return 1
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

    p = sub.add_parser(
        "audit",
        help="Extract raw inputs from a goal with the fine-tuned extractor, derive and verify",
    )
    p.add_argument("goal", help="Experimental goal prose")
    p.add_argument("--model", default="/data/hf_models/Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="results/extractor/lora")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser(
        "verify-protocol",
        help="Recompute a published protocol's claimed numbers and flag inconsistencies",
    )
    p.add_argument("file", help="JSON file with {chip, flow, claimed, reference}")
    p.set_defaults(func=cmd_verify_protocol)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
