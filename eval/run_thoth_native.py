"""Run Thoth in its *native* output mode and check internal consistency.

The prompt-only harness (``run_thoth.py``) forces every model through the same
JSON schema; Thoth gets a 0 % parse rate there because its training output is
protocol prose, not JSON. That comparison is a strawman: it tests whether Thoth
can translate into a foreign schema, not whether it writes self-consistent
numbers.

This script removes the strawman. We prompt Thoth exactly as it was trained
(goal → protocol with ``<think>``/``<key>``/``<orc>``/``<note>`` segments), parse
its native output, then run the *same* deterministic calculators Labwright
applies to its own designs (``verify_published_protocol``): do the derived
numbers Thoth states follow from the raw inputs Thoth states? That is the exact
property Labwright guarantees for itself — a fair, schema-free comparison.

Metrics (with Wilson CIs from ``eval.ci``):

- ``n_parse`` — protocols that emitted usable ``<key>``/``<orc>`` segments.
- ``n_with_claims`` — protocols asserting at least one unit-typed derived number.
- ``n_checkable`` — claims + recoverable raw inputs, so the calculators can run.
- ``internal_consistency_rate`` — of the *checkable* protocols, the fraction
  whose claimed derived numbers agree with the computed ones (the headline).
- ``unverifiable_rate`` — of the checkable set that could not be cross-checked.

Usage::

    python -m eval.run_thoth_native --out results/eval_thoth_native.json
        --model-dir /data/hf_models/manglu3935/Thoth
        --adapter results/extractor/lora --model Qwen/Qwen2.5-1.5B-Instruct
        --batch-size 6
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.ci import format_ci, wilson_ci
from eval.run_scirecipe_audit import audit_row, harvest_claims, route_domain
from eval.run_thoth import _build_chat
from labwright.extract.pipeline import Extractor


#: Native segment delimiters Thoth was trained to emit (SciRecipe format).
_SEGMENTS = re.compile(
    r"<(?P<tag>think|key|orc|note)>(?P<body>.*?)</(?P=tag)>", re.S | re.I
)


def parse_segments(text: str) -> dict[str, str]:
    """Pull the named segments out of Thoth's native output."""
    out: dict[str, str] = {}
    for m in _SEGMENTS.finditer(text or ""):
        out.setdefault(m.group("tag").lower(), m.group("body").strip())
    return out


def native_prompt(goal: str) -> str:
    """Mirror SciRecipe's training prompt: goal → structured protocol."""
    return (
        f"{goal}\n\n"
        "Please provide the experimental protocol to solve this problem. "
        "Use the structured format with <think>, <key>, <orc> and <note> components."
    )


def run_native_audit(
    chat: Callable[[str], str],
    extract_fn: Callable[[str], dict | None],
    gold: list,
    batch_size: int = 1,
    batch_extract_fn: Callable[[list[str]], list[dict | None]] | None = None,
) -> dict:
    """Generate + audit each gold goal; return the native-mode report."""
    rows: list[dict] = []
    n_parse = n_claims = n_checkable = n_ok = n_review = n_unverifiable = 0
    t0 = time.time()

    # Generate natively, then extract raw inputs in batches for GPU warmth
    # (mirrors the SciRecipe audit's left-padded batch decode).
    pending: list[tuple[dict, str]] = []  # (gold, generated_text)
    for g in gold:
        try:
            text = chat(native_prompt(g.goal))
        except Exception:
            text = ""
        pending.append((g, text))
    if batch_size > 1 and batch_extract_fn is not None:
        raws = batch_extract_fn(
            [parse_segments(t).get("orc") or t for _, t in pending]
        )
    else:
        raws = [extract_fn(parse_segments(t).get("orc") or t) for _, t in pending]

    for (g, text), raw in zip(pending, raws):
        segments = parse_segments(text)
        # The orc (natural-language protocol) carries the numeric assertions;
        # fall back to the whole output when the model skipped <orc>.
        orc = segments.get("orc") or text
        has_key = "key" in segments
        has_orc = "orc" in segments
        rec: dict = {
            "id": g.id,
            "has_key": has_key,
            "has_orc": has_orc,
            "n_segments": len(segments),
            "generated_len": len(text),
        }
        if not text.strip():
            rec["verdict"] = "unverifiable"
            rec["reason"] = "empty_generation"
            rows.append(rec)
            continue
        n_parse += 1

        claimed = harvest_claims(orc)
        rec["claimed"] = claimed
        if not claimed:
            rec["verdict"] = "unverifiable"
            rec["reason"] = "no_numeric_claims"
            rows.append(rec)
            continue
        n_claims += 1

        r = audit_row(orc, raw, reference=g.id)
        rec["domain"] = r["domain"]
        rec["quote"] = r["quote"]
        rec["raw"] = r.get("raw")
        rec["computed"] = r.get("computed")
        rec["discrepancy_fields"] = r.get("discrepancy_fields", [])
        rec["verdict"] = r["verdict"]
        rec["reason"] = r.get("reason")
        rows.append(rec)

    for r in rows:
        v = r["verdict"]
        if v in ("ok", "review_required"):
            n_checkable += 1
        if v == "ok":
            n_ok += 1
        elif v == "review_required":
            n_review += 1
        elif v == "unverifiable":
            n_unverifiable += 1

    verdict_counts = {
        "ok": n_ok,
        "review_required": n_review,
        "unverifiable": n_unverifiable,
    }
    consistency = n_ok / n_checkable if n_checkable else float("nan")
    ci = wilson_ci(n_ok, n_checkable) if n_checkable else (0.0, 0.0)

    return {
        "n_total": len(gold),
        "n_parse": n_parse,
        "n_with_claims": n_claims,
        "n_checkable": n_checkable,
        "verdict_counts": verdict_counts,
        "internal_consistency_rate": round(consistency, 4),
        "consistency_ci": [round(ci[0], 4), round(ci[1], 4)],
        "consistency_ci_str": format_ci(n_ok, n_checkable) if n_checkable else "n/a",
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="eval/gold_experiments.json")
    ap.add_argument("--blind", default="eval/gold_blind.json")
    ap.add_argument("--out", default="results/eval_thoth_native.json")
    ap.add_argument("--model-dir", default="/data/hf_models/manglu3935/Thoth")
    ap.add_argument("--adapter", default="results/extractor/lora")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args(argv)

    from eval.benchmark import load_gold

    gold = load_gold(args.gold)
    try:
        blind = load_gold(args.blind)
    except FileNotFoundError:
        blind = []
    print(f"thoth native audit: {len(gold)} reading + {len(blind)} blind golds", flush=True)

    chat = _build_chat(args.model_dir, args.max_new_tokens)
    ext = Extractor(model_path=args.model, adapter_path=args.adapter)

    report = run_native_audit(chat, ext.extract, gold, args.batch_size, ext.extract_batch)
    blind_report = run_native_audit(chat, ext.extract, blind, args.batch_size, ext.extract_batch)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = {
        "model": "thoth-8b (native mode)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "reading": report,
        "blind": blind_report,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print("\n=== reading set ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    print("\n=== blind set ===", flush=True)
    print(json.dumps({k: v for k, v in blind_report.items() if k != "rows"}, indent=2), flush=True)
    print(f"\nsaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
