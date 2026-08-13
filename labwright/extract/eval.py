"""Evaluation of the fine-tuned extractor against synthetic eval + blind golds.

Three headline metrics, all computed identically for the fine-tuned model and
for API baselines so the comparison is apples-to-apples:

- ``json_parse_rate`` — fraction of outputs that parse to a JSON object.
- ``field_recovery`` — per-key relative error of the extracted raw vs the
  gold raw (synthetic eval only; gold raws are the sampler's instance, fully
  determined by the prose). A row is "recovered" when every gold numeric key
  is within ±5 %.
- ``consistency_rate`` — fraction of rows where extract → build_design →
  verify_design yields **zero errors** (the headline: numbers follow from the
  model's own raw inputs, exactly the property the verifier enforces).

Blind golds (no stored raw) are scored by consistency only, plus a
``target_recovery`` check: does the built design land within ±20 % of the
gold's stated physiological target (e.g. hepatic sinusoidal shear 0.05 Pa)?

Usage::

    python -m labwright.extract.eval --data results/extractor \
        --out results/extractor/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

from labwright.design import DesignInput, _reject_derived_fields, build_design
from labwright.extract.data import SCHEMA_PROMPT, SYSTEM_PROMPT
from labwright.extract.pipeline import Extractor, parse_json
from labwright.verify.checker import has_errors, verify_design

_RECOVERY_TOL = 0.05
_TARGET_TOL = 0.20  # blind-gold physiology: must land in the right ballpark

_REPO = Path(__file__).resolve().parents[2]
_BLIND_GOLD = _REPO / "eval" / "gold_blind.json"


def flatten(raw: dict | None, prefix: str = "") -> dict[str, float]:
    """Flatten a nested raw block to dotted numeric keys, e.g. ``chip.width_um``."""
    out: dict[str, float] = {}
    for k, v in (raw or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def field_errors(got_raw: dict | None, gold_raw: dict | None) -> dict[str, float | None]:
    """Per-key relative error of extracted vs gold raw (None = key missing)."""
    gold, got = flatten(gold_raw), flatten(got_raw)
    errs: dict[str, float | None] = {}
    for key, value in gold.items():
        g = got.get(key)
        if g is None:
            errs[key] = None
        else:
            errs[key] = abs(g - value) / abs(value) if value != 0 else abs(g - value)
    return errs


def errors_all_within(errs: dict[str, float | None], tol: float = _RECOVERY_TOL) -> bool:
    return bool(errs) and all(e is not None and e <= tol for e in errs.values())


def build_from_raw(goal: str, raw: dict) -> tuple[object | None, list | None, str | None]:
    """Run raw through the real pipeline; return (plan, issues, error).

    The audit path crosses the same gate as the agent's ``submit_design``: a
    derived field invented by the extractor is rejected, and a malformed block
    that would raise ``TypeError`` (e.g. a duplicate keyword from a ``goal``
    key in the raw block, or ``cells.seed_count`` reaching ``CellPlan``) is a
    clean ``schema_error``, never a crash.
    """
    try:
        _reject_derived_fields(raw)
    except ValueError:
        return None, None, "derived_field_rejected"
    try:
        inp = DesignInput(goal=goal, rationale="eval", **raw)
    except Exception as exc:  # pydantic ValidationError / duplicate keyword
        return None, None, f"schema_error"
    try:
        plan = build_design(inp)
    except (ValueError, KeyError, TypeError) as exc:  # partial/typed raw block
        return None, None, f"schema_error"
    issues = verify_design(plan)
    return plan, issues, None


def score_one(goal: str, raw: dict | None, gold_raw: dict | None) -> dict:
    """Score a single extraction attempt (one row's record)."""
    rec: dict[str, Any] = {"parsed": raw is not None}
    if raw is None:
        rec.update({"schema_ok": False, "consistent": False})
        return rec
    plan, issues, err = build_from_raw(goal, raw)
    rec["schema_ok"] = plan is not None
    rec["consistent"] = plan is not None and not has_errors(issues)
    if gold_raw is not None:
        errs = field_errors(raw, gold_raw)
        rec["field_errs"] = errs
        rec["recovered"] = errors_all_within(errs)
    return rec


def _derive_shear(plan) -> float | None:
    if plan is not None and plan.derived is not None:
        return plan.derived.shear_pa
    return None


def score_batch(
    extract_fn: Callable[[str], dict | None],
    eval_rows: list[dict],
    blind_golds: list[dict],
    extract_batch_fn: Callable[[list[str]], list[dict | None]] | None = None,
    batch_size: int = 1,
) -> dict:
    """Run ``extract_fn`` over eval rows + blind golds; aggregate the report.

    With ``extract_batch_fn`` supplied, rows are extracted in left-padded chunks
    so the GPU decode is batched. Each blind gold is scored defensively — one
    row that trips an unexpected code path is recorded as a failure, never an
    abort (the run is ~45 min on the V100; it must survive a bad row).
    """
    n_parse = n_schema = n_consistent = n_recovered = 0
    errs_all: list[float] = []
    n_target = n_target_ok = 0
    n_rows = len(eval_rows)
    goals = [r["goal"] for r in eval_rows]
    if extract_batch_fn is not None and batch_size > 1:
        for start in range(0, len(eval_rows), batch_size):
            chunk = eval_rows[start:start + batch_size]
            raws = extract_batch_fn([r["goal"] for r in chunk])
            for row, raw in zip(chunk, raws):
                rec = score_one(row["goal"], raw, row["raw"])
                n_parse += int(rec["parsed"])
                n_schema += int(rec["schema_ok"])
                n_consistent += int(rec["consistent"])
                if "recovered" in rec:
                    n_recovered += int(rec["recovered"])
                    errs_all += [e for e in rec["field_errs"].values() if e is not None]
    else:
        for row in eval_rows:
            rec = score_one(row["goal"], extract_fn(row["goal"]), row["raw"])
            n_parse += int(rec["parsed"])
            n_schema += int(rec["schema_ok"])
            n_consistent += int(rec["consistent"])
            if "recovered" in rec:
                n_recovered += int(rec["recovered"])
                errs_all += [e for e in rec["field_errs"].values() if e is not None]
    for gold in blind_golds:
        try:
            raw = extract_fn(gold["goal"])
            if raw is None:
                continue
            plan, issues, _err = build_from_raw(gold["goal"], raw)
            n_parse += 1
            if plan is None:
                continue
            n_schema += 1
            n_consistent += int(not has_errors(issues))
            target = gold["expected"].get("shear_pa")
            if target is not None and plan.derived is not None:
                n_target += 1
                shear = plan.derived.shear_pa
                n_target_ok += int(abs(shear - target) <= _TARGET_TOL * abs(target))
        except Exception:
            # Never abort the whole run for one gold; count it as not-consistent.
            continue
    return {
        "n_rows": n_rows,
        "n_blind": len(blind_golds),
        "json_parse_rate": round(n_parse / max(n_rows + len(blind_golds), 1), 4),
        "schema_ok_rate": round(n_schema / max(n_parse, 1), 4),
        "consistency_rate": round(n_consistent / max(n_parse, 1), 4),
        "field_recovery_ok_rate": round(n_recovered / max(n_rows, 1), 4),
        "mean_field_rel_error": round(
            sum(errs_all) / len(errs_all), 6
        ) if errs_all else None,
        "target_recovery_rate": round(n_target_ok / max(n_target, 1), 4),
    }


def api_extract(chat: Callable[[str], str], system_prompt: str = SYSTEM_PROMPT):
    """Wrap an LLMClient.chat-style callable into an extract function.

    ``system_prompt`` defaults to the bare SYSTEM_PROMPT; API baselines pass
    :data:`SCHEMA_PROMPT` so the exact raw-input key contract is spelled out
    (the fine-tuned model learned it during training, so a bare prompt would
    test key-name guessing instead of value extraction).
    """
    def _fn(goal: str) -> dict | None:
        try:
            text = chat(system_prompt + "\n\nGoal: " + goal)
        except Exception:
            return None
        return parse_json(text or "")
    return _fn


def load_blind_golds() -> list[dict]:
    with open(_BLIND_GOLD) as fh:
        return json.load(fh)


def load_eval_rows(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    path = data_dir / "eval.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/extractor")
    parser.add_argument("--out", default="results/extractor/eval_report.json")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default results/extractor/lora)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--api", nargs="*", default=[], help="API model names to compare, e.g. --api flash pro")
    parser.add_argument("--limit", type=int, default=0, help="cap eval rows (smoke)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="left-padded GPU batch decode (e.g. 6); 1 = sequential")
    args = parser.parse_args()

    data_dir = Path(args.data)
    eval_rows = load_eval_rows(data_dir)
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    blind = load_blind_golds()
    print(f"eval rows: {len(eval_rows)}, blind golds: {len(blind)}")

    report: dict[str, Any] = {"n_eval_rows": len(eval_rows), "n_blind": len(blind), "systems": {}}

    # Fine-tuned extractor
    if Path(args.adapter or _DEFAULT_ADAPTER).exists():
        ext = Extractor(model_path=args.model, adapter_path=args.adapter)
        report["systems"]["fine-tuned-1.5B"] = score_batch(
            ext.extract, eval_rows, blind,
            extract_batch_fn=ext.extract_batch, batch_size=args.batch_size,
        )
        print("fine-tuned:", report["systems"]["fine-tuned-1.5B"])
    else:
        print(f"[warn] adapter not found at {args.adapter or _DEFAULT_ADAPTER}; skipping fine-tuned eval")

    # API baselines
    if args.api:
        from labwright.agent import LLMClient

        for name in args.api:
            client = LLMClient(model=name)
            chat = lambda prompt, _c=client: (_c.chat([{"role": "user", "content": prompt}], max_tokens=512).content or "")
            report["systems"][name] = score_batch(
                api_extract(chat, system_prompt=SCHEMA_PROMPT), eval_rows, blind)
            print(name, ":", report["systems"][name])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("report ->", out)
    return 0


_DEFAULT_ADAPTER = "results/extractor/lora"


if __name__ == "__main__":
    raise SystemExit(main())
