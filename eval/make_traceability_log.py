"""Build the supplementary traceability log from committed benchmark results.

Every derived number Labwright produces is traced back to its computation: the
formula, each input value with its unit, the verifier's verdict, the code
version, and the exact sequence of tools the agent called. That record is what
this script assembles into ``supplementary/traceability/`` — one JSON per
(entry, system) that carries the full plan and its provenance, plus an
``INDEX.json`` and a human-readable ``README.md``.

The input must be a committed results file written by ``eval.run_benchmark``
(v0.7+), i.e. the labwright records carry a full ``plan`` dict and a
``provenance`` list. Older results (``plan: bool``) are skipped and counted so
the coverage number is honest.

Usage::

    python -m eval.make_traceability_log \
        --results results/eval_labmath_flash.json \
        --out supplementary/traceability
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

#: Systems whose records carry a design plan + provenance (the verified paths).
DESIGN_SYSTEMS = ("labwright", "labwright_iter", "tool_no_gate", "finetuned")


def iter_design_records(results: dict):
    """Yield (entry, system, record) for every design-path record in results."""
    for entry in results.get("per_entry", []):
        entry_id = entry.get("id")
        goal = (entry.get("gold") or {}).get("goal", "")
        for sysname in DESIGN_SYSTEMS:
            rec = entry.get(sysname)
            if not isinstance(rec, dict):
                continue
            yield entry_id, goal, sysname, rec


def build(results: dict, out: Path) -> dict:
    """Write the per-entry traceability JSONs; return the aggregate summary."""
    out = Path(out)
    model = results.get("model", "unknown")
    model_dir = out / model
    model_dir.mkdir(parents=True, exist_ok=True)

    covered = 0          # entries with a plan + provenance
    with_plan_no_prov = 0
    no_plan = 0
    prov_fields: Counter = Counter()
    n_prov_records = 0
    tool_counts: Counter = Counter()
    per_system: Counter = Counter()
    status_counts: Counter = Counter()

    for entry_id, goal, sysname, rec in iter_design_records(results):
        plan = rec.get("plan")
        provenance = rec.get("provenance") or []
        if not isinstance(plan, dict) or not provenance:
            if not isinstance(plan, dict):
                no_plan += 1
            else:
                with_plan_no_prov += 1
            continue

        covered += 1
        per_system[sysname] += 1
        for p in provenance:
            n_prov_records += 1
            prov_fields[p.get("field", "")] += 1
            status_counts[p.get("status", "ok")] += 1
        for tool in (rec.get("tool_trace") or []):
            tool_counts[tool] += 1

        (model_dir / f"{entry_id}__{sysname}.json").write_text(
            json.dumps({
                "entry_id": entry_id,
                "goal": goal,
                "system": sysname,
                "model": model,
                "verdict": {
                    "valid": rec.get("valid"),
                    "recovery": rec.get("recovery"),
                    "hallucination_rate": rec.get("hallucination_rate"),
                    "failure": rec.get("failure"),
                },
                "tool_trace": rec.get("tool_trace") or [],
                "plan": plan,
                "provenance": provenance,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "source_models": [model],
        "entries_with_provenance": covered,
        "entries_no_plan": no_plan,
        "entries_plan_without_prov": with_plan_no_prov,
        "provenance_records": n_prov_records,
        "derived_fields_covered": len(prov_fields),
        "field_record_counts": dict(prov_fields.most_common()),
        "per_system_entries": dict(per_system),
        "status_counts": dict(status_counts),
        "tool_usage": dict(tool_counts.most_common()),
    }
    (out / "INDEX.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_readme(out: Path, summaries: list[dict]) -> None:
    """A short human-facing README next to the log (all numbers from summaries)."""
    out = Path(out)
    total = sum(s["entries_with_provenance"] for s in summaries)
    fields = set()
    for s in summaries:
        fields.update(s["field_record_counts"])
    tools = Counter()
    for s in summaries:
        tools.update(s["tool_usage"])
    top_tools = " · ".join(f"{k} {v}" for k, v in tools.most_common(8)) or "—"
    (out / "README.md").write_text(
        "# Supplementary traceability log\n\n"
        "Every JSON in this directory is one benchmark entry × one design-path "
        "system (labwright / labwright_iter / tool_no_gate / finetuned), rebuilt "
        "from the committed `results/eval_labmath_*.json` files by "
        "`python -m eval.make_traceability_log`. Each file carries the full "
        "DesignPlan JSON the agent produced, its computation provenance (one "
        "record per derived field: formula, inputs with units, value, unit, "
        "verifier status, code version) and the ordered tool-call trace.\n\n"
        "Coverage: "
        f"**{total}** entries with a plan + provenance; "
        f"**{len(fields)}** distinct derived fields; "
        f"{len(summaries)} model file(s).\n\n"
        "Most-used tools across the traced entries:\n"
        f"    {top_tools}\n\n"
        "`INDEX.json` holds the full aggregate. The number on every edge of "
        "`paper/fig_protocol_dag.py` and every value in the LabMath-Bench tables "
        "can be traced to a record here.\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True,
                    help="committed results/*.json files to trace")
    ap.add_argument("--out", default="supplementary/traceability")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in args.results:
        results = json.loads(Path(path).read_text(encoding="utf-8"))
        s = build(results, out)
        summaries.append(s)
        print(f"{path}: {s['entries_with_provenance']} traced entries, "
              f"{s['provenance_records']} provenance records, "
              f"{s['derived_fields_covered']} derived fields "
              f"(no-plan: {s['entries_no_plan']})")
    write_readme(out, summaries)
    print(f"wrote traceability log to {out}/ (INDEX.json + README.md + per-entry JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
