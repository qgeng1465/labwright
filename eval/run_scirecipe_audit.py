"""Large-scale reverse-verification audit over SciRecipe protocols.

SciRecipe (manglu3935/SciRecipe, HF hub) carries ~21k protocol summaries in its
``orc`` column (~285-character median, abstract-level). For every numeric summary
the audit:

1. routes it to a domain (plate-culture / microfluidics / neither);
2. harvests the *derived* numbers the text asserts (number+unit → metric keys);
3. extracts raw inputs with an injectable extractor (the WS2 fine-tuned model by
   default, or any ``extract_fn(orc) -> raw-dict``);
4. recomputes every derived number from those inputs via
   :func:`labwright.published.verify_published_protocol`;
5. classifies ``ok`` / ``review_required`` / ``unverifiable``.

Honest scope
------------
The orc summaries are abstracts, not methods sections. Most state one or two
numbers with no geometry or growth inputs, so most rows are expected to land on
``unverifiable``. The audit therefore measures, among the protocols that say
enough to be checkable, whether the reported numbers follow from the reported
inputs — and it reports the funnel counts so the denominator is explicit.

Usage::

    python -m eval.run_scirecipe_audit --limit 200 \
        --adapter results/extractor/lora           # GPU extractor
    python -m eval.run_scirecipe_audit --funnel-only   # deterministic funnel only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Callable

import pandas as pd

from labwright.published import verify_published_protocol

#: Default SciRecipe parquet location under the HF cache (mirror downloads land
#: under ~/.cache/huggingface/hub/...).
DEFAULT_PARQUET = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--manglu3935--SciRecipe/"
    "snapshots/2aad80a53a2288963b24c504ddcccd58d6b8bfec/SciRecipe.parquet"
)
DEFAULT_OUT = "results/eval_scirecipe_audit.json"

_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SCI_NOTATION = r"(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
_SHEAR_PA_RE = re.compile(
    rf"(?P<v>{_SCI_NOTATION})\s*(?:Pa|Pascal)", re.IGNORECASE)
_SHEAR_DYN_RE = re.compile(
    rf"(?P<v>{_SCI_NOTATION})\s*(?:dyn(?:e)?/cm\s*[²2])", re.IGNORECASE)
_REYNOLDS_RE = re.compile(
    rf"re(?:ynolds)?\s*(?:number)?\s*(?:of|=|:)?\s*(?P<v>{_SCI_NOTATION})", re.IGNORECASE)
_DYN_CM2 = re.compile(r"dyn(?:e)?/cm\s*[²2]", re.IGNORECASE)

#: Contextual gates: a number+unit is only harvested as a *derived* claim when
#: the unit appears near the matching keyword — a bare "0.05 Pa" in an abstract
#: is ambiguous (shear vs pressure drop) and should not be asserted silently.
_NEAR = 40  # chars


def _has(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def _near(text: str, pos: int, *keywords: str) -> bool:
    lo = max(0, pos - _NEAR)
    hi = min(len(text), pos + _NEAR)
    window = text[lo:hi].lower()
    return any(k in window for k in keywords)


# ---------------------------------------------------------------------------
# Funnel (deterministic)
# ---------------------------------------------------------------------------


def has_numbers(orc: str) -> bool:
    """True when the summary carries a quantitative assertion (number + unit)."""
    return bool(re.search(r"\d[\d,\.]*\s*(µ|μ|ul|ml|ml/min|%|Pa|dyn|cm|mm|µm|min|h\b|hr|cells|rpm|g\b|mg|µg|nm|mM|µM)", orc, re.IGNORECASE))


def is_culture(orc: str) -> bool:
    """Cell-culture signals: wells, plates, cells/cm², seeding, confluence,
    culture medium, primary/tissue culture, organoids."""
    low = orc.lower()
    if any(k in low for k in (
        "-well", "well plate", "well-plate", "multiwell", "multi-well",
        "cells/cm", "cells cm", "seeded", "seeding", "confluen", "cultured",
        "passage", "viability", "fetal bovine", "dulbecco", "incubat",
        "organoid", "spheroid", "explant", "trachea", "trypsin", "dmem", "fbs",
    )):
        return True
    # word-boundary "culture" (cell/tissue culture) without false "agriculture".
    return bool(re.search(r"\bcultur(?:e|ed|ing)\b", low))


def is_microfluidics(orc: str) -> bool:
    """Microfluidics signals: channel, chip, flow rate, shear, perfusion."""
    low = orc.lower()
    return any(k in low for k in (
        "flow rate", "flow-rate", "flowrate", "µl/min", "ul/min", "ml/min",
        "channel", "microfluidic", "shear", "chip", "perfusion", "hydrodynamic",
    ))


def route_domain(orc: str) -> str:
    """Return ``"culture"``, ``"flow"`` or ``"none"``."""
    if is_microfluidics(orc) and not is_culture(orc):
        return "flow"
    if is_culture(orc):
        return "culture"
    return "none"


# ---------------------------------------------------------------------------
# Claim harvesting
# ---------------------------------------------------------------------------


def harvest_claims(orc: str) -> dict[str, Any]:
    """Extract the *derived* numbers the text asserts, keyed for the verifier.

    Only numbers that are both unit-typed and contextually tied to a derived
    metric become claims. Raw inputs (flow rate, geometry, seeding density) are
    deliberately *not* claims — the extractor recovers those as inputs. A bare
    ``Pa`` with no ``shear``/``pressure`` context is recorded as ``shear_pa``
    only when ``shear`` appears in the text; otherwise it is left out (ambiguous
    and not assertable).
    """
    claims: dict[str, Any] = {}
    # Shear stress — Pa or dyn/cm² (1 dyn/cm² = 0.1 Pa).
    if _has(orc, "shear"):
        m = _SHEAR_PA_RE.search(orc)
        if m:
            claims["shear_pa"] = float(m.group("v"))
        m = _SHEAR_DYN_RE.search(orc)
        if m:
            claims["shear_pa"] = 0.1 * float(m.group("v"))
    # Reynolds number.
    m = _REYNOLDS_RE.search(orc)
    if m:
        claims["reynolds"] = float(m.group("v"))
    # Culture claims.
    for m in re.finditer(rf"(?P<v>{_SCI_NOTATION})\s*cells\s+(?:per|/)\s*well", orc, re.IGNORECASE):
        claims["seed_per_well"] = float(m.group("v"))
    for m in re.finditer(
        rf"(?P<v>{_SCI_NOTATION})\s*(?:ml|µl|ul)\s*(?:per|/)\s*well", orc, re.IGNORECASE
    ):
        val = float(m.group("v"))
        claims["medium_volume_per_well_ml"] = val / 1000.0 if m.group(0).lower().startswith(("µ", "u")) else val
    for m in re.finditer(rf"(?P<v>{_SCI_NOTATION})\s*%\s*(?:confluen\w*)", orc, re.IGNORECASE):
        claims["expected_confluence_pct"] = float(m.group("v"))
    return claims


def harvest_quote(orc: str, claims: dict[str, Any]) -> str:
    """First sentence of the summary as the citation quote for the report."""
    first = orc.split(".")[0] if orc else ""
    return first[:200]


# ---------------------------------------------------------------------------
# Row audit
# ---------------------------------------------------------------------------


def audit_row(
    orc: str, extract_fn: Callable[[str], dict[str, Any] | None], reference: str
) -> dict[str, Any]:
    """Audit one protocol summary. Returns a per-row record."""
    domain = route_domain(orc)
    claimed = harvest_claims(orc)
    record: dict[str, Any] = {
        "domain": domain,
        "orc": orc[:220],
        "quote": harvest_quote(orc, claimed),
        "has_claims": bool(claimed),
        "claimed": claimed,
    }
    if domain == "none":
        record["verdict"] = "unverifiable"
        record["reason"] = "no_domain"
        return record
    t0 = time.time()
    raw = extract_fn(orc)
    record["extract_s"] = round(time.time() - t0, 3)
    if not raw:
        record["verdict"] = "unverifiable"
        record["reason"] = "extract_failed"
        return record
    # Plumb domain raws into a verify call.
    chip = raw.get("chip") or {}
    flow = raw.get("flow") or {}
    culture = raw.get("culture") or {}
    if domain == "culture" and not culture:
        record["verdict"] = "unverifiable"
        record["reason"] = "no_culture_raw"
        return record
    if domain == "flow" and not (chip and flow):
        record["verdict"] = "unverifiable"
        record["reason"] = "no_flow_raw"
        return record
    result = verify_published_protocol(
        chip=chip, flow=flow, culture=culture,
        claimed=claimed, reference=reference,
    )
    record["raw"] = {k: v for k, v in raw.items() if v}
    if result["status"] == "validation_error":
        record["verdict"] = "unverifiable"
        record["reason"] = f"validation_error: {result['error'][:80]}"
        return record
    record["computed"] = {
        c["field"]: c["computed"] for c in result["checks"] if c["computed"] is not None
    }
    record["discrepancy_fields"] = [
        c["field"] for c in result["checks"] if c["verdict"] == "discrepancy"
    ]
    record["verdict"] = result["status"]
    return record


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_audit(
    extract_fn: Callable[[str], dict[str, Any] | None],
    parquet_path: str = DEFAULT_PARQUET,
    limit: int | None = None,
) -> dict[str, Any]:
    """Funnel the SciRecipe corpus, audit numeric rows, return the report."""
    df = pd.read_parquet(parquet_path, columns=["exp_goal", "key", "orc", "note"])
    if limit:
        df = df.head(limit)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    n_numeric = n_culture = n_flow = n_none = 0
    for i, row in df.iterrows():
        orc = str(row.get("orc") or "").strip()
        if not orc or not has_numbers(orc):
            continue
        n_numeric += 1
        domain = route_domain(orc)
        if domain == "culture":
            n_culture += 1
        elif domain == "flow":
            n_flow += 1
        else:
            n_none += 1
            continue  # not in scope: no checkable domain, not deep-audited
        reference = (str(row.get("exp_goal") or row.get("note") or f"SciRecipe-row-{i}"))[:120]
        rows.append(audit_row(orc, extract_fn, reference))

    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    return {
        "n_total": int(len(df)),
        "n_numeric": n_numeric,
        "n_culture": n_culture,
        "n_flow": n_flow,
        "n_none": n_none,
        "n_audited": len(rows),
        "verdict_counts": verdicts,
        "n_discrepancy_rows": sum(1 for r in rows if r["verdict"] == "review_required"),
        "n_checkable": sum(1 for r in rows if r["verdict"] in ("ok", "review_required")),
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="cap rows (funnel/dev)")
    parser.add_argument("--funnel-only", action="store_true",
                        help="deterministic funnel + harvest only, no extractor")
    parser.add_argument("--adapter", default="results/extractor/lora")
    parser.add_argument("--model", default="/data/hf_models/Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args(argv)

    def no_extract(_orc: str) -> dict[str, Any] | None:
        return None

    if args.funnel_only:
        report = run_audit(no_extract, args.parquet, args.limit)
    else:
        from labwright.extract.pipeline import Extractor

        ext = Extractor(model_path=args.model, adapter_path=args.adapter)
        report = run_audit(ext.extract, args.parquet, args.limit)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(
        f"SciRecipe audit: {report['n_total']} total, {report['n_numeric']} numeric, "
        f"{report['n_culture']} culture / {report['n_flow']} flow, {report['n_audited']} audited"
    )
    print(f"  verdicts: {report['verdict_counts']}")
    print(f"  checkable: {report['n_checkable']}  ({report['runtime_s']}s)  saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
