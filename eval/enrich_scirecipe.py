"""Attach real-literature provenance to the SciRecipe audit.

The audit (``results/eval_scirecipe_audit.json``) classifies 5700 SciRecipe
protocol summaries, but each row carries only a truncated ``orc`` and a
``quote`` — no pointer back to the source record, and no paper reference. This
script restores that provenance:

1. Rebuild the deterministic funnel (``has_numbers`` + ``route_domain``) over
   the SciRecipe parquet and map every audited row back to its ``source_idx``
   and full ``exp_goal`` (matching on the truncated ``orc`` prefix, verified to
   be a perfect 1:1 join).
2. Harvest the *quoted protocol title* embedded in ``exp_goal`` (e.g. `In the
   "Spot Assays for Viability Analysis of Cyanobacteria" protocol, ...`).
3. Resolve each distinct title to real literature via Crossref
   (``works?query.bibliographic``), gated by a title-similarity check so a
   loose query never silently substitutes a different paper.
4. Emit the enriched audit dataset + a provenance CSV, and report the coverage
   funnel honestly — including the majority of rows whose ``exp_goal`` carries
   no quoted title (reported as a limitation, not hidden).

Honesty rules
-------------
* A DOI is attached only when the resolved title matches the quoted title to a
  high confidence (``match >= HIGH_MATCH``). ``medium`` and ``low`` matches are
  recorded for transparency but flagged; they are NOT counted as citations.
* The parity between the audited population and the *titled* subpopulation is
  stated explicitly so the checkable rate is never misread as applying to the
  whole corpus.
* No editorialising: the exp_goal title is the SciRecipe authors' own text; we
  resolve it to a DOI, we do not choose which "source paper" a row came from.

Usage::

    python -m eval.enrich_scirecipe                       # run (incremental)
    python -m eval.enrich_scirecipe --max-titles 20      # small smoke run
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from eval.run_scirecipe_audit import DEFAULT_PARQUET, has_numbers, route_domain

#: The audit output we are enriching.
DEFAULT_AUDIT = "results/eval_scirecipe_audit.json"
#: Enriched dataset output (the open-source artifact).
DEFAULT_OUT = "results/scirecipe_audit_enriched.json"
#: Incremental Crossref cache — survives reruns and mid-run restarts.
DEFAULT_CACHE = "results/scirecipe_crossref_cache.json"
#: Machine-readable provenance table for the README / dataset card.
DEFAULT_PROV = "results/scirecipe_provenance.csv"

#: Similarity at which a resolved title is counted as the same protocol.
HIGH_MATCH = 0.90
#: Minimum similarity for a ``medium`` (transparent-but-flagged) match.
MEDIUM_MATCH = 0.60

#: Anything below MEDIUM_MATCH is recorded as ``no_match``.
MAILTO = "qgeng1465@users.noreply.github.com"
USER_AGENT = f"Labwright-SciRecipe-audit/1.0 (mailto:{MAILTO})"
#: Politeness: one Crossref request per row of this many seconds.
SLEEP_S = 1.0

#: A quoted protocol title inside an exp_goal, e.g.
#: ``In the "Spot Assays for Viability Analysis of Cyanobacteria" protocol`` or
#: ``... the 'Sorghum bicolor Extracellular Vesicle Isolation' protocol ...``.
#: The straight apostrophe is a legitimately-used quote delimiter in the source
#: text ("in the 'Assessment ...' protocol"); the >=8-char body plus the goal
#: template ("In the ... protocol") keeps contraction false positives out.
_TITLE_RE = re.compile(r'["\'“”‘’]([^"\'“”‘’]{8,180})["\'“”‘’]')

_STOP = frozenset(
    "a an the in of for on with and or to from by at as into using method protocol "
    "what how steps exact please provide experimental experiment protocol".split()
)


def _normalize(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", title.lower())


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def harvest_titles(exp_goal: str) -> list[str]:
    """Quoted protocol titles embedded in an exp_goal, cleaned of trailing punct."""
    out = []
    for m in _TITLE_RE.finditer(exp_goal or ""):
        t = re.sub(r"[,\\.;:]+$", "", m.group(1)).strip()
        if len(t) >= 8 and t.lower() not in out:
            out.append(t)
    return out


def _best_item(items: list[dict], title: str) -> dict[str, Any] | None:
    """Pick the highest-similarity Crossref item; carry the score for gating."""
    best: dict[str, Any] | None = None
    for it in items:
        it_title = (it.get("title") or [""])[0]
        score = _sim(title, it_title)
        if best is None or score > best["score"]:
            best = {
                "score": round(score, 3),
                "doi": it.get("DOI", ""),
                "resolved_title": it_title[:200],
                "container": (it.get("container-title") or [""])[0][:120],
                "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
                "type": it.get("type", ""),
            }
    return best


def resolve_crossref(title: str) -> dict[str, Any] | None:
    """Resolve one title to its Crossref top match (unscored)."""
    url = (
        "https://api.crossref.org/works"
        f"?query.bibliographic={urllib.parse.quote(title)}"
        f"&rows=4&mailto={MAILTO}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]["items"]


def _fmt_cached(c: dict[str, Any]) -> dict[str, Any]:
    """Normalise a cache entry back to the live-record shape."""
    return {
        "score": c.get("score"),
        "doi": c.get("doi", ""),
        "resolved_title": c.get("resolved_title", ""),
        "container": c.get("container", ""),
        "year": c.get("year"),
        "type": c.get("type", ""),
        "quality": c.get("quality", "none"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--audit", default=DEFAULT_AUDIT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--prov", default=DEFAULT_PROV)
    ap.add_argument("--max-titles", type=int, default=0, help="cap distinct titles (smoke)")
    ap.add_argument("--no-live", action="store_true", help="cache only; no new Crossref calls")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel Crossref resolution threads (default 6)")
    args = ap.parse_args(argv)

    # -- load parquet, rebuild funnel, map orc[:220] -> (source_idx, exp_goal) --
    df = pd.read_parquet(args.parquet, columns=["exp_goal", "key", "orc", "note"])
    idx_by_prefix: dict[str, tuple[int, str, str]] = {}
    for i, row in df.iterrows():
        orc = str(row.get("orc") or "").strip()
        if not orc or not has_numbers(orc):
            continue
        if route_domain(orc) == "none":
            continue
        idx_by_prefix[orc[:220]] = (i, str(row.get("exp_goal") or "").strip(),
                                    str(row.get("note") or "").strip())

    audit = json.load(open(args.audit, encoding="utf-8"))
    rows = audit["rows"]
    missing = [r for r in rows if r["orc"] not in idx_by_prefix]
    if missing:
        print(f"WARNING: {len(missing)}/{len(rows)} audit rows unmapped (orc prefix drift)")
    else:
        print(f"mapped {len(rows)}/{len(rows)} audit rows -> parquet source_idx")

    # -- harvest titles per row --
    per_row_titles: list[list[str]] = []
    titled_rows = 0
    distinct: dict[str, None] = {}
    for r in rows:
        goal = idx_by_prefix.get(r["orc"], (None, "", ""))[1]
        ts = harvest_titles(goal)
        if ts:
            titled_rows += 1
        for t in ts:
            distinct.setdefault(t, None)
        per_row_titles.append(ts)
    print(f"rows with >=1 quoted protocol title: {titled_rows}/{len(rows)} "
          f"({titled_rows / len(rows):.1%}); distinct titles: {len(distinct)}")

    # -- incremental Crossref resolution --
    cache: dict[str, Any] = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache, encoding="utf-8"))
    todo = [t for t in distinct if t not in cache]
    if args.max_titles:
        todo = todo[: args.max_titles]
    print(f"Crossref cache: {len(cache)} titles cached, {len(todo)} to resolve"
          + ("" if not args.no_live else " (live calls disabled)"))
    if todo and not args.no_live:
        # Parallel resolution: Crossref is rate-limited politely per key, but a
        # modest thread pool (~6) with a shared lock keeps the 2400-title corpus
        # tractable while one hung request never blocks the batch.
        import threading
        from concurrent.futures import ThreadPoolExecutor

        lock = threading.Lock()
        done = [0]

        def work(t: str) -> None:
            try:
                items = resolve_crossref(t)
                best = _best_item(items, t)
                if best is None:
                    rec = {"doi": "", "score": None, "quality": "none"}
                else:
                    s = best["score"]
                    best["quality"] = ("high" if s >= HIGH_MATCH
                                       else "medium" if s >= MEDIUM_MATCH else "none")
                    rec = best
            except Exception as exc:  # noqa: BLE001 - one bad title must not kill the batch
                rec = {"doi": "", "score": None, "quality": "error",
                       "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
            with lock:
                cache[t] = rec
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(todo):
                    json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False)
                    n_hi = sum(1 for c in cache.values() if c.get("quality") == "high")
                    print(f"  resolved {done[0]}/{len(todo)}; high-confidence so far: {n_hi}",
                          flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        # Final flush (threaded saves may have missed the tail).
        json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False)

    # -- attach provenance to each row --
    n_high = n_medium = n_none = 0
    for r, ts in zip(rows, per_row_titles):
        source_idx, exp_goal, note = idx_by_prefix.get(r["orc"], (None, "", ""))
        best = None
        if ts:
            # In capped/smoke runs some titles are not yet in the cache; a row
            # only gets literature provenance when its titles were resolved.
            hits = [_fmt_cached(cache[t]) for t in ts if t in cache]
            best = max(hits, key=lambda h: h.get("score") or -1) if hits else None
            if best:
                q = best.get("quality", "none")
                if q == "high":
                    n_high += 1
                elif q == "medium":
                    n_medium += 1
                else:
                    n_none += 1
        r["source_idx"] = source_idx
        r["exp_goal"] = exp_goal
        r["quoted_titles"] = ts
        if best:
            r["literature"] = best
    print(f"rows with high-confidence DOI: {n_high}, medium: {n_medium}, none/low: {n_none}")

    # -- write enriched dataset --
    out_report = dict(audit)
    out_report["provenance"] = {
        "n_audited": len(rows),
        "n_titled": titled_rows,
        "n_distinct_titles": len(distinct),
        "n_high_doi": n_high,
        "n_medium_doi": n_medium,
        "n_none_doi": n_none,
        "high_doi_rate_over_titled": round(n_high / max(1, titled_rows), 3),
        "high_doi_rate_over_audited": round(n_high / max(1, len(rows)), 3),
        "method": "Crossref works?query.bibliographic, title-similarity gated "
                  f"(high>={HIGH_MATCH}, medium>={MEDIUM_MATCH})",
        "source_dataset": "manglu3935/SciRecipe (parquet)",
        "note": "Rows without a quoted protocol title in exp_goal have no DOI; "
                "that is a real limitation of the source text, reported not hidden.",
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out_report, fh, indent=2, ensure_ascii=False)
    print(f"saved enriched dataset -> {args.out}")

    # -- provenance CSV --
    import csv

    with open(args.prov, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_idx", "domain", "verdict", "reason", "exp_goal",
                    "quoted_title", "doi", "resolved_title", "container", "year", "type",
                    "match", "quality"])
        for r in rows:
            src, goal = r.get("source_idx"), r.get("exp_goal", "")
            lit = r.get("literature") or {}
            for t in r.get("quoted_titles", []) or [""]:
                w.writerow([src, r.get("domain"), r.get("verdict"), r.get("reason"),
                            goal[:300], t, lit.get("doi", ""), lit.get("resolved_title", ""),
                            lit.get("container", ""), lit.get("year", ""), lit.get("type", ""),
                            lit.get("score", ""), lit.get("quality", "")])
    print(f"saved provenance CSV -> {args.prov}")

    # -- verdict split within the high-confidence-DOI subpopulation --
    from collections import Counter

    hi = [r for r in rows if (r.get("literature") or {}).get("quality") == "high"]
    if hi:
        vc = Counter(r["verdict"] for r in hi)
        n_ok = vc.get("ok", 0)
        n_aud = sum(vc.get(k, 0) for k in ("ok", "review_required"))
        print(f"high-DOI subpopulation: n={len(hi)} verdicts={dict(vc)} "
              f"honest-consistent {n_ok}/{n_aud} = {n_ok / max(1, n_aud):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
