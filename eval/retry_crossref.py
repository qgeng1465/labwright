"""Slow, polite retry pass for Crossref titles that errored during the parallel
enrichment (mostly HTTP 429 rate-limits). Serial, one request at a time, with
backoff on 429, then the enriched dataset + provenance CSV are regenerated from
the updated cache via ``--no-live``.

Usage::

    python -m eval.retry_crossref                # retry error entries, then re-emit
    python -m eval.retry_crossref --dry-run      # only report how many to retry
"""

from __future__ import annotations

import argparse
import json
import time

from eval.enrich_scirecipe import (
    DEFAULT_CACHE,
    _best_item,
    resolve_crossref,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--max-errors", type=int, default=0, help="cap retries (smoke)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pause", type=float, default=2.0, help="base seconds between calls")
    args = ap.parse_args(argv)

    cache = json.load(open(args.cache, encoding="utf-8"))
    todo = [t for t, v in cache.items() if v.get("quality") == "error"]
    print(f"{len(todo)} error titles to retry")
    if args.max_errors:
        todo = todo[: args.max_errors]
        print(f"  capped to {len(todo)}")
    if args.dry_run:
        return 0

    fixed = still_error = 0
    for i, t in enumerate(todo, 1):
        pause = args.pause
        for attempt in range(4):
            try:
                items = resolve_crossref(t)
                best = _best_item(items, t)
                if best is None:
                    rec = {"doi": "", "score": None, "quality": "none"}
                else:
                    s = best["score"]
                    best["quality"] = (
                        "high" if s >= 0.90 else "medium" if s >= 0.60 else "none"
                    )
                    rec = best
                cache[t] = rec
                fixed += 1 if rec["quality"] in ("high", "medium", "none") else 0
                break
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc) and attempt < 3:
                    pause *= 3  # back off hard on rate-limit
                    time.sleep(pause)
                    continue
                cache[t] = {"doi": "", "score": None, "quality": "error",
                            "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
                still_error += 1
                break
        if i % 10 == 0 or i == len(todo):
            json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {i}/{len(todo)} retried; fixed so far {fixed}, still error {still_error}",
                  flush=True)
        time.sleep(args.pause)  # polite baseline between calls

    json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"done: fixed {fixed}, still error {still_error}")

    # Regenerate enriched output + provenance CSV from the updated cache.
    from eval.enrich_scirecipe import main as enrich_main
    return enrich_main(["--no-live"])


if __name__ == "__main__":
    raise SystemExit(main())
