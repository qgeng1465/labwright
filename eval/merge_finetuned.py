"""Graft the fine-tuned extractor's rows into the main benchmark result files.

The fine-tuned extractor (``results/eval_finetuned_*.json``) is a *fixed local
model*: it does not depend on the API model, so its per-entry records carry no
model axis. The headline benchmark files are per set x model (``eval_flash.json``
and ``eval_pro.json`` for the reading set, ``eval_blind_*`` for the blind set,
``eval_spheroid_*`` for the spheroid set). To let ``eval.report.derive()`` and
``paper/fig_benchmark.py`` render the extractor as a fifth system next to the
four API systems, this script copies each finetuned per-entry record into the
matching entry of **both** model files of its set (the numbers are identical
under flash and pro by construction).

The merge is id-keyed and idempotent: re-running overwrites the finetuned key
in place. It never touches the other systems' rows.

The extractor is trained in a few labelled versions (``_lora_v2``, ``_lora_v3``,
``_lora_v4``, ...). Each version is benchmarked to
``eval_finetuned_<set>_<adapter>.json`` and merged with ``--adapter <adapter>``.
The default adapter is the one the current figures were generated from, and
**re-running with the default must be idempotent**: the unversioned
``eval_finetuned_<set>.json`` files are refreshed from the chosen adapter's rows
so a later bare re-run can never silently regress the figures to an old
adapter's data. Use ``--adapter`` after a new training run, e.g.
``python -m eval.merge_finetuned --adapter lora_v4``.

Usage::

    python -m eval.merge_finetuned                      # merge default adapter
    python -m eval.merge_finetuned --adapter lora_v4    # merge a new training run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "results"

#: set name -> main result files to merge it into (both model variants of the
#: same set, since the extractor is model-independent). The finetuned input is
#: ``eval_finetuned_<set>_<adapter>.json``.
SETS = {
    "reading": ("eval_flash.json", "eval_pro.json",
                "eval_k3.json", "eval_kimicode.json"),
    "blind": ("eval_blind_flash.json", "eval_blind_pro.json",
              "eval_blind_k3.json", "eval_blind_kimicode.json"),
    "spheroid": ("eval_spheroid_flash.json", "eval_spheroid_pro.json",
                 "eval_spheroid_k3.json", "eval_spheroid_kimicode.json"),
    "culture": ("eval_culture_flash.json", "eval_culture_pro.json",
                "eval_culture_k3.json", "eval_culture_kimicode.json"),
    "pk": ("eval_pk_flash.json", "eval_pk_pro.json",
           "eval_pk_k3.json", "eval_pk_kimicode.json"),
    "newdomains": ("eval_new_domains_labwright_flash.json",
                   "eval_new_domains_labwright_pro.json"),
}

DEFAULT_ADAPTER = "lora_v3"


def _finetuned_paths(set_name: str, adapter: str) -> tuple[Path, Path]:
    """Versioned and unversioned finetuned files for a set.

    The versioned file (``<set>_<adapter>.json``) is the source of truth for a
    given adapter. The unversioned file (``<set>.json``) is legacy: early runs
    wrote it and the merge used to read only it, so after an adapter retrain it
    held stale rows and a bare re-run regressed the figures. We keep it in sync
    so it never lies.
    """
    base = RESULTS / f"eval_finetuned_{set_name}"
    return RESULTS / f"{base}_{adapter}.json", base.with_suffix(".json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help=f"adapter tag to merge (default: {DEFAULT_ADAPTER})")
    args = ap.parse_args(argv)

    for set_name, mains in SETS.items():
        fin_versioned, fin_unversioned = _finetuned_paths(set_name, args.adapter)
        if fin_versioned.exists():
            fin_path = fin_versioned
        elif fin_unversioned.exists():
            fin_path = fin_unversioned
            print(f"warn: {fin_versioned.name} missing; "
                  f"falling back to unversioned {fin_unversioned.name} "
                  f"(re-run with --adapter {args.adapter} after the run writes it)")
        else:
            print(f"skip {set_name}: no finetuned result for adapter {args.adapter!r}")
            continue

        with open(fin_path) as fh:
            fin = json.load(fh)
        fin_rows = {e["id"]: e.get("finetuned") for e in fin["per_entry"]}

        for main_name in mains:
            main_path = RESULTS / main_name
            if not main_path.exists():
                print(f"skip {main_name}: not present yet")
                continue
            with open(main_path) as fh:
                main = json.load(fh)
            n_added = 0
            for e in main["per_entry"]:
                row = fin_rows.get(e["id"])
                if row is None:
                    continue
                e["finetuned"] = row  # id-keyed, idempotent overwrite
                n_added += 1
            with open(main_path, "w", encoding="utf-8") as fh:
                json.dump(main, fh, indent=2, ensure_ascii=False)
            print(f"{main_name}: merged finetuned rows for {n_added}/{len(main['per_entry'])} entries")

        # Keep the unversioned file in sync with the adapter's rows so a later
        # bare run cannot regress the figures to an old adapter.
        with open(fin_unversioned, "w", encoding="utf-8") as fh:
            json.dump(fin, fh, indent=2, ensure_ascii=False)
        print(f"{fin_unversioned.name}: refreshed from {fin_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
