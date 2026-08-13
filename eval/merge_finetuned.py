"""Graft the fine-tuned extractor's rows into the main benchmark result files.

The fine-tuned extractor (``results/eval_finetuned_*.json``) is a *fixed local
model*: it does not depend on the API model, so its per-entry records carry no
model axis. The headline benchmark files are per set × model (``eval_flash.json``
and ``eval_pro.json`` for the reading set, ``eval_blind_*`` for the blind set,
``eval_spheroid_*`` for the spheroid set). To let ``eval.report.derive()`` and
``paper/fig_benchmark.py`` render the extractor as a fifth system next to the
four API systems, this script copies each finetuned per-entry record into the
matching entry of **both** model files of its set (the numbers are identical
under flash and pro by construction).

The merge is id-keyed and idempotent: re-running overwrites the finetuned key
in place. It never touches the other systems' rows.

Usage::

    python -m eval.merge_finetuned
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "results"

#: finetuned result file -> main result files to merge it into (both model
#: variants of the same set, since the extractor is model-independent).
SETS = {
    "eval_finetuned_reading.json": ("eval_flash.json", "eval_pro.json",
                                    "eval_k3.json", "eval_kimicode.json"),
    "eval_finetuned_blind.json": ("eval_blind_flash.json", "eval_blind_pro.json",
                                  "eval_blind_k3.json", "eval_blind_kimicode.json"),
    "eval_finetuned_spheroid.json": ("eval_spheroid_flash.json", "eval_spheroid_pro.json",
                                     "eval_spheroid_k3.json", "eval_spheroid_kimicode.json"),
    "eval_finetuned_culture.json": ("eval_culture_flash.json", "eval_culture_pro.json",
                                    "eval_culture_k3.json", "eval_culture_kimicode.json"),
    "eval_finetuned_pk.json": ("eval_pk_flash.json", "eval_pk_pro.json",
                               "eval_pk_k3.json", "eval_pk_kimicode.json"),
}


def main() -> int:
    for fin_name, mains in SETS.items():
        fin_path = RESULTS / fin_name
        if not fin_path.exists():
            print(f"skip {fin_name}: not present yet")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
