"""Synthetic (goal → raw) training/eval pairs for the extractor fine-tune.

The pairs are **self-consistent by construction**: raw inputs are sampled in
the calculators' legal ranges, every derived number is computed by the real
calculators, and the prose is rendered from the same raw. No derived number is
ever a claimed "real" reference value — unlike the source-pinned gold set in
``eval/gold_*.json``, these instances exist only to teach the raw→prose
mapping and are all reproducible from the sampler's seed.

An important coupling rule: **raw contains exactly the fields the prose states
or that are computable from the prose** (defaults and inverse-calculus
included). A field the prose never mentions is omitted from the raw too, so
the model is never asked to recover an underdetermined number — this teaches
it to *omit* rather than invent.

Prose variance comes from template choice, units (Pa vs dyn/cm²), number
formatting, an occasional neutral distractor clause, and optional fields that
appear in both prose and raw only when sampled in.

Domain split
------------
- ``flow``: a microfluidic channel. raw = {chip, flow, cells}. The prose
  states a physiological shear target (organ-pinned display values, reused
  from the benchmark gold set) and geometry; the flow rate in the raw is
  solved from the target via ``flow_rate_for_shear_stress`` — the hard,
  useful task (infer the pump setting, don't report shear).
- ``culture``: a multi-well plate. raw = {culture}. Derived seed/volume/
  confluence numbers are never in the raw.

Usage::

    python -m labwright.extract.synthetic --out results/extractor \\
        --n-flow 2500 --n-culture 1500 --split 0.9 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import microfluidics as mf
from labwright.extract.data import raw_to_json

# ---------------------------------------------------------------------------
# Organ → target-shear table. Display values are the already-pinned targets
# from eval/gold_experiments.json + eval/gold_blind.json (source DOIs in the
# gold entries); synthetic instances sample a round shear near each target so
# the prose stays defensible without inventing new physiology.
# ---------------------------------------------------------------------------
FLOW_ORGANS: list[dict] = [
    {"name": "hepatic sinusoidal", "cell": "primary human hepatocytes", "shears": [0.04, 0.05, 0.06]},
    {"name": "renal proximal tubule", "cell": "human proximal tubular epithelial cells", "shears": [0.02, 0.03, 0.04]},
    {"name": "venular", "cell": "HUVEC", "shears": [0.1, 0.2, 0.3]},
    {"name": "arterial", "cell": "human aortic endothelial cells", "shears": [1.0, 1.5, 2.0]},
    {"name": "lung alveolar", "cell": "alveolar epithelial type II cells", "shears": [0.02, 0.03, 0.05]},
    {"name": "blood-brain barrier", "cell": "human brain microvascular endothelial cells", "shears": [0.5, 1.0, 1.5]},
    {"name": "lymphatic", "cell": "lymphatic endothelial cells", "shears": [0.1, 0.2, 0.4]},
    {"name": "retinal arteriole", "cell": "human retinal endothelial cells", "shears": [3.0, 5.4, 8.0]},
]

#: Cell type → (seeding density window cells/cm², doubling time h, confluent
#: density cells/cm²). HepG2 window from Sci Rep 2021 (10.1038/s41598-021-81733-3),
#: PHH sandwich 1.5e5 from Bioengineering 2023 (10.3390/bioengineering10020131),
#: other windows are typical growth-phase/plating ranges used only to plan
#: self-consistent training instances (not claimed reference values).
_CELL_PROFILES: dict[str, tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]] = {
    "HepG2": ((2e4, 6.3e4), (30, 40), (1.5e5, 2.2e5)),
    "primary human hepatocytes": ((1.4e5, 1.6e5), None, (1.2e5, 1.5e5)),
    "HUVEC": ((4e4, 8e4), (24, 34), (1.2e5, 1.8e5)),
    "Caco-2": ((4e4, 1e5), (30, 60), (1.5e5, 2.2e5)),
    "human proximal tubular epithelial cells": ((2e4, 1e5), None, (1.0e5, 1.5e5)),
    "human aortic endothelial cells": ((4e4, 8e4), (24, 34), (1.2e5, 1.8e5)),
    "alveolar epithelial type II cells": ((5e4, 1.5e5), None, (1.0e5, 1.5e5)),
    "human brain microvascular endothelial cells": ((4e4, 8e4), (24, 34), (1.2e5, 1.8e5)),
    "lymphatic endothelial cells": ((4e4, 8e4), (24, 34), (1.2e5, 1.8e5)),
    "human retinal endothelial cells": ((4e4, 8e4), (24, 34), (1.2e5, 1.8e5)),
}

_PLATE_FORMATS = ["6", "12", "24", "48", "96"]
_PLATE_CAPACITY = {"6": 6, "12": 12, "24": 24, "48": 48, "96": 96}

#: Neutral, true-for-any-design distractor clauses. They never contradict the
#: raw inputs and never carry a number the model must reproduce.
_DISTRACTORS = [
    "Incubate at 37 °C in a 5 % CO2 humidified incubator.",
    "Sterilise the device under UV for 30 min before seeding.",
    "Pre-warm all reagents to 37 °C before use.",
    "Run the experiment in triplicate where practical.",
]


def _r(x: float, nd: int = 6) -> float:
    """Round a float to ``nd`` decimals for a clean, short JSON target."""
    return round(float(x), nd)


def _pick(rng: random.Random, seq: list) -> object:
    return seq[rng.randrange(len(seq))]


def _maybe_distractor(rng: random.Random, p: float = 0.3, out: list[str] | None = None) -> str:
    if rng.random() < p:
        d = _pick(rng, _DISTRACTORS)
        if out is not None:
            out.append(d)
        return " " + d
    return ""


# ---------------------------------------------------------------------------
# Flow domain
# ---------------------------------------------------------------------------


def _sample_flow(rng: random.Random) -> tuple[dict, float, int, int, int, float, float]:
    """Sample a flow instance; return (organ, shear_pa, w, h, L, mu, q)."""
    while True:
        organ = _pick(rng, FLOW_ORGANS)
        shear = _pick(rng, organ["shears"])
        w = rng.randint(400, 1000)
        h = _pick(rng, [50, 75, 100, 120, 150])
        if h >= w:
            continue
        L = _pick(rng, [5, 10, 20, 30])
        mu = _pick(rng, [0.0009, 0.001, 0.0011])
        q = mf.flow_rate_for_shear_stress(shear, w, h, mu)
        if 0.001 <= q <= 100:
            return organ, shear, w, h, L, mu, q


def _flow_cells(rng: random.Random, organ: dict, w: int, L: int) -> dict:
    area = calc_cell.culture_area(w, L)
    prof = _CELL_PROFILES[organ["cell"]]
    density = round(rng.uniform(*prof[0]))  # clean integer, stated verbatim
    out = {
        "cell_type": organ["cell"],
        "seeding_density_cells_cm2": float(density),
        "culture_area_cm2": _r(area, 4),
    }
    if rng.random() < 0.5 and prof[1] is not None:
        dt = rng.uniform(*prof[1])
        dur = rng.choice([24, 48, 72, 96, 120, 144])
        out["doubling_time_h"] = _r(dt, 1)
        out["culture_duration_h"] = float(dur)
    return out


def _flow_prose(
    rng: random.Random, organ: dict, shear: float, w: int, h: int, L: int, mu: float, cells: dict
) -> str:
    cell = organ["cell"]
    density = int(cells["seeding_density_cells_cm2"])
    target = f"{shear:g} Pa" if rng.random() < 0.5 else f"{shear * 10:g} dyn/cm²"
    templ = _pick(rng, _FLOW_TEMPLATES)
    prose = templ.format(
        cell=cell,
        organ=organ["name"],
        target=target,
        w=w,
        h=h,
        L=L,
        mu=mu,
        density=density,
        material="PDMS",
    )
    extras: list[str] = []
    if "doubling_time_h" in cells:
        extras.append(
            f"Cells double roughly every {cells['doubling_time_h']:g} h; "
            f"culture for {cells['culture_duration_h']:g} h."
        )
    d = _maybe_distractor(rng)
    if d:
        extras.append(d.strip())
    if extras:
        prose = prose.rstrip() + " " + " ".join(extras)
    return prose


_FLOW_TEMPLATES = [
    "Design a {organ} {material} channel {w} µm wide, {h} µm high and {L} mm long, seed {cell} "
    "at {density} cells/cm², and perfuse at a target wall shear of {target} (medium viscosity "
    "{mu} Pa·s). What flow rate (µL/min) do you set on the pump?",
    "Culture {cell} seeded at {density} cells/cm² in a {w} µm × {h} µm × {L} mm microfluidic "
    "channel and reproduce the {organ} wall shear of {target}. Choose the perfusion flow rate "
    "(medium viscosity {mu} Pa·s).",
    "A {material} organ-on-chip with a {w} µm wide, {h} µm high and {L} mm long channel is used "
    "for {cell} seeded at {density} cells/cm². The target is {target} wall shear, medium "
    "viscosity {mu} Pa·s. Report the flow rate (µL/min) to run.",
    "Perfuse {cell} (seeded at {density} cells/cm²) in a {w} × {h} µm, {L} mm long channel so "
    "that they experience the {organ} shear of {target}. Give the pump flow rate and the channel "
    "geometry (viscosity {mu} Pa·s).",
]

#: The flow-rate wording the prose uses, so a template can ask for the pump
#: setting without giving it away.
_FLOW_ASK = [
    "What flow rate (µL/min) do you set on the pump?",
    "Choose the perfusion flow rate (µL/min).",
    "Report the flow rate (µL/min) to run.",
]


def _flow_raw(organ: dict, w: int, h: int, L: int, mu: float, q: float, cells: dict) -> dict:
    return {
        "chip": {"width_um": float(w), "height_um": float(h), "length_mm": float(L),
                 "channel_count": 1, "material": "PDMS"},
        "flow": {"flow_rate_uLmin": _r(q), "viscosity_pas": float(mu), "density_kgm3": 1000.0},
        "cells": cells,
    }


def generate_flow(rng: random.Random) -> dict:
    organ, shear, w, h, L, mu, q = _sample_flow(rng)
    cells = _flow_cells(rng, organ, w, L)
    raw = _flow_raw(organ, w, h, L, mu, q, cells)
    prose = _flow_prose(rng, organ, shear, w, h, L, mu, cells)
    return {"goal": prose, "raw": raw, "domain": "flow"}


# ---------------------------------------------------------------------------
# Culture domain
# ---------------------------------------------------------------------------


def _sample_culture(rng: random.Random) -> tuple[str, int, str, float, dict, list[str]]:
    """Sample a culture instance; return (pf, wells, cell_type, density, raw, prose_parts)."""
    pf = _pick(rng, _PLATE_FORMATS)
    wells = rng.randint(1, _PLATE_CAPACITY[pf])
    cell_type = _pick(rng, list(_CELL_PROFILES))
    prof = _CELL_PROFILES[cell_type]
    density = round(rng.uniform(*prof[0]))
    out = {
        "plate_format": f"{pf}-well",
        "wells": wells,
        "cell_type": cell_type,
        "seeding_density_cells_cm2": float(density),
    }
    parts: list[str] = []
    if rng.random() < 0.5:
        v = rng.randint(78, 99)
        out["viability_pct"] = float(v)
        parts.append(f"Post-thaw viability is about {v}%.")
    if rng.random() < 0.5 and prof[1] is not None:
        clo, chi = prof[2]
        out["confluent_density_cells_cm2"] = float(round(rng.uniform(clo, chi)))
        dt = rng.uniform(*prof[1])
        dur = rng.choice([24, 48, 72, 96, 120, 144])
        out["doubling_time_h"] = _r(dt, 1)
        out["culture_duration_h"] = float(dur)
        parts.append(
            f"Confluence density is ~{out['confluent_density_cells_cm2']:.0f} cells/cm²; cells "
            f"double every ~{dt:.0f} h; harvest after {dur} h."
        )
    return pf, wells, cell_type, density, out, parts


def _culture_prose(
    rng: random.Random, pf: str, wells: int, cell_type: str, density: float, parts: list[str]
) -> str:
    templ = _pick(rng, _CULTURE_TEMPLATES)
    prose = templ.format(
        pf=f"{pf}-well",
        wells=wells,
        cell=cell_type,
        density=density,
        density_units=_pick(rng, ["cells/cm²", "cells per cm²"]),
    )
    prose = prose.rstrip()
    for part in parts:
        prose += " " + part
    prose += _maybe_distractor(rng)
    return prose


_CULTURE_TEMPLATES = [
    "Seed {cell} in a {pf} plate at {density} {density_units} across {wells} well(s).",
    "Plate {cell} onto {wells} wells of a {pf} plate at a seeding density of {density} "
    "{density_units}.",
    "{cell} are cultured in a {pf} plate, {wells} well(s), seeded at {density} "
    "{density_units}.",
    "Set up a {pf} plate with {cell} in {wells} well(s) at {density} {density_units}.",
]


def generate_culture(rng: random.Random) -> dict:
    pf, wells, cell_type, density, raw, parts = _sample_culture(rng)
    prose = _culture_prose(rng, pf, wells, cell_type, density, parts)
    return {"goal": prose, "raw": {"culture": raw}, "domain": "culture"}


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------


def generate(n_flow: int, n_culture: int, seed: int = 1234) -> list[dict]:
    rng = random.Random(seed)
    rows = [generate_flow(rng) for _ in range(n_flow)]
    rows += [generate_culture(rng) for _ in range(n_culture)]
    rng.shuffle(rows)
    return rows


def write_split(rows: list[dict], out: Path, split: float = 0.9) -> tuple[Path, Path]:
    """Split rows (already shuffled) into train/eval jsonl and write them."""
    n_train = int(len(rows) * split)
    out.mkdir(parents=True, exist_ok=True)
    train_p, eval_p = out / "train.jsonl", out / "eval.jsonl"
    for path, part in ((train_p, rows[:n_train]), (eval_p, rows[n_train:])):
        with open(path, "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_p, eval_p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/extractor", help="output directory")
    parser.add_argument("--n-flow", type=int, default=2500)
    parser.add_argument("--n-culture", type=int, default=1500)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", type=int, default=0, help="generate N of each instead")
    args = parser.parse_args()

    if args.smoke:
        nf = nc = args.smoke
    else:
        nf, nc = args.n_flow, args.n_culture
    rows = generate(nf, nc, seed=args.seed)
    train_p, eval_p = write_split(rows, Path(args.out), args.split)
    print(f"wrote {len(rows)} rows ({nf} flow + {nc} culture) -> {train_p} / {eval_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
