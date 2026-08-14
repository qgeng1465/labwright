"""Supervised (goal → raw) pairs built from the source-pinned gold set.

The gold files (``eval/gold_experiments.json``, ``eval/gold_cell_culture.json``,
``eval/gold_spheroid.json``, ``eval/gold_pk.json``) store a goal and the
*expected derived numbers* — not the raw inputs. This module instantiates each
reading gold as one concrete raw input block that, when run through the
Labwright pipeline, reproduces the gold's expected numbers. Blind golds are
deliberately **not** used here: they are reserved for self-consistency-only
eval (no stored raw, and no leakage into supervision).

Coupling rule (same as :mod:`labwright.extract.synthetic`): raw contains only
fields the goal prose states or that are computable from it. Some golds leave
a raw input implicit — e.g. the spheroid golds that only ask the standard
working volume name no cell count or size. For those a missing raw input is
appended to the prose (a training-instance augmentation; the gold entry itself
is unchanged), pinned to the same physiology the goal rests on — the 1000-cell,
20 µm anchor that reproduces the gold's own numbers.

Golds excluded with reason: the culture ``plate-hemocytometer-seed-96well`` /
``plate-thaw-viability-6well`` answer (``wells``) is derived from intermediate
hemocytometer/viability math the plate-culture raw block does not carry;
``spheroid-count-from-suspension``, ``spheroid-um-mm-unit-ambiguity`` and the
pk ``repeat-dose-24h`` / ``cell-free-subtraction`` / high- and low-extraction
golds answer a derived quantity that no single raw block reproduces; the pk
``mass-cleared`` / ``complete-clearance-panel`` golds assert MW 464 for an
unnamed drug, and no real compound at 464 g/mol can be pinned without
inventing a fact; ``spheroid-doxorubicin-dosing`` is cross-domain (spheroid +
dosing) and the dosing block is out of scope. Compound-less pk golds get a real
probe compound (warfarin) appended to the prose, and the ``dmso-at-guideline-
1mM`` goal gets a compound + real MW (same augmentation rule).
"""

from __future__ import annotations

import json
from pathlib import Path

from labwright.calc import microfluidics as mf

_REPO = Path(__file__).resolve().parents[2]
_GOLD_FLOW = _REPO / "eval" / "gold_experiments.json"
_GOLD_CULTURE = _REPO / "eval" / "gold_cell_culture.json"
_GOLD_SPHEROID = _REPO / "eval" / "gold_spheroid.json"
_GOLD_PK = _REPO / "eval" / "gold_pk.json"

#: Dosing golds whose prose names a real compound, with its standard MW.
_DOSE_APAP_MW = 151.2
_DOSE_EXTRA = {
    "dmso-at-guideline-1mM": "Use warfarin (MW 308.3 g/mol) as the compound.",
}

#: Golds whose answer is derived through a tool chain (hemocytometer/vial
#: count) the plate-culture raw block does not carry — excluded with reason.
_EXCLUDED_CULTURE = {
    "plate-hemocytometer-seed-96well": "wells is derived from hemocytometer counting",
    "plate-thaw-viability-6well": "wells is derived from thaw viability",
}

#: Prose augmentations that pin the raw inputs a gold's goal leaves implicit
#: (training-instance augmentation; the gold entries are unchanged). The pinned
#: values are source-pinned to the same physiology the goal rests on — e.g. the
#: spheroid volume/working-volume golds get the 1000-cell, 20 µm anchor that
#: reproduces their own expected numbers (gold_spheroid.json: 1000 cells of
#: 20 µm ≈ a 200 µm spheroid), and the PK golds whose goal omits the perfusion
#: flow get an arbitrary-but-consistent flow so derive_pk is not underdetermined.
_SPHEROID_EXTRA = {
    "spheroid-96ula-medium": "The spheroids are HepG2, 1000 cells each, with a mean cell diameter of 20 µm.",
    "spheroid-384ula-medium": "The spheroids are HepG2, 1000 cells each, with a mean cell diameter of 20 µm.",
    "spheroid-hanging-drop-total": "The spheroids are HepG2, 1000 cells each, with a mean cell diameter of 20 µm.",
    "spheroid-diameter-from-cells": "The spheroids are formed in a 96-well ULA plate.",
    "spheroid-growth-72h": "The spheroids are HepG2, cultured in a 96-well ULA plate; the cells have a mean diameter of 20 µm.",
    "spheroid-200um-hypoxic": "The spheroids are primary human hepatocytes, formed in a 96-well ULA plate.",
    "spheroid-volume-from-diameter": "The spheroid is formed from 1000 cells of 20 µm mean diameter, packed as a solid sphere.",
}
_PK_EXTRA = {
    "pk-extraction-ratio": "The chip is perfused at 2 µL/min.",
    "pk-mM-unit-trap": "The chip is perfused at 2 µL/min.",
    "pk-clearance": "The fluorescent marker is warfarin.",
    "pk-half-life": "The compound is warfarin.",
    "pk-half-life-min-trap": "The compound is warfarin.",
    "pk-accumulation-ratio": "The compound is warfarin.",
}

#: Golds whose answer is a derived quantity no single raw block reproduces —
#: excluded with reason (see module docstring).
_EXCLUDED_SPHEROID = {
    "spheroid-count-from-suspension": "spheroid_count is derived from a suspension count",
    "spheroid-um-mm-unit-ambiguity": "answer is a unit conversion + geometry, not one raw block",
    "spheroid-doxorubicin-dosing": "cross-domain spheroid + dosing; dosing block out of scope",
}
_EXCLUDED_PK = {
    "pk-repeat-dose-24h": "answer is derived from a stated half-life (a derived number)",
    "pk-cell-free-subtraction": "answer is a control subtraction across two chips",
    "pk-high-extraction-clearance": "answer follows from a target E, not stated inlet/outlet",
    "pk-low-extraction-clearance": "answer follows from a target E, not stated inlet/outlet",
    "pk-mass-cleared": "states MW 464 without naming a compound; no real drug at 464 g/mol to pin without inventing a fact",
    "pk-complete-clearance-panel": "same MW 464 / unnamed-compound constraint as pk-mass-cleared",
}


def _canonical_flow(target_shear_pa: float, w: int = 400, h: int = 100, L: int = 10, mu: float = 1e-3) -> dict:
    q = mf.flow_rate_for_shear_stress(target_shear_pa, w, h, mu)
    return {
        "chip": {"width_um": float(w), "height_um": float(h), "length_mm": float(L),
                 "channel_count": 1, "material": "PDMS"},
        "flow": {"flow_rate_uLmin": round(q, 6), "viscosity_pas": mu, "density_kgm3": 1000.0},
    }


def _chip_flow(w: int, h: int, L: int, q: float, mu: float = 1e-3) -> dict:
    return {
        "chip": {"width_um": float(w), "height_um": float(h), "length_mm": float(L),
                 "channel_count": 1, "material": "PDMS"},
        "flow": {"flow_rate_uLmin": round(q, 6), "viscosity_pas": mu, "density_kgm3": 1000.0},
    }


def _chip_cells(w: int, h: int, L: int, cell_type: str, density: float) -> dict:
    area = (w * 1e-6) * (L * 1e-3) * 1e4
    return {
        "chip": {"width_um": float(w), "height_um": float(h), "length_mm": float(L),
                 "channel_count": 1, "material": "PDMS"},
        "cells": {"cell_type": cell_type, "seeding_density_cells_cm2": density,
                  "culture_area_cm2": round(area, 4)},
    }


def _culture(plate_format: str, wells: int, cell_type: str, density: float, **extra) -> dict:
    raw = {"plate_format": f"{plate_format}-well", "wells": wells, "cell_type": cell_type,
           "seeding_density_cells_cm2": density}
    raw.update(extra)
    return {"culture": raw}


def _spheroid(
    cell_type: str | None, fmt: str, count: int, cps: int, cell_d: float, **extra
) -> dict:
    raw = {"spheroid_format": fmt, "spheroid_count": int(count),
           "cells_per_spheroid": int(cps), "cell_diameter_um": float(cell_d)}
    if cell_type is not None:
        raw["cell_type"] = cell_type
    raw.update(extra)
    return {"spheroid": raw}


def _pk(compound: str, cin: float, cout: float, q: float, **extra) -> dict:
    raw = {"compound": compound, "inlet_concentration_uM": float(cin),
           "outlet_concentration_uM": float(cout), "flow_rate_uLmin": float(q)}
    raw.update(extra)
    return {"pk": raw}


#: Build a (goal, raw, domain) triple for one reading gold. Returns None for
#: golds that are not representable as a single raw block.
def _build_flow_pair(g: dict) -> tuple[str, dict, str] | None:
    gid = g["id"]
    # "Pick the geometry and flow that achieve target shear" golds.
    if gid in {
        "liver-sinusoid-shear", "kidney-proximal-tubule", "microvascular-venular",
        "lung-alveolar-shear", "arterial-shear-15dyn", "venular-shear-3dyn",
    }:
        shear = g["expected"]["shear_pa"]
        return g["goal"], _canonical_flow(shear), "flow"

    raw: dict | None = None
    if gid == "selfconsistent-400x100-shear":
        raw = _chip_flow(400, 100, 20, q=2.0)
    elif gid == "selfconsistent-800x100-shear":
        # target 0.1 Pa on water → flow rate is the answer (q computed).
        raw = _canonical_flow(0.1, w=800, h=100, L=20)
    elif gid in {"reynolds-laminar-check", "reynolds-800x100-100uLmin"}:
        w, h, q = (400, 100, 50.0) if "reynolds-laminar" in gid else (800, 100, 100.0)
        raw = _chip_flow(w, h, 20, q=q)
    elif gid == "selfconsistent-residence-60s":
        # 400×100×10 mm → volume 0.4 µL; 60 s = 1 min → q = 0.4 µL/min.
        raw = _chip_flow(400, 100, 10, q=0.4)
    elif gid == "selfconsistent-channel-volume":
        raw = _chip_flow(800, 100, 20, q=1.0)  # q arbitrary; volume depends only on geometry
    elif gid == "selfconsistent-mean-velocity":
        raw = _chip_flow(400, 100, 20, q=2.0)
    elif gid == "selfconsistent-pressure-drop-40mm":
        raw = _chip_flow(400, 100, 40, q=2.0)
    elif gid == "selfconsistent-flow-for-01Pa-200x50":
        raw = _canonical_flow(0.1, w=200, h=50, L=20)
    elif gid in {"seeding-density-hepg2", "seeding-endothelial-5e4"}:
        cell = "HepG2" if "hepg2" in gid else "HUVEC"
        raw = _chip_cells(400, 100, 20, cell, 5e4)
    elif gid == "seeding-primary-hepatocytes":
        raw = _chip_cells(800, 100, 20, "primary human hepatocytes", 1.5e5)
    elif gid == "dmso-vehicle-check":
        raw = {"dosing": {"compound": "APAP", "molecular_weight_g_mol": _DOSE_APAP_MW,
                          "stock_mM": 500.0, "working_mM": 0.5, "vehicle_control": True}}
    elif gid == "dmso-at-guideline-1mM":
        raw = {"dosing": {"compound": "warfarin", "molecular_weight_g_mol": 308.3,
                          "stock_mM": 200.0, "working_mM": 1.0, "vehicle_control": True}}
    elif gid in {"power-80-effect-1", "power-80-effect-half", "power-80-effect-02", "power-90-effect-1"}:
        effect = {"power-80-effect-1": 1.0, "power-80-effect-half": 0.5,
                  "power-80-effect-02": 0.2, "power-90-effect-1": 1.0}[gid]
        power = 0.9 if "power-90" in gid else 0.8
        raw = {"stats": {"effect_size": effect, "std_dev": 1.0, "alpha": 0.05, "power": power}}

    if raw is None:
        return None
    goal = g["goal"]
    if gid in _DOSE_EXTRA:
        goal = goal + " " + _DOSE_EXTRA[gid]
    return goal, raw, "flow"


def _build_culture_pair(g: dict) -> tuple[str, dict, str] | None:
    gid = g["id"]
    if gid in _EXCLUDED_CULTURE:
        return None
    if gid == "plate-96well-hepg2-seed":
        raw = _culture("96", 1, "HepG2", 1e4)
    elif gid == "plate-6well-phh-seed":
        raw = _culture("6", 6, "primary human hepatocytes", 1.5e5)
    elif gid == "plate-confluence-hepg2-72h":
        raw = _culture("96", 1, "HepG2", 2e4, confluent_density_cells_cm2=1e6,
                       doubling_time_h=30.0, culture_duration_h=72.0)
    elif gid == "plate-24well-seed-from-count":
        raw = _culture("24", 1, "HepG2", 8e4)
    elif gid == "plate-split-replating":
        raw = _culture("6", 4, "HepG2", 2e4)
    elif gid == "plate-48well-medium":
        raw = _culture("48", 1, "cells", 1e4)
    elif gid == "plate-12well-seed-hepg2":
        raw = _culture("12", 1, "HepG2", 1e5)
    elif gid == "plate-96well-total-medium":
        raw = _culture("96", 96, "cells", 1e4)
    else:
        return None
    return g["goal"], raw, "culture"


def _build_spheroid_pair(g: dict) -> tuple[str, dict, str] | None:
    gid = g["id"]
    if gid in _EXCLUDED_SPHEROID:
        return None
    # A spheroid raw block must carry at least cells_per_spheroid,
    # cell_diameter_um and spheroid_format for derive_spheroid to run; the
    # golds below state (or are augmented to state) exactly that.
    raw: dict | None = None
    if gid == "spheroid-96well-total":
        raw = _spheroid("HepG2", "96-ula", 96, 1000, 20.0)
    elif gid in {"spheroid-96ula-medium", "spheroid-384ula-medium"}:
        raw = _spheroid("HepG2", "96-ula" if "96" in gid else "384-ula", 1, 1000, 20.0)
    elif gid == "spheroid-hanging-drop-total":
        raw = _spheroid("HepG2", "hanging-drop", 48, 1000, 20.0)
    elif gid == "spheroid-diameter-from-cells":
        raw = _spheroid("primary human hepatocytes", "96-ula", 1, 1000, 20.0)
    elif gid == "spheroid-growth-72h":
        raw = _spheroid("HepG2", "96-ula", 1, 1000, 20.0,
                        doubling_time_h=30.0, culture_duration_h=72.0)
    elif gid == "spheroid-200um-hypoxic":
        raw = _spheroid("primary human hepatocytes", "96-ula", 1, 1000, 20.0)
    elif gid == "spheroid-volume-from-diameter":
        raw = _spheroid("primary human hepatocytes", "96-ula", 1, 1000, 20.0)
    if raw is None:
        return None
    goal = g["goal"]
    if gid in _SPHEROID_EXTRA:
        goal = goal + " " + _SPHEROID_EXTRA[gid]
    return goal, raw, "spheroid"


def _build_pk_pair(g: dict) -> tuple[str, dict, str] | None:
    gid = g["id"]
    if gid in _EXCLUDED_PK:
        return None
    # A pk raw block must carry inlet/outlet/flow for derive_pk to run; the
    # golds below state (or are augmented to state) exactly that.
    raw: dict | None = None
    if gid == "pk-extraction-ratio":
        raw = _pk("diclofenac", 10.0, 7.0, 2.0)
    elif gid == "pk-clearance":
        raw = _pk("warfarin", 10.0, 7.0, 2.0)
    elif gid in {"pk-half-life", "pk-half-life-min-trap"}:
        raw = _pk("warfarin", 10.0, 7.0, 2.0, system_volume_uL=200.0)
    elif gid == "pk-accumulation-ratio":
        raw = _pk("warfarin", 10.0, 7.0, 2.0, system_volume_uL=200.0, dose_interval_h=24.0)
    elif gid == "pk-mM-unit-trap":
        raw = _pk("diclofenac", 500.0, 350.0, 2.0)
    if raw is None:
        return None
    goal = g["goal"]
    if gid in _PK_EXTRA:
        goal = goal + " " + _PK_EXTRA[gid]
    return goal, raw, "pk"


def load_flow_golds() -> list[dict]:
    with open(_GOLD_FLOW) as fh:
        return json.load(fh)


def load_culture_golds() -> list[dict]:
    with open(_GOLD_CULTURE) as fh:
        return json.load(fh)


def load_spheroid_golds() -> list[dict]:
    with open(_GOLD_SPHEROID) as fh:
        return json.load(fh)


def load_pk_golds() -> list[dict]:
    with open(_GOLD_PK) as fh:
        return json.load(fh)


def gold_pairs() -> list[dict]:
    """All supervised (goal → raw) pairs from the reading golds."""
    pairs: list[dict] = []
    skipped: list[str] = []
    for g in load_flow_golds():
        built = _build_flow_pair(g)
        if built is None:
            skipped.append(g["id"])
            continue
        goal, raw, domain = built
        pairs.append({"goal": goal, "raw": raw, "domain": domain, "gold": g["id"]})
    for g in load_culture_golds():
        if g["id"].startswith("blind-"):
            continue
        built = _build_culture_pair(g)
        if built is None:
            skipped.append(g["id"])
            continue
        goal, raw, domain = built
        pairs.append({"goal": goal, "raw": raw, "domain": domain, "gold": g["id"]})
    for g in load_spheroid_golds():
        if g["id"].startswith("blind-"):
            continue
        built = _build_spheroid_pair(g)
        if built is None:
            skipped.append(g["id"])
            continue
        goal, raw, domain = built
        pairs.append({"goal": goal, "raw": raw, "domain": domain, "gold": g["id"]})
    for g in load_pk_golds():
        if g["id"].startswith("blind-"):
            continue
        built = _build_pk_pair(g)
        if built is None:
            skipped.append(g["id"])
            continue
        goal, raw, domain = built
        pairs.append({"goal": goal, "raw": raw, "domain": domain, "gold": g["id"]})
    return pairs, skipped


def write_pairs(out: Path) -> int:
    pairs, skipped = gold_pairs()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gold_pairs.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} gold pairs -> {out / 'gold_pairs.jsonl'}")
    if skipped:
        print(f"skipped (not representable as one raw block): {sorted(skipped)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(write_pairs(Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "results" / "extractor"))
