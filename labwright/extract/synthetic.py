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
- ``spheroid``: a 3D-spheroid culture. raw = {spheroid}. Five patterns mirror
  the benchmark gold prose (eval/gold_spheroid.json): forward (vessel, cell
  type, seed density and cell size all stated), inverse geometry (a target
  diameter is stated and the raw carries the solved cells-per-spheroid,
  asking for volume/cells the way the gold does), default-bearing geometry
  (neither plate nor cell size stated — the raw carries the canonical
  96-ula / 20 µm / 1000-cell defaults), partial-info medium (the plate is
  stated but cells-per-spheroid and cell size fall back to defaults), and
  multi-target (total cells + total medium). The "solid sphere" phrase is
  descriptive only; it never becomes the spheroid_format value.
- ``pk``: a perfused-system pharmacokinetics readout. raw = {pk}. Inlet/outlet/
  flow are stated (a milli-molar variant states mM and the raw carries µM,
  teaching the 1000× conversion); extraction ratio, clearance, half-life,
  accumulation ratio and mass cleared are derived by ``labwright.calc.pk``.
  Compounds are real drugs with standard molecular weights (facts, not
  invented physiology).
- ``barrier``: monolayer QC. raw = {barrier}. TEER, Papp and clearance are
  derived by ``labwright.calc.barrier``; TEER windows come from the physiology
  registry (Caco-2 250–1000, hCMEC/D3 100–240 Ω·cm², sourced there).
- ``oxygen``: dissolved-pO2 culture. raw = {oxygen}. Henry concentration,
  Krogh penetration depth and necrotic-core fraction come from
  ``labwright.calc.o2`` using the registry OCR; cell density is sampled dense
  (1e8–1e9 cells/mL) so penetration lands in the physiological 10–400 µm band.
- ``pumpless``: a gravity-flow rocking platform. raw = {pumpless}. The
  hydrostatic head, driven flow, peak wall shear, OSI and cycles/hour are
  derived by ``labwright.calc.pumpless``; the platform geometry is resampled so
  the peak shear sits in [0.5, 2]× the registry physiological WSS target.
- ``breathing``: lung ALI + cyclic stretch. raw = {breathing}. Breaths/min,
  membrane stroke, strain rate, cycle budget, duty and the ALI film thickness
  come from ``labwright.calc.breathing``.
- ``pulsatile``: cardiac waveform. raw = {pulsatile}. Womersley number, OSI,
  peak shear and pulsatility index are derived by ``labwright.calc.pulsatile``;
  ~85% of rows sample a non-reversing waveform (OSI 0), the rest a reversing
  variant the verifier flags.
- ``scaling``: body-on-chip allometry. raw = {scaling}. Organ flow fraction,
  perfusion flow, mass-proportional cells and the Kleiber factor come from
  ``labwright.calc.scaling`` physiology tables (muscle excluded so every row
  verifies clean).
- ``gradient``: chemotaxis source-sink. raw = {gradient}. Steepness, mid-gap
  concentration, relaxation time and Fick flux come from
  ``labwright.calc.gradient``; ~half the under-10τ rows are resampled to stable
  durations so most rows verify clean while the transient case stays in the mix.

Usage::

    python -m labwright.extract.synthetic --out results/extractor \\
        --n-flow 6000 --n-culture 4000 --n-spheroid 8000 --n-pk 7000 \\
        --n-barrier 4000 --n-oxygen 4000 --n-pumpless 4500 \\
        --n-breathing 4500 --n-pulsatile 4500 --n-scaling 4000 \\
        --n-gradient 4500 --split 0.9 --seed 1234

    # 11-domain v2 (identical base rows + diversity):
    python -m labwright.extract.synthetic --out results/extractor_11dom_v2 \\
        --n-flow 6000 --n-culture 4000 --n-spheroid 8000 --n-pk 7000 \\
        --n-barrier 4000 --n-oxygen 4000 --n-pumpless 4500 \\
        --n-breathing 4500 --n-pulsatile 4500 --n-scaling 4000 \\
        --n-gradient 4500 --n-composite 2000 --neg-frac 0.10 \\
        --split 0.9 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from labwright.calc import barrier as calc_barrier
from labwright.calc import breathing as cb
from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import gradient as cg
from labwright.calc import microfluidics as mf
from labwright.calc import o2 as calc_o2
from labwright.calc import pk as calc_pk
from labwright.calc import pulsatile as calc_pulsatile
from labwright.calc import pumpless as calc_pumpless
from labwright.calc import scaling as cs
from labwright.calc import spheroid as calc_spheroid
from labwright.extract.data import raw_to_json
from labwright.physiology import lookup_cell

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


#: ``≈123 µm`` embedded derived claims that some generators put in the prose as
#: a readability hint (breathing displacement, scaling cell counts, gradient
#: steepness). Matching one lets us build a *negative sample*: the goal then
#: asserts a derived number that the calculators will NOT reproduce from the
#: stated raws, so the extractor must ignore it and the verifier flags the
#: mismatch as a review_required warning.
_APPROX_RE = re.compile(r"≈\s*([0-9]+(?:\.[0-9]+)?)(\s*[A-Za-zµ°/·²³⁻¹]+)?")


def _maybe_perturb_approx(rng: random.Random, goal: str, p: float = 0.12) -> str:
    """Negative-sample hook: flip one ``≈value unit`` claim in the prose to a
    wrong value.

    The raw block is untouched — the extractor's target stays correct. The
    *training signal* is that the goal may assert a derived number that
    disagrees with what the calculators will compute; the model must return the
    raw inputs and never echo the claimed derived value back. Verifier hits the
    mismatch as a review_required warning (tolerated by the data gate), which is
    exactly how a real contradictory lab-note goal behaves.
    """
    hits = list(_APPROX_RE.finditer(goal))
    if not hits or rng.random() >= p:
        return goal
    m = rng.choice(hits)
    num = float(m.group(1))
    wrong = num * rng.choice([0.45, 0.65, 1.4, 1.7, 2.1])
    rendered = f"{wrong:.0f}" if wrong >= 10 else f"{wrong:.2f}".rstrip("0").rstrip(".")
    return goal[: m.start()] + "≈" + rendered + (m.group(2) or "") + goal[m.end():]


#: Physically coherent cross-domain pairs for composite goals. Each pair merges
#: two single-domain generators: the goal describes two subsystems in one
#: platform and the raw carries two top-level blocks. These teach the extractor
#: to emit more than one block when the goal warrants it (body-on-chip style
#: prompts) while the single-block rows keep it conservative elsewhere.
_COMPOSITE_PAIRS: list[tuple[str, str]] = [
    ("pumpless", "oxygen"),    # rocking liver-chip with dissolved-pO2 control
    ("breathing", "barrier"),  # lung ALI + stretch platform with monolayer QC
    ("barrier", "oxygen"),     # BBB insert with dissolved-pO2 culture
    ("pulsatile", "oxygen"),   # cardiac waveform chip with pO2 control
    ("gradient", "scaling"),   # chemotaxis module on a body-on-chip scale
    ("culture", "barrier"),    # plate culture whose wells double as QC inserts
]


def generate_composite(rng: random.Random) -> dict:
    """Cross-domain composite row: two single-domain generators, one goal."""
    a, b = _pick(rng, _COMPOSITE_PAIRS)
    row_a = _SYNTHETIC_GENERATORS[a](rng)
    row_b = _SYNTHETIC_GENERATORS[b](rng)
    goal = row_a["goal"].rstrip()
    if not goal.endswith((".", "?", "!")):
        goal += "."
    goal += " In the same platform, " + row_b["goal"].strip()
    raw = dict(row_a["raw"])
    raw.update(row_b["raw"])
    return {"goal": goal, "raw": raw, "domain": f"composite:{a}+{b}", "_composite": (a, b)}


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
        # Full precision: the verifier cross-checks this against width × length
        # with a 1e-6 relative tolerance, so rounding here would raise spurious
        # "disagrees with width × length" warnings.
        "culture_area_cm2": area,
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
    "A {organ} {material} chip for {cell} has a {w} µm × {h} µm × {L} mm channel and targets "
    "{target} wall shear (medium viscosity {mu} Pa·s). Seed at {density} cells/cm² and report "
    "the perfusion flow rate (µL/min).",
    "For a {cell} monolayer under {target} shear in a {w} µm × {h} µm, {L} mm channel (viscosity "
    "{mu} Pa·s, seeding density {density} cells/cm²), pick the flow rate (µL/min) the pump "
    "must deliver.",
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
        # Bound the predicted confluence: seeding × 2^(dur/dt) can outgrow the
        # confluent density by orders of magnitude (a latent >1000 % hard-band
        # failure), so resample growth until the prediction is sane. A planned
        # harvest should not be over-confluent, so target the *soft* band
        # (≤100 %), matching the verifier's over-confluence warning. Fall back
        # to a no-growth row if no combo fits — a plain culture instance is
        # still a valid training target, and an out-of-band one is not.
        area = calc_culture.well_surface_area_cm2(pf)
        per_well = calc_culture.cells_per_well(density, pf)
        for _ in range(30):
            clo, chi = prof[2]
            conf = float(round(rng.uniform(clo, chi)))
            dt = rng.uniform(*prof[1])
            dur = float(rng.choice([24, 48, 72, 96, 120, 144]))
            pct = calc_culture.cell_count_to_confluence(
                calc_cell.cell_count_after_time(per_well, dt, dur), conf, area
            )
            if pct <= 100.0:
                break
        else:
            conf = dt = dur = None
        if conf is not None:
            out["confluent_density_cells_cm2"] = conf
            out["doubling_time_h"] = _r(dt, 1)
            out["culture_duration_h"] = dur
            parts.append(
                f"Confluence density is ~{conf:.0f} cells/cm²; cells "
                f"double every ~{dt:.0f} h; harvest after {dur:.0f} h."
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
    "Prepare a {pf} plate for {cell}: seed {wells} well(s) at {density} {density_units}.",
    "Culture {cell} in {wells} well(s) of a {pf} plate, plating density {density} "
    "{density_units}.",
]


def generate_culture(rng: random.Random) -> dict:
    pf, wells, cell_type, density, raw, parts = _sample_culture(rng)
    prose = _culture_prose(rng, pf, wells, cell_type, density, parts)
    return {"goal": prose, "raw": {"culture": raw}, "domain": "culture"}


# ---------------------------------------------------------------------------
# Spheroid / 3D-culture domain
# ---------------------------------------------------------------------------

#: Cell type → (canonical mean diameter µm, doubling-time window h | None).
#: The 20 µm hepatocyte and the functional < ~2k cells/spheroid bound are the
#: source-pinned values from eval/gold_spheroid.json (1000 cells of 20 µm →
#: ≈ a 200 µm spheroid; larger spheroids develop necrotic cores — Drug Metab
#: Dispos 2024, doi:10.1124/dmd.124.001653). Doubling windows are the same
#: growth-phase ranges already used by _CELL_PROFILES; primary hepatocytes are
#: treated as non-proliferating (no growth option), consistent with physiology.
_SPHEROID_CELLS: list[tuple[str, float, tuple[float, float] | None]] = [
    ("primary human hepatocytes", 20.0, None),
    ("HepG2", 20.0, (30, 40)),
    ("HUVEC", 15.0, (24, 34)),
    ("Caco-2", 12.0, (30, 60)),
]

#: spheroid vessel → (seed-count window, display name). Working volumes (100/50/
#: 20 µL) are the source-pinned values in labwright/calc/spheroid.py (Corning
#: CLS-AN-235; InSphero Akura 384; Wanigasekara et al., PLOS ONE 2023,
#: doi:10.1371/journal.pone.0276248); the count windows are the practical
#: per-plate capacities.
_SPHEROID_FORMAT_RANGES: dict[str, tuple[int, int]] = {
    "96-ula": (1, 96),
    "384-ula": (1, 384),
    "hanging-drop": (6, 48),
}
#: Display names mirror the benchmark gold prose (eval/gold_spheroid.json),
#: including "ultra-low-attachment" so the model maps that phrase onto the
#: "96-ula"/"384-ula" raw keys instead of guessing.
_SPHEROID_FMT_NAME: dict[str, str] = {
    "96-ula": "96-well ultra-low-attachment (ULA)",
    "384-ula": "384-well ultra-low-attachment (ULA)",
    "hanging-drop": "hanging-drop",
}

#: Canonical defaults the gold set forces the model to infer when a goal omits
#: them: a hepatocyte spheroid is 1000 cells of ~20 µm (≈200 µm, solid-sphere
#: geometry) and an unspecified vessel falls back to the 96-ULA working volume.
_SPHEROID_DEFAULT_CPS = 1000
_SPHEROID_DEFAULT_CD = 20.0
_SPHEROID_DEFAULT_FMT = "96-ula"

_SPHEROID_TEMPLATES = [
    "Form {cell} spheroids in {fmt} plates, {count} per plate, {cps} cells per "
    "spheroid with a mean cell diameter of {cd} µm.",
    "Set up a {fmt} spheroid culture of {cell}: seed {cps} cells per spheroid "
    "(mean cell diameter ~{cd} µm) across {count} wells.",
    "Generate {count} {cell} spheroids in {fmt} plates at {cps} cells each, "
    "with a mean cell diameter of {cd} µm.",
    "Culture {cell} as spheroids in {fmt} vessels — {count} spheroids, {cps} "
    "cells per spheroid, mean cell diameter {cd} µm.",
    "Plate {cell} into {fmt} plates, {count} well(s), {cps} cells per spheroid "
    "(~{cd} µm mean cell diameter).",
]

#: Derived-number questions the prose may ask, so the model learns to extract
#: raw inputs even when the goal asks for a calculator-owned number.
_SPHEROID_ASK = [
    "What expected diameter (µm) will the spheroids reach?",
    "What total medium volume (mL) does the culture require?",
    "What total number of cells is needed to seed the spheroids?",
]

#: Inverse-geometry asks (target diameter → derived number) mirroring the gold
#: prose. The "solid sphere" phrase is descriptive only: the model must learn
#: it is never a spheroid_format value.
_SPHEROID_INVERSE_ASKS = [
    "What is the spheroid volume in µL?",
    "What total number of cells is needed to reach this size?",
]


def _spheroid_raw(
    cell: str,
    fmt: str,
    count: int,
    cps: int,
    cell_d: float,
    growth: tuple[float, float] | None,
) -> dict:
    raw = {
        "cell_type": cell,
        "spheroid_format": fmt,
        "spheroid_count": int(count),
        "cells_per_spheroid": int(cps),
        "cell_diameter_um": float(cell_d),
    }
    if growth is not None:
        raw["doubling_time_h"] = _r(growth[0], 1)
        raw["culture_duration_h"] = float(growth[1])
    return {"spheroid": raw}


def _spheroid_prose(
    rng: random.Random,
    cell: str,
    fmt: str,
    count: int,
    cps: int,
    cell_d: float,
    growth: tuple[float, float] | None,
    target_um: float | None,
) -> str:
    if target_um is not None:
        prose = (
            f"Form {cell} spheroids at a target diameter of {target_um:g} µm in "
            f"{_SPHEROID_FMT_NAME[fmt]} plates, one per well; the cells have a "
            f"mean diameter of {cell_d:g} µm. How many cells per spheroid?"
        )
    else:
        prose = _pick(rng, _SPHEROID_TEMPLATES).format(
            cell=cell, fmt=_SPHEROID_FMT_NAME[fmt], count=count, cps=cps, cd=cell_d
        )
    prose = prose.rstrip()
    if growth is not None:
        dt, dur = growth
        prose += f" Cells double roughly every {dt:g} h; harvest after {dur:g} h."
    if target_um is None and rng.random() < 0.3:
        prose += " " + _pick(rng, _SPHEROID_ASK)
    return prose + _maybe_distractor(rng)


def _spheroid_inverse_prose(cell: str, fmt: str, target_um: float, cell_d: float) -> str:
    """Inverse geometry (gold-style): a target diameter is stated, the raw
    carries the solved cells-per-spheroid. Asks for a derived number."""
    return (
        f"A {cell} spheroid is {target_um:g} µm in diameter, formed in "
        f"{_SPHEROID_FMT_NAME[fmt]} plates one per well; the cells have a mean "
        f"diameter of {cell_d:g} µm. Assuming a solid sphere, what is the "
        f"spheroid volume in µL?"
    )


def _spheroid_cells_prose(cell: str, fmt: str, target_um: float, cell_d: float) -> str:
    """Inverse geometry asking how many cells give a target diameter."""
    return (
        f"You want {cell} spheroids safely below the ~400 µm necrotic-core "
        f"threshold. With a mean cell diameter of {cell_d:g} µm, how many cells "
        f"per spheroid give a {target_um:g} µm diameter? (in "
        f"{_SPHEROID_FMT_NAME[fmt]}, one per well)"
    )


def _spheroid_default_prose(cell: str, target_um: float) -> str:
    """Default-bearing geometry, matching the excluded gold phrasing verbatim:
    neither plate nor cell size is stated; the raw carries the canonical
    96-ula / 20 µm / solved-cell-count defaults."""
    return (
        f"A {cell} spheroid is {target_um:g} µm in diameter. "
        f"Assuming a solid sphere, what is its volume in µL?"
    )


def _spheroid_medium_prose(cell: str, fmt: str, count: int) -> str:
    """Partial-info medium question: the plate (+count) is stated but the
    working volume is a calculator constant recalled via the format key."""
    if fmt == "hanging-drop":
        return (
            f"A hanging-drop array forms {count} {cell} spheroids in standard 20 µL "
            f"droplets. What total medium volume in mL does this require?"
        )
    return (
        f"You form {cell} spheroids one per well in a {_SPHEROID_FMT_NAME[fmt]} "
        f"plate ({count} wells). What is the standard working medium volume per "
        f"spheroid (µL)?"
    )


def _spheroid_multi_prose(cell: str, fmt: str, count: int, cps: int, cell_d: float) -> str:
    """Multi-target forward row: full plate stated, asks total cells + medium."""
    return (
        f"Form one {cell} spheroid per well across {count} wells of a "
        f"{_SPHEROID_FMT_NAME[fmt]} plate, {cps} cells/spheroid of {cell_d:g} µm "
        f"mean cell diameter. What total number of cells and what total medium "
        f"volume (mL) are needed?"
    )


def generate_spheroid(rng: random.Random) -> dict:
    cell, cell_d_base, dt_window = _pick(rng, _SPHEROID_CELLS)
    fmt = _pick(rng, list(_SPHEROID_FORMAT_RANGES))
    lo, hi = _SPHEROID_FORMAT_RANGES[fmt]
    cell_d = float(cell_d_base + rng.randint(-2, 2))
    is_20 = cell_d_base == 20.0

    roll = rng.random()

    if roll < 0.16 and is_20:
        # Default-bearing geometry: no plate, no cell size in the prose; the raw
        # carries the canonical 96-ula / 20 µm and the solved cell count. This
        # teaches the model to fill defaults and never read "solid sphere" as a
        # spheroid_format.
        target_um = float(_pick(rng, [120.0, 150.0, 180.0, 200.0, 220.0]))
        cps = int(round(calc_spheroid.cells_per_spheroid_for_diameter(target_um, _SPHEROID_DEFAULT_CD)))
        raw = _spheroid_raw(cell, _SPHEROID_DEFAULT_FMT, 1, cps, _SPHEROID_DEFAULT_CD, None)
        return {"goal": _spheroid_default_prose(cell, target_um), "raw": raw, "domain": "spheroid"}

    if roll < 0.40 and is_20:
        # Inverse geometry with plate + cell size stated; the ask mirrors the
        # gold ("what is the volume", or the necrotic-core cells question).
        target_um = float(_pick(rng, [120.0, 150.0, 180.0, 200.0, 220.0]))
        cps = int(round(calc_spheroid.cells_per_spheroid_for_diameter(target_um, cell_d)))
        if rng.random() < 0.4:
            prose = _spheroid_cells_prose(cell, fmt, target_um, cell_d)
        else:
            prose = _spheroid_inverse_prose(cell, fmt, target_um, cell_d)
        raw = _spheroid_raw(cell, fmt, 1, cps, cell_d, None)
        return {"goal": prose, "raw": raw, "domain": "spheroid"}

    if roll < 0.60 and is_20:
        # Partial-info medium: plate (+count) stated, cps/cell-diameter defaulted
        # to the canonical values. Teaches recall of the format → working volume.
        count = rng.randint(lo, hi)
        raw = _spheroid_raw(
            cell, fmt, count, _SPHEROID_DEFAULT_CPS, _SPHEROID_DEFAULT_CD, None
        )
        return {"goal": _spheroid_medium_prose(cell, fmt, count), "raw": raw, "domain": "spheroid"}

    if roll < 0.72:
        # Multi-target forward: total cells + total medium (mirrors
        # spheroid-96well-total / spheroid-doxorubicin-dosing).
        count = rng.randint(lo, hi)
        cps = rng.randint(500, 1500)
        prose = _spheroid_multi_prose(cell, fmt, count, cps, cell_d)
        raw = _spheroid_raw(cell, fmt, count, cps, cell_d, None)
        return {"goal": prose, "raw": raw, "domain": "spheroid"}

    # Forward complete, occasionally with growth.
    count = rng.randint(lo, hi)
    cps = rng.randint(500, 1500)
    growth = None
    if dt_window is not None and rng.random() < 0.5:
        growth = (round(rng.uniform(*dt_window), 1), float(rng.choice([24, 48, 72, 96])))
    raw = _spheroid_raw(cell, fmt, count, cps, cell_d, growth)
    prose = _spheroid_prose(rng, cell, fmt, count, cps, cell_d, growth, None)
    return {"goal": prose, "raw": raw, "domain": "spheroid"}


# ---------------------------------------------------------------------------
# PK (perfused-system) domain
# ---------------------------------------------------------------------------

#: Real compounds with their standard molecular weights (g/mol) — facts, not
#: invented physiology. Extraction ratio, clearance, half-life, accumulation
#: ratio and mass cleared are always derived by labwright.calc.pk.
_PK_COMPOUNDS: list[tuple[str, float]] = [
    ("diclofenac", 296.1),
    ("warfarin", 308.3),
    ("propranolol", 259.3),
    ("antipyrine", 188.2),
    ("acetaminophen", 151.2),
    ("doxorubicin", 543.5),
]

_PK_TEMPLATES = [
    "Measure the first-pass clearance of {compound} on a perfused organ-chip: "
    "inlet {cin:g} µM, outlet {cout:g} µM, perfusion flow {q:g} µL/min.",
    "A perfused chip is dosed with {compound} at an inlet concentration of "
    "{cin:g} µM; the outlet reads {cout:g} µM at a perfusion flow of {q:g} µL/min.",
    "Characterize the clearance of {compound} on-chip: {cin:g} µM in, {cout:g} µM "
    "out, perfusion {q:g} µL/min.",
    "A perfused kidney/liver-on-chip is dosed with {compound}: inlet {cin:g} µM, "
    "outlet {cout:g} µM, perfusion flow {q:g} µL/min.",
    "Measure the intrinsic clearance of {compound} on-chip at {cin:g} µM inlet, "
    "{cout:g} µM outlet and {q:g} µL/min perfusion.",
]

#: Derived-number questions the prose may ask (calculator-owned outputs).
_PK_ASK = [
    "What is the extraction ratio?",
    "What is the clearance (µL/min)?",
    "What is the elimination half-life (h)?",
    "What mass does the chip clear per hour (µg/h)?",
    "Report the full clearance profile: extraction ratio, clearance (µL/min), "
    "half-life (h) and mass cleared (µg/h).",
]

#: Milli-molar inlet concentrations for the mM → µM unit-trap rows (the raw
#: always carries µM, mirroring eval/gold_pk.json "pk-mM-unit-trap").
_PK_MILLI_MOLAR = [0.05, 0.1, 0.2, 0.5, 1.0]


def _pk_raw(
    compound: str,
    cin: float,
    cout: float,
    q: float,
    mw: float | None,
    V: float | None,
    tau: float | None,
) -> dict:
    raw = {
        "compound": compound,
        "inlet_concentration_uM": float(cin),
        "outlet_concentration_uM": float(cout),
        "flow_rate_uLmin": float(q),
    }
    if mw is not None:
        raw["molecular_weight_g_mol"] = float(mw)
    if V is not None:
        raw["system_volume_uL"] = float(V)
    if tau is not None:
        raw["dose_interval_h"] = float(tau)
    return {"pk": raw}


def _pk_prose(
    rng: random.Random,
    compound: str,
    cin: float,
    cout: float,
    q: float,
    mw: float | None,
    V: float | None,
    tau: float | None,
) -> str:
    prose = _pick(rng, _PK_TEMPLATES).format(compound=compound, cin=cin, cout=cout, q=q)
    parts: list[str] = []
    if V is not None:
        parts.append(
            f"The recirculating system volume (reservoir + chip + tubing) is {V:g} µL."
        )
    if tau is not None:
        parts.append(f"Repeat doses are given every {tau:g} h.")
    if mw is not None:
        parts.append(f"The compound has a molecular weight of {mw:g} g/mol.")
    if rng.random() < 0.35:
        if tau is not None and rng.random() < 0.35:
            parts.append("What is the steady-state accumulation ratio?")
        else:
            parts.append(_pick(rng, _PK_ASK))
    if parts:
        prose = prose.rstrip() + " " + " ".join(parts)
    return prose + _maybe_distractor(rng)


def generate_pk(rng: random.Random) -> dict:
    compound, mw_full = _pick(rng, _PK_COMPOUNDS)
    cin = float(_pick(rng, [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]))
    # Sample a physiological extraction ratio (E in [0.1, 0.7]) and derive the
    # outlet from it; E itself is a calculator output and never enters the raw.
    e = round(rng.uniform(0.1, 0.7), 2)
    cout = round(cin * (1.0 - e), 2)
    q = round(rng.uniform(1.0, 8.0), 2)

    roll = rng.random()
    if roll < 0.18:
        # mM → µM unit trap (mirrors gold "pk-mM-unit-trap"): the prose states
        # milli-molar and the raw must carry micro-molar (1000× conversion).
        cin_m = float(_pick(rng, _PK_MILLI_MOLAR))
        cout_m = round(cin_m * (1.0 - e), 3)
        raw = _pk_raw(compound, cin_m * 1000.0, cout_m * 1000.0, q, None, None, None)
        prose = (
            f"A perfused {compound} clearance run reports inlet {cin_m:g} mM and "
            f"outlet {cout_m:g} mM at {q:g} µL/min. The chip's data sheet gives "
            f"concentrations in micromolar (µM). Report the inlet and outlet "
            f"concentrations converted to µM."
        )
        return {"goal": prose, "raw": raw, "domain": "pk"}

    if roll < 0.30:
        # Forward with a derived-number distractor (mirrors gold
        # "pk-half-life-min-trap"): the goal states the half-life in minutes as
        # context; the raw still carries only the inputs.
        V = float(_pick(rng, [100.0, 200.0, 300.0, 500.0]))
        mw = mw_full if rng.random() < 0.5 else None
        raw = _pk_raw(compound, cin, cout, q, mw, V, None)
        t_min = calc_pk.half_life_h(V, e * q) * 60.0
        prose = (
            f"A perfused chip is dosed with {compound} at {cin:g} µM (outlet "
            f"{cout:g} µM) at {q:g} µL/min; the recirculating system volume is "
            f"{V:g} µL. The half-life works out to {t_min:.0f} minutes. "
            f"Report the elimination half-life in hours."
        )
        if mw is not None:
            prose += f" The compound has a molecular weight of {mw:g} g/mol."
        return {"goal": prose, "raw": raw, "domain": "pk"}

    mw = mw_full if rng.random() < 0.7 else None
    V = float(_pick(rng, [100.0, 200.0, 300.0, 500.0])) if rng.random() < 0.8 else None
    tau = float(_pick(rng, [8.0, 12.0, 24.0, 48.0])) if V is not None and rng.random() < 0.7 else None
    raw = _pk_raw(compound, cin, cout, q, mw, V, tau)
    prose = _pk_prose(rng, compound, cin, cout, q, mw, V, tau)
    return {"goal": prose, "raw": raw, "domain": "pk"}


# ---------------------------------------------------------------------------
# Barrier domain (TEER / Papp QC)
# ---------------------------------------------------------------------------

#: Cell type → (TEER window Ω·cm², barrier label). The windows are the registry
#: ranges in labwright/physiology.py (Caco-2 250–1000, hCMEC/D3 100–240), which
#: carry their own sources; nothing here is invented.
_BARRIER_CELLS: list[tuple[str, tuple[float, float], str]] = [
    ("Caco-2", (250.0, 1000.0), "intestinal"),
    ("hCMEC/D3", (100.0, 240.0), "blood-brain"),
]

#: Standard insert areas, cm² (24-well ≈ 0.33, 12-well ≈ 1.12).
_BARRIER_AREAS_CM2 = [0.33, 1.12]

#: Typical cell-free insert resistance (electrode + medium), Ω.
_BARRIER_BLANK_R_OHM = (80.0, 200.0)

_BARRIER_PROBES = ["Lucifer yellow", "FITC-dextran 4 kDa", "mannitol", "rhodamine-123"]

_BARRIER_TEMPLATES = [
    "Measure the barrier function of {cell} on a {area} cm² insert: total resistance "
    "{rtot} Ω against a {rblank} Ω blank. What is the TEER in Ω·cm²?",
    "A {cell} monolayer is grown on a {area} cm² Transwell; the resistance reads "
    "{rtot} Ω with a {rblank} Ω cell-free blank. Report the TEER (Ω·cm²).",
    "QC the {cell} monolayer on a {area} cm² insert — {rtot} Ω total, {rblank} Ω blank. "
    "What is the area-normalised TEER?",
]


def generate_barrier(rng: random.Random) -> dict:
    cell, teer_range, _barrier_label = _pick(rng, _BARRIER_CELLS)
    area = _pick(rng, _BARRIER_AREAS_CM2)
    teer = round(rng.uniform(*teer_range), 1)
    rblank = round(rng.uniform(*_BARRIER_BLANK_R_OHM), 1)
    rtot = round(teer / area + rblank, 1)
    raw = {
        "cell_type": cell,
        "insert_area_cm2": area,
        "resistance_total_ohm": rtot,
        "resistance_blank_ohm": rblank,
    }

    if rng.random() < 0.55:
        # Add a permeability readout: sample Papp in the tight-barrier band and
        # derive the flux backwards so the raw stays consistent (flux in band).
        probe = _pick(rng, _BARRIER_PROBES)
        conc = round(rng.uniform(10.0, 500.0), 1)
        for _ in range(30):
            papp = rng.uniform(3e-7, 1e-5)
            flux = calc_barrier.flux_nmol_min(papp, conc, area)
            if 1e-3 <= flux <= 100.0:
                break
        raw["probe"] = probe
        raw["donor_conc_um"] = conc
        raw["flux_nmol_min"] = _r(flux, 6)
        prose = (
            f"{cell} on a {area:g} cm² insert shows a {probe} flux of {flux:.4g} "
            f"nmol/min from a {conc:g} µM donor. What is the apparent permeability "
            f"(cm/s)? (TEER {teer:g} Ω·cm².)"
        )
    else:
        prose = _pick(rng, _BARRIER_TEMPLATES).format(
            cell=cell, area=area, rtot=rtot, rblank=rblank
        )
    return {"goal": prose + _maybe_distractor(rng), "raw": {"barrier": raw}, "domain": "barrier"}


# ---------------------------------------------------------------------------
# Oxygen domain (dissolved pO2, Krogh penetration)
# ---------------------------------------------------------------------------

#: Cell types with a registry OCR (never proposed by the model).
_O2_CELLS = ["HepG2", "primary human hepatocytes"]


def generate_oxygen(rng: random.Random) -> dict:
    cell = _pick(rng, _O2_CELLS)
    po2 = float(rng.choice([20, 40, 60, 80, 100, 120, 140, 160]))
    # Dense tissue to push Krogh penetration into the physiological 10–400 µm band.
    density = round(rng.uniform(1e8, 9e8))
    while True:
        prof = lookup_cell(cell)
        ocr = prof.o2_consumption_nmol_min_1e6
        ocr_mid = (ocr[0] + ocr[1]) / 2.0
        pen = calc_o2.o2_penetration_depth_um(
            calc_o2.volumetric_o2_consumption(
                calc_o2.nmol_min_per_1e6_to_fmol_s(ocr_mid), density
            )
        )
        if 10.0 <= pen <= 400.0:
            break
        density = round(rng.uniform(1e8, 9e8))

    raw = {
        "cell_type": cell,
        "target_po2_mmhg": po2,
        "cell_density_cells_ml": float(density),
    }
    if rng.random() < 0.6:
        raw["spheroid_diameter_um"] = float(rng.choice([200, 300, 400, 500, 600, 800, 1000]))

    parts = [
        f"Culture {cell} at a target pO2 of {po2:g} mmHg with a cell density of "
        f"{density:.2e} cells/mL."
    ]
    if "spheroid_diameter_um" in raw:
        parts.append(
            f"The culture forms {raw['spheroid_diameter_um']:g} µm spheroids — "
            "what fraction of the spheroid is hypoxic?"
        )
    parts.append("How deep does oxygen penetrate (µm)?")
    prose = " ".join(parts)
    return {"goal": prose + _maybe_distractor(rng), "raw": {"oxygen": raw}, "domain": "oxygen"}


# ---------------------------------------------------------------------------
# Pumpless domain (gravity-flow rocking platform)
# ---------------------------------------------------------------------------

#: Cell types with a registry physiological WSS (the target for the rocker).
_PUMPLESS_CELLS = ["hepg2", "huvec", "a549"]


def generate_pumpless(rng: random.Random) -> dict:
    cell = _pick(rng, _PUMPLESS_CELLS)
    prof = lookup_cell(cell)
    lo, hi = prof.shear_range_pa
    target = (lo + hi) / 2.0
    # Sample the platform geometry so the peak shear lands in [0.5, 2]×target
    # (shear = ρ·g·sin(θ)·h/2 — independent of L).
    while True:
        tilt = rng.randint(3, 25)
        h = rng.choice([75, 100, 125, 150, 200, 250])
        w = rng.randint(400, 1000)
        L = rng.choice([10, 20, 30, 40])
        period = rng.choice([5, 10, 20, 30, 60])
        shear = calc_pumpless.peak_wall_shear_from_head(
            calc_pumpless.hydrostatic_pressure_pa(
                calc_pumpless.CULTURE_MEDIUM_DENSITY_KGM3, tilt, L
            ),
            w, h, L,
        )
        ratio = shear / target
        if 0.5 <= ratio <= 2.0:
            break

    bwd = _pick(rng, [0.0, 0.0, 0.5, 1.0, 1.0])
    raw = {
        "cell_type": cell,
        "tilt_angle_deg": float(tilt),
        "channel_length_mm": float(L),
        "width_um": float(w),
        "height_um": float(h),
        "rocking_half_period_s": float(period),
        "backward_shear_fraction": float(bwd),
    }
    prose = (
        f"A rocking-platform chip cultures {cell} (physiological wall shear "
        f"{lo:g}–{hi:g} Pa) on a {w} µm × {h} µm × {L} mm channel tilted {tilt}° "
        f"with a {period} s rocking half-period. What peak wall shear (Pa) do the "
        f"cells experience and how does it compare with the physiological range?"
    )
    return {"goal": prose + _maybe_distractor(rng), "raw": {"pumpless": raw}, "domain": "pumpless"}


# ---------------------------------------------------------------------------
# Breathing domain (lung ALI + cyclic stretch)
# ---------------------------------------------------------------------------

_BREATHING_CELLS = ["alveolar epithelial type II cells", "primary human bronchial epithelial cells"]


def generate_breathing(rng: random.Random) -> dict:
    cell = _pick(rng, _BREATHING_CELLS)
    freq = rng.choice([0.2, 0.2, 0.25])
    strain = round(rng.uniform(5.0, 12.0), 1)
    span = rng.choice([150, 200, 250, 300, 350])
    dur = rng.choice([24, 48, 72, 120, 168])
    # Keep the residual ALI film within the physiological soft band (1–1000 µm):
    # a 50 µL dose on the smallest insert would give a 1515 µm film, which the
    # sanity check correctly flags — so resample the dose/area pair instead of
    # generating a soft-band violation as a routine training row.
    apical = rng.choice([5, 10, 20, 30, 50])
    area = rng.choice([0.33, 0.66, 1.12])
    while cb.ali_liquid_film_um(apical, area) > 1000:
        apical = rng.choice([5, 10, 20, 30, 50])
        area = rng.choice([0.33, 0.66, 1.12])
    cycle = rng.choice([1.0, 2.0])
    stretch = round(rng.uniform(0.1, 0.5) * cycle, 2)
    raw = {
        "cell_type": cell,
        "frequency_hz": float(freq),
        "strain_pct": float(strain),
        "membrane_span_um": float(span),
        "culture_duration_h": float(dur),
        "apical_volume_ul": float(apical),
        "surface_area_cm2": float(area),
        "stretch_seconds": float(stretch),
        "cycle_seconds": float(cycle),
    }
    bpm = cb.breaths_per_minute(freq)
    film = cb.ali_liquid_film_um(apical, area)
    prose = (
        f"A lung-on-chip for {cell} cycles at {freq:g} Hz with {strain:g}% linear "
        f"strain over a {span} µm membrane for {dur} h. At ALI the apical surface "
        f"carries {apical} µL over {area:g} cm². What are the breathing rate "
        f"({bpm:g} breaths/min is the expected rate), the total stretch cycles, and "
        f"the residual apical film thickness (≈{film:.0f} µm)?"
    )
    return {"goal": prose + _maybe_distractor(rng), "raw": {"breathing": raw}, "domain": "breathing"}


# ---------------------------------------------------------------------------
# Pulsatile domain (cardiac waveform)
# ---------------------------------------------------------------------------

_PULSATILE_CELLS = ["HUVEC", "human aortic endothelial cells", "hiPSC-derived cardiomyocytes"]


def generate_pulsatile(rng: random.Random) -> dict:
    cell = _pick(rng, _PULSATILE_CELLS)
    freq = rng.choice([0.8, 1.0, 1.2, 1.5, 2.0])
    height = rng.choice([100, 150, 200, 300])
    mean_shear = round(rng.uniform(0.5, 3.0), 2)
    if rng.random() < 0.85:
        amp = round(rng.uniform(0.2, 0.8) * mean_shear, 2)  # no reversal → OSI 0
    else:
        amp = round(rng.uniform(1.0, 1.5) * mean_shear, 2)  # reversing variant
    mflow = round(rng.uniform(1.0, 20.0), 1)
    pi = round(rng.uniform(0.3, 1.5), 2)
    peak = round(mflow * (1.0 + pi), 1)
    mn = round(mflow * (1.0 - pi), 1)
    if mn < 0:
        mn = 0.0
    raw = {
        "cell_type": cell,
        "frequency_hz": float(freq),
        "channel_height_um": float(height),
        "shear_mean_pa": float(mean_shear),
        "shear_amplitude_pa": float(amp),
        "peak_flow_uLmin": float(peak),
        "minimum_flow_uLmin": float(mn),
        "mean_flow_uLmin": float(mflow),
    }
    prose = (
        f"A heart-on-chip perfuses {cell} with a pulsatile waveform at {freq:g} Hz "
        f"in a {height} µm channel: mean shear {mean_shear:g} Pa with amplitude "
        f"{amp:g} Pa (peak flow {peak:g} µL/min, minimum {mn:g} µL/min, mean "
        f"{mflow:g} µL/min). Compute the Womersley number, the oscillatory shear "
        f"index and the Gosling pulsatility index."
    )
    return {"goal": prose + _maybe_distractor(rng), "raw": {"pulsatile": raw}, "domain": "pulsatile"}


# ---------------------------------------------------------------------------
# Scaling domain (body-on-chip allometry)
# ---------------------------------------------------------------------------

#: Organs whose allometric factor stays inside the 0.01–0.5 sanity band
#: (muscle at 0.503 is excluded so every generated row verifies clean).
_SCALING_ORGANS = ["liver", "kidneys", "brain", "heart", "gut", "skin", "lungs"]

_SCALING_ORGAN_NAME = {
    "liver": "liver", "kidneys": "kidneys", "brain": "brain", "heart": "heart",
    "gut": "gut", "skin": "skin", "lungs": "lungs",
}


def generate_scaling(rng: random.Random) -> dict:
    organ = _pick(rng, _SCALING_ORGANS)
    co = float(rng.randint(200, 5000))  # keeps every organ flow ≥ 10 mL/min
    total_cells = round(rng.uniform(1e5, 1e7))
    volume = float(rng.choice([10, 50, 100, 250, 500, 1000]))
    flow = round(rng.uniform(1.0, 1000.0), 1)
    transit = cs.transit_time_s(volume, flow)
    if not (1.0 <= transit <= 1e4):
        # Resample into the transit band.
        for _ in range(30):
            flow = round(rng.uniform(1.0, 1000.0), 1)
            transit = cs.transit_time_s(volume, flow)
            if 1.0 <= transit <= 1e4:
                break
    target = round(transit + rng.uniform(-300, 300), 1)
    if target < 0:
        target = round(transit, 1)
    raw = {
        "organ": organ,
        "total_cells_chip": float(total_cells),
        "cardiac_output_mlmin": float(co),
        "chip_volume_ul": float(volume),
        "flow_rate_uLmin": float(flow),
        "target_transit_s": float(target),
    }
    frac = cs.organ_flow_fraction(organ)
    organ_flow = frac * co
    cells = cs.scale_cell_number(cs.ORGAN_MASS_G[organ], cs.BODY_MASS_G, total_cells)
    prose = (
        f"Scale a body-on-chip for a {_SCALING_ORGAN_NAME[organ]} compartment: the chip "
        f"supports {total_cells:.3g} total cells perfused at a cardiac output of "
        f"{co:g} mL/min. How many cells go to the {organ} compartment, what perfusion "
        f"flow ({organ_flow:.0f} mL/min) does it receive, and does a {volume:g} µL "
        f"compartment at {flow:g} µL/min match the ~{target:g} s in-vivo transit? "
        f"(≈{cells:.0f} cells at the {frac * 100:.0f}% flow fraction.)"
    )
    return {"goal": prose + _maybe_distractor(rng), "raw": {"scaling": raw}, "domain": "scaling"}


# ---------------------------------------------------------------------------
# Gradient domain (chemotaxis source-sink)
# ---------------------------------------------------------------------------

#: Chemoattractant → chemotactic cell pair (facts from the assay literature;
#: the numbers themselves are always calculator-derived).
_GRADIENT_PAIRS: list[tuple[str, str]] = [
    ("CXCL12", "primary human neutrophils"),
    ("fMLP", "primary human neutrophils"),
    ("EGF", "cancer cells"),
    ("SDF-1α", "neural progenitor cells"),
    ("CCL2", "monocyte-derived dendritic cells"),
]


def generate_gradient(rng: random.Random) -> dict:
    chemo, cell = _pick(rng, _GRADIENT_PAIRS)
    src = float(rng.choice([10, 50, 100, 200, 500, 1000]))
    sink = float(round(rng.uniform(0.0, 0.3 * src), 1))
    distance = float(rng.choice([300, 500, 800, 1000, 1500, 2000]))
    hours = float(rng.choice([6, 12, 24, 48]))

    raw = {
        "chemoattractant": chemo,
        "source_conc_um": float(src),
        "sink_conc_um": float(sink),
        "distance_um": float(distance),
        "experiment_hours": hours,
    }
    tau = cg.diffusive_relaxation_time_s(distance)
    steep = cg.linear_gradient_steepness_um_per_mm(src, sink, distance)
    stable = cg.gradient_stability_check(tau, hours)["stable"]
    if not stable:
        # Unstable rows still teach extraction; keep ~half of them by resampling.
        if rng.random() < 0.5:
            hours = 48.0
            raw["experiment_hours"] = hours
    prose = (
        f"A chemotaxis chip exposes {cell} to {src:g} µM {chemo} in the source "
        f"channel vs {sink:g} µM buffer across a {distance:g} µm agarose bridge, run "
        f"for {hours:g} h. What is the gradient steepness (≈{steep:.0f} µM/mm) and "
        f"how long until it reaches steady state?"
    )
    return {"goal": prose + _maybe_distractor(rng), "raw": {"gradient": raw}, "domain": "gradient"}


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------


#: Registry used by ``generate_composite`` — filled in at the end of the module
#: (all generators are defined above). Indirection keeps the pair list readable.
_SYNTHETIC_GENERATORS: dict[str, object] = {}


def generate(
    n_flow: int,
    n_culture: int,
    n_spheroid: int = 0,
    n_pk: int = 0,
    n_barrier: int = 0,
    n_oxygen: int = 0,
    n_pumpless: int = 0,
    n_breathing: int = 0,
    n_pulsatile: int = 0,
    n_scaling: int = 0,
    n_gradient: int = 0,
    n_composite: int = 0,
    neg_frac: float = 0.0,
    seed: int = 1234,
) -> list[dict]:
    rng = random.Random(seed)
    rows = [generate_flow(rng) for _ in range(n_flow)]
    rows += [generate_culture(rng) for _ in range(n_culture)]
    rows += [generate_spheroid(rng) for _ in range(n_spheroid)]
    rows += [generate_pk(rng) for _ in range(n_pk)]
    rows += [generate_barrier(rng) for _ in range(n_barrier)]
    rows += [generate_oxygen(rng) for _ in range(n_oxygen)]
    rows += [generate_pumpless(rng) for _ in range(n_pumpless)]
    rows += [generate_breathing(rng) for _ in range(n_breathing)]
    rows += [generate_pulsatile(rng) for _ in range(n_pulsatile)]
    rows += [generate_scaling(rng) for _ in range(n_scaling)]
    rows += [generate_gradient(rng) for _ in range(n_gradient)]
    rows += [generate_composite(rng) for _ in range(n_composite)]
    if neg_frac > 0:
        # Negative samples: flip one embedded "≈value unit" derived claim to a
        # wrong value. The raw block stays correct — the point is that the goal
        # can assert a derived number the calculators will contradict.
        n_neg = max(1, round(len(rows) * neg_frac))
        candidates = [i for i, r in enumerate(rows) if _APPROX_RE.search(r["goal"])]
        if candidates:
            for i in rng.sample(candidates, min(n_neg, len(candidates))):
                rows[i]["goal"] = _maybe_perturb_approx(rng, rows[i]["goal"], p=1.0)
    rng.shuffle(rows)
    return rows


def write_split(rows: list[dict], out: Path, split: float = 0.9) -> tuple[Path, Path]:
    """Split rows (already shuffled) into train/eval jsonl and write them.

    The eval half is deduplicated against the train half: no eval row may share
    a goal string or a raw block with a train row. Without this, the spheroid
    inverse target-diameter variant — a small lattice (cell type × diameter ×
    format) — would place byte-identical rows on both sides of the split, so
    the held-out consistency/field-recovery numbers would be inflated by rows
    the model had already seen verbatim.
    """
    n_train = int(len(rows) * split)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = rows[:n_train], rows[n_train:]
    train_goals = {json.dumps(r["goal"], ensure_ascii=False) for r in train_rows}
    train_raws = {json.dumps(r["raw"], ensure_ascii=False, sort_keys=True) for r in train_rows}
    kept: list[dict] = []
    for r in eval_rows:
        if json.dumps(r["goal"], ensure_ascii=False) in train_goals:
            continue
        if json.dumps(r["raw"], ensure_ascii=False, sort_keys=True) in train_raws:
            continue
        kept.append(r)
    train_p, eval_p = out / "train.jsonl", out / "eval.jsonl"
    for path, part in ((train_p, train_rows), (eval_p, kept)):
        with open(path, "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_p, eval_p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="output directory (default: results/extractor; a smoke "
                             "run with no explicit --out goes to results/smoke so it "
                             "can never clobber the production split)")
    parser.add_argument("--n-flow", type=int, default=2500)
    parser.add_argument("--n-culture", type=int, default=1500)
    parser.add_argument("--n-spheroid", type=int, default=3000)
    parser.add_argument("--n-pk", type=int, default=3000)
    parser.add_argument("--n-barrier", type=int, default=0)
    parser.add_argument("--n-oxygen", type=int, default=0)
    parser.add_argument("--n-pumpless", type=int, default=0)
    parser.add_argument("--n-breathing", type=int, default=0)
    parser.add_argument("--n-pulsatile", type=int, default=0)
    parser.add_argument("--n-scaling", type=int, default=0)
    parser.add_argument("--n-gradient", type=int, default=0)
    parser.add_argument("--n-composite", type=int, default=0,
                        help="cross-domain composite rows (two blocks in one goal)")
    parser.add_argument("--neg-frac", type=float, default=0.0,
                        help="fraction of ≈-bearing rows to turn into negative "
                             "samples (contradictory derived claim in the goal)")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", type=int, default=0, help="generate N of each instead")
    args = parser.parse_args()

    if args.smoke:
        nf = nc = ns = npk = args.smoke
        nb = no = npu = nbr = nps = nsc = ng = args.smoke
    else:
        nf = args.n_flow
        nc = args.n_culture
        ns = args.n_spheroid
        npk = args.n_pk
        nb = args.n_barrier
        no = args.n_oxygen
        npu = args.n_pumpless
        nbr = args.n_breathing
        nps = args.n_pulsatile
        nsc = args.n_scaling
        ng = args.n_gradient
    out = Path(args.out or ("results/smoke" if args.smoke else "results/extractor"))
    rows = generate(
        nf, nc, ns, npk, nb, no, npu, nbr, nps, nsc, ng,
        n_composite=args.n_composite, neg_frac=args.neg_frac, seed=args.seed,
    )
    train_p, eval_p = write_split(rows, out, args.split)
    print(
        f"wrote {len(rows)} rows "
        f"({nf} flow + {nc} culture + {ns} spheroid + {npk} pk + {nb} barrier + "
        f"{no} oxygen + {npu} pumpless + {nbr} breathing + {nps} pulsatile + "
        f"{nsc} scaling + {ng} gradient + {args.n_composite} composite, "
        f"{args.neg_frac:.0%} negatives) -> {train_p} / {eval_p}"
    )
    return 0


_SYNTHETIC_GENERATORS.update(
    {
        "flow": generate_flow,
        "culture": generate_culture,
        "spheroid": generate_spheroid,
        "pk": generate_pk,
        "barrier": generate_barrier,
        "oxygen": generate_oxygen,
        "pumpless": generate_pumpless,
        "breathing": generate_breathing,
        "pulsatile": generate_pulsatile,
        "scaling": generate_scaling,
        "gradient": generate_gradient,
    }
)


if __name__ == "__main__":
    raise SystemExit(main())
