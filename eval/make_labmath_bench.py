"""Generate the ~500-entry LabMath-Bench dataset (eval/gold_labmath_bench.json).

The reviewer-facing benchmark asks for a 500–1000 entry QA set with three
difficulty levels:

* **L1 — Fluid & spatial engineering**: micro-extrusion geometry/kinematics
  (bioprinting) and microfluidic channel shear/transport (flow).
* **L2 — Biochemical stoichiometry**: two-population co-culture seeding ratios
  (coculture) and competitive-inhibition stoichiometry / Cheng–Prusoff
  IC50 conversions (enzyme).
* **L3 — Pipeline parameterization**: ChAMP methylation-array batching,
  PLINK genotype-dataset sizing, and hanging-drop solvent evaporation /
  plate-edge gradients (champ, plink, solvent).

Every entry is **complete-info and self-consistent**: the goal prose states all
raw inputs, and the expected values are exactly what :mod:`labwright.calc`
produces from those inputs via a real ``submit_design``. No number is invented —
the gold is *defined by* the deterministic calculators, so no entry can be
unverifiable. Each entry also carries ``level`` (L1/L2/L3), a ``difficulty``
tag (easy/medium/hard = 1–2 / 3–4 / all derived targets) and an honest
``source`` string.

Values are sampled from each domain's sanity-soft range so every generated
entry passes the verifier with zero errors (warnings, which reflect edge
conditions, are avoided by construction where possible and accepted otherwise).
The seed makes the dataset reproducible.

Usage::

    python -m eval.make_labmath_bench --seed 20260817 --out eval/gold_labmath_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labwright.design import submit_design
from labwright.calc.solvent import drop_volume_after_time, edge_well_factor
from labwright.schema.design import DesignPlan

#: Honest source strings — "defined by the Labwright calculator", never a
#: fabricated measurement. Mirror the equations documented on each calc module.
_SOURCES = {
    "bioprinting": (
        "V = π(d/2)²·L filament-extrusion geometry; print time L/v; m = ρV. "
        "Nozzle diameters are equipment-spec conventions (Labwright calc.bioprinting)."
    ),
    "flow": (
        "τ = 6µQ/(w·h²), Re = ρvDh/µ, ΔP = 12µQL/(w·h³) laminar microchannel "
        "flow (Labwright calc.microfluidics)."
    ),
    "coculture": (
        "N_A = f·N, N_B = N − N_A, ratio = N_A/N_B — fraction-to-count "
        "stoichiometry (Labwright calc.coculture)."
    ),
    "enzyme": (
        "v_i/v_0 = [S]/(Km(1+[I]/Ki)+[S]) (competitive inhibition); IC50 = "
        "Ki(1+[S]/Km) — Cheng & Prusoff, Biochem. Pharmacol. 22:3099–3108 (1973), "
        "doi:10.1016/0006-2952(73)90196-2 (Labwright calc.enzyme)."
    ),
    "champ": (
        "one array per sample; chips = ⌈n/chip_capacity⌉ (450k:12, EPIC:8) — "
        "Illumina product convention, not a measurement (Labwright calc.bioinformatics)."
    ),
    "plink": (
        ".bed = 2 bits/sample/variant (4 samples/byte); 25 per-chromosome files "
        "— PLINK 1.9 format convention (Labwright calc.bioinformatics)."
    ),
    "solvent": (
        "Langmuir d²-law droplet evaporation (diffusion-limited, Magnus "
        "saturation); 96-well edge effect = documented-range parameter "
        "(Labwright calc.solvent)."
    ),
}

#: Derived keys each domain owns, in report order.
_DERIVED_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "bioprinting": ("extrusion_volume_nl", "print_time_s", "extrusion_rate_nl_min",
                    "filament_mass_ug", "lines_to_cover"),
    "flow": ("shear_pa", "reynolds", "pressure_drop_pa", "residence_time_s",
             "channel_volume_ul", "mean_velocity_mms"),
    "coculture": ("cells_per_well_a", "cells_per_well_b", "total_cells_a",
                  "total_cells_b", "seeding_ratio_ab"),
    "enzyme": ("fractional_activity", "percent_inhibition", "ic50_um",
               "apparent_km_um", "velocity_umol_min", "inhibitor_substrate_ratio"),
    "champ": ("n_arrays", "n_chips", "n_expected_failed_arrays"),
    "plink": ("bed_size_mb", "n_per_chr_files", "per_chr_bed_size_mb"),
    "solvent": ("evaporation_rate_ul_hr", "residual_volume_ul",
                "edge_evaporation_factor"),
}

#: Human phrasing of each derived target, for the goal's "report …" clause.
_PHRASE: dict[str, str] = {
    # bioprinting
    "extrusion_volume_nl": "the extruded ink volume (nL)",
    "print_time_s": "the print time (s)",
    "extrusion_rate_nl_min": "the deposition rate (nL/min)",
    "filament_mass_ug": "the filament mass (µg)",
    "lines_to_cover": "the number of fill lines",
    # flow
    "shear_pa": "the wall shear stress (Pa)",
    "reynolds": "the Reynolds number",
    "pressure_drop_pa": "the pressure drop (Pa)",
    "residence_time_s": "the residence time (s)",
    "channel_volume_ul": "the channel volume (µL)",
    "mean_velocity_mms": "the mean velocity (mm/s)",
    # coculture
    "cells_per_well_a": "the per-well count of population A",
    "cells_per_well_b": "the per-well count of population B",
    "total_cells_a": "the total A cells across all wells",
    "total_cells_b": "the total B cells across all wells",
    "seeding_ratio_ab": "the A:B seeding ratio",
    # enzyme
    "fractional_activity": "the fractional activity remaining",
    "percent_inhibition": "the percent inhibition",
    "ic50_um": "the run-condition IC50 (µM)",
    "apparent_km_um": "the apparent Km (µM)",
    "velocity_umol_min": "the inhibited reaction velocity (µmol/min)",
    "inhibitor_substrate_ratio": "the inhibitor:substrate molar ratio",
    # champ
    "n_arrays": "the number of arrays",
    "n_chips": "the number of physical chips",
    "n_expected_failed_arrays": "the expected failed arrays",
    # plink
    "bed_size_mb": "the binary .bed size (MB)",
    "n_per_chr_files": "the number of per-chromosome files",
    "per_chr_bed_size_mb": "the per-chromosome .bed size (MB)",
    # solvent
    "evaporation_rate_ul_hr": "the evaporation rate (µL/h)",
    "residual_volume_ul": "the residual drop volume (µL)",
    "edge_evaporation_factor": "the plate-edge evaporation factor",
}

#: How many derived targets each difficulty tag requests.
_DIFFICULTY_N_TARGETS = {"easy": (1, 2), "medium": (3, 4), "hard": (5, 6)}

_NOZZLE_LABEL = {
    "nozzle_1": "cryo-1 (250 µm)", "nozzle_2": "cryo-2 (410 µm)",
    "nozzle_3": "cryo-3 (500 µm)", "nozzle_4": "UV-4 (200 µm)",
    "nozzle_5": "UV-5 (300 µm)", "nozzle_6": "standard-6 (150 µm)",
}
_WELL_FMT = {6: "6-well", 12: "12-well", 24: "24-well", 48: "48-well", 96: "96-well"}
_WELLS = (6, 12, 24, 48, 96)
_ROWS = tuple("ABCDEFGH")


def _fmt(v: float) -> str:
    """A human-readable number for goal prose."""
    if isinstance(v, int):
        return str(v)
    av = abs(v)
    if av >= 10000 or (0.0 < av < 0.001):
        return f"{v:.4g}"
    return f"{v:g}"


def _report_clause(targets: tuple[str, ...]) -> str:
    words = [_PHRASE[k] for k in targets]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


# ---------------------------------------------------------------------------
# Samplers — each draws a raw dict whose values land in the domain's sanity
# soft range, so the verifier accepts it with zero errors.
# ---------------------------------------------------------------------------


def _sample_bioprinting(rng: random.Random) -> dict:
    return {
        "nozzle_id": rng.choice(list(_NOZZLE_LABEL)),
        "travel_distance_um": rng.randint(2000, 50000),
        "feed_rate_mm_min": round(rng.uniform(1.0, 60.0), 2),
        "density_g_cm3": round(rng.uniform(0.95, 1.20), 3),
        "footprint_width_um": rng.randint(1000, 50000),
        "line_pitch_um": rng.randint(100, 1000),
    }


def _goal_bioprinting(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"Extrude bioink through the {_NOZZLE_LABEL[raw['nozzle_id']]} nozzle along a "
        f"{raw['travel_distance_um']} µm G-code path at {raw['feed_rate_mm_min']} mm/min "
        f"(ink density {raw['density_g_cm3']} g/cm³). The footprint is "
        f"{raw['footprint_width_um']} µm wide and will be filled at "
        f"{raw['line_pitch_um']} µm centre-to-centre line pitch. Report "
        f"{_report_clause(targets)}."
    )


def _sample_flow(rng: random.Random) -> dict:
    return {
        "width_um": rng.randint(200, 1000),
        "height_um": rng.randint(50, 200),
        "length_mm": round(rng.uniform(5.0, 40.0), 1),
        "flow_rate_uLmin": round(rng.uniform(1.0, 30.0), 2),
        "viscosity_pas": rng.choice((0.001, 0.0014, 0.002, 0.0035)),
        "density_kgm3": rng.choice((1000, 1060)),
    }


def _goal_flow(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"A microfluidic channel of {raw['width_um']} µm × {raw['height_um']} µm × "
        f"{raw['length_mm']} mm is perfused with medium (μ = {raw['viscosity_pas']} Pa·s, "
        f"ρ = {raw['density_kgm3']} kg/m³) at {raw['flow_rate_uLmin']} µL/min. Report "
        f"{_report_clause(targets)}."
    )


def _sample_coculture(rng: random.Random) -> dict:
    return {
        "cell_type_a": rng.choice(("HUVEC-T1", "HMEC-1", "HUVEC", "ECFC")),
        "cell_type_b": rng.choice(("HepG2", "PHH", "LX-2", "Huh7")),
        "total_density_cells_cm2": round(rng.uniform(2e4, 5e5), -3),
        "area_cm2": round(rng.uniform(0.1, 10.0), 2),
        "fraction_a": round(rng.uniform(0.05, 0.95), 3),
        "wells": rng.choice(_WELLS),
    }


def _goal_coculture(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"Co-seed {raw['cell_type_a']} and {raw['cell_type_b']} in a "
        f"{_WELL_FMT[raw['wells']]} plate at {_fmt(raw['total_density_cells_cm2'])} cells/cm² "
        f"total across {raw['area_cm2']} cm² per well, with {raw['fraction_a']} of the "
        f"total assigned to {raw['cell_type_a']}. Report {_report_clause(targets)}."
    )


def _sample_enzyme(rng: random.Random) -> dict:
    km = round(rng.uniform(20.0, 2000.0), 1)
    s = round(km * rng.uniform(0.5, 3.0), 1)
    ki = round(rng.uniform(20.0, 300.0), 1)
    i = round(rng.uniform(2.0, min(2.0 * ki, 5.0 * s)), 1)
    return {
        "enzyme": rng.choice(("UGT2B7", "UGT1A1", "CYP3A4", "CYP2D6")),
        "substrate": rng.choice(("UDPGA", "4-methylumbelliferone", "testosterone")),
        "km_um": km,
        "s_conc_um": s,
        "ki_um": ki,
        "i_conc_um": i,
        "vmax_umol_min": round(rng.uniform(0.05, 10.0), 4),
    }


def _goal_enzyme(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"Co-incubate {raw['enzyme']} with {raw['substrate']} at {raw['s_conc_um']} µM "
        f"([S], Km = {raw['km_um']} µM) in the presence of a competitive inhibitor "
        f"(Ki = {raw['ki_um']} µM) at {raw['i_conc_um']} µM, Vmax = "
        f"{raw['vmax_umol_min']} µmol/min. Report {_report_clause(targets)}."
    )


def _sample_champ(rng: random.Random) -> dict:
    return {
        "n_samples": rng.randint(24, 500),
        "platform": rng.choice(("450k", "epic")),
        "fail_rate_pct": round(rng.uniform(1.0, 15.0), 1),
    }


def _goal_champ(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"A HumanMethylation{raw['platform'].upper()} study processes "
        f"{raw['n_samples']} samples at a {raw['fail_rate_pct']}% QC fail rate. Report "
        f"{_report_clause(targets)}."
    )


def _sample_plink(rng: random.Random) -> dict:
    return {
        "n_samples": rng.randint(100, 20000),
        "n_variants": rng.randint(1_000_000, 10_000_000),
        "n_variants_chr": rng.randint(100_000, 1_500_000),
    }


def _goal_plink(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"A GWAS genotype dataset has {raw['n_samples']} samples and "
        f"{raw['n_variants']} variants ({raw['n_variants_chr']} on the largest "
        f"chromosome). Report {_report_clause(targets)}."
    )


def _sample_solvent(rng: random.Random) -> dict:
    """Sample solvent conditions where the drop has *not* fully dried out.

    The Langmuir d²-law projects a residual volume clipped at 0 once a drop
    dries out — and a fully-dried drop is a degenerate benchmark target
    (relative-error recovery divides by zero). Rejection-sample so the residual
    stays ≥ 15 % of the initial volume (edge factor included): the well-posed
    positive-target regime.
    """
    for _ in range(500):
        drop_volume_ul = round(rng.uniform(1.0, 10.0), 2)
        hours = round(rng.uniform(0.05, 4.0), 3)
        temp_c = round(rng.uniform(10.0, 37.0), 1)
        rh = round(rng.uniform(0.3, 0.8), 3)
        well_row = rng.choice(_ROWS)
        well_col = rng.randint(1, 12)
        residual = drop_volume_after_time(
            drop_volume_ul, hours, temp_c, rh,
            evaporation_factor=edge_well_factor(well_row, well_col),
        )
        if residual >= 0.15 * drop_volume_ul:
            return {
                "drop_volume_ul": drop_volume_ul,
                "hours": hours,
                "temp_c": temp_c,
                "rh": rh,
                "well_row": well_row,
                "well_col": well_col,
            }
    raise RuntimeError("solvent sampler could not find a non-dried regime in 500 draws")


def _goal_solvent(raw: dict, targets: tuple[str, ...]) -> str:
    return (
        f"A hanging-drop screen plates {raw['drop_volume_ul']} µL drops at "
        f"{raw['temp_c']} °C and {100.0 * raw['rh']:.0f}% relative humidity. For a well "
        f"in row {raw['well_row']}, column {raw['well_col']}, over {raw['hours']} h, "
        f"report {_report_clause(targets)}."
    )


#: (level, count, sampler, goal builder) per domain.
_DOMAINS: list[dict] = [
    {"domain": "bioprinting", "level": "L1", "count": 85, "sample": _sample_bioprinting,
     "goal": _goal_bioprinting},
    {"domain": "flow", "level": "L1", "count": 85, "sample": _sample_flow,
     "goal": _goal_flow},
    {"domain": "coculture", "level": "L2", "count": 85, "sample": _sample_coculture,
     "goal": _goal_coculture},
    {"domain": "enzyme", "level": "L2", "count": 85, "sample": _sample_enzyme,
     "goal": _goal_enzyme},
    {"domain": "champ", "level": "L3", "count": 57, "sample": _sample_champ,
     "goal": _goal_champ},
    {"domain": "plink", "level": "L3", "count": 57, "sample": _sample_plink,
     "goal": _goal_plink},
    {"domain": "solvent", "level": "L3", "count": 56, "sample": _sample_solvent,
     "goal": _goal_solvent},
]


def _pick_targets(rng: random.Random, domain: str, difficulty: str) -> tuple[str, ...]:
    keys = _DERIVED_BY_DOMAIN[domain]
    lo, hi = _DIFFICULTY_N_TARGETS[difficulty]
    hi = min(hi, len(keys))
    lo = min(lo, hi)  # a domain with few keys cannot be "hard"
    n = rng.randint(lo, hi)
    return tuple(rng.sample(list(keys), n))


def _payload(domain: str, goal: str, raw: dict) -> dict:
    if domain == "flow":
        return {
            "goal": goal,
            "rationale": "complete-info LabMath-Bench (deterministic calculator self-consistent)",
            "caveats": [],
            "chip": {"width_um": raw["width_um"], "height_um": raw["height_um"],
                     "length_mm": raw["length_mm"]},
            "flow": {"flow_rate_uLmin": raw["flow_rate_uLmin"],
                     "viscosity_pas": raw["viscosity_pas"],
                     "density_kgm3": raw["density_kgm3"]},
        }
    return {"goal": goal,
            "rationale": "complete-info LabMath-Bench (deterministic calculator self-consistent)",
            "caveats": [], domain: raw}


def _derive_expected(plan: DesignPlan, domain: str, targets: tuple[str, ...]) -> dict[str, float]:
    obj = plan.derived if domain == "flow" else getattr(plan, domain)
    out: dict[str, float] = {}
    for key in targets:
        v = getattr(obj, key)
        if v is None or isinstance(v, bool):
            return {}
        out[key] = float(v)
    return out


def _build_entry(domain_cfg: dict, idx: int, rng: random.Random) -> dict:
    domain = domain_cfg["domain"]
    difficulty = rng.choices(("easy", "medium", "hard"), weights=(0.30, 0.40, 0.30))[0]
    for _attempt in range(60):
        raw = domain_cfg["sample"](rng)
        targets = _pick_targets(rng, domain, difficulty)
        goal = domain_cfg["goal"](raw, targets)
        result = submit_design(_payload(domain, goal, raw))
        errors = [i for i in result["verification"] if i["level"] == "error"]
        if errors:
            continue
        expected = _derive_expected(DesignPlan(**result["design"]), domain, targets)
        if not expected:
            continue
        return {
            "id": f"lmb-{domain}-{idx:03d}",
            "goal": goal,
            "expected": expected,
            "source": _SOURCES[domain],
            "level": domain_cfg["level"],
            "scenario": "complete-info",
            "difficulty": difficulty,
        }
    raise RuntimeError(f"failed to build a clean {domain} entry after 60 attempts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "gold_labmath_bench.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    entries: list[dict] = []
    for cfg in _DOMAINS:
        made = 0
        while made < cfg["count"]:
            entries.append(_build_entry(cfg, made, rng))
            made += 1
        print(f"[ok] {cfg['domain']:>12} L{cfg['level'][1]} x{cfg['count']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    # Summary: count by level and difficulty (all deterministic from the seed).
    from collections import Counter
    by_level = Counter(e["level"] for e in entries)
    by_diff = Counter(e["difficulty"] for e in entries)
    n_keys = sum(len(e["expected"]) for e in entries)
    print(f"\nsaved -> {args.out}  ({len(entries)} entries, {n_keys} scored targets)")
    print("by level :", dict(sorted(by_level.items())))
    print("difficulty:", dict(by_diff))
    for level in ("L1", "L2", "L3"):
        if by_level[level] < 140:
            print(f"[warn] {level} has only {by_level[level]} entries (< 140)", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
