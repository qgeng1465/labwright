"""Construct + validate the seven post-v1 gold entries (eval/gold_new_domains.json).

Each gold is **complete-info**: the goal prose states every raw input, so the
expected derived values are exactly what :mod:`labwright.calc` produces from
those inputs. The constructor reads the derived values off a real ``submit_design``
plan — the gold is *defined* by the calculators, so no entry can ever be
unverifiable or impossible to hit. Every value lands in the block's sanity
bands, and the physiological anchors (TEER, WSS, breathing rate, organ flow
fraction, gradient geometry) are source-pinned below.

Usage::

    python -m eval.make_gold_new_domains --out eval/gold_new_domains.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labwright.design import submit_design
from labwright.schema.design import DesignPlan

#: Source-pinned physiological anchors — no invented numbers. Every gold above
#: sits on one of these.
_SOURCES = {
    "barrier": (
        "TEER = (R_total − R_blank) × A and Papp = (flux/60)/(A·C0): standard "
        "Transwell QC (Millipore/Costar app notes). Caco-2 TEER 250–1000 Ω·cm²; "
        "hCMEC/D3 (BBB) higher — Gericke et al., Fluids Barriers CNS 2020, "
        "doi:10.1186/s12987-020-00212-5."
    ),
    "oxygen": (
        "Krogh penetration δ = √(2DC0/q) and Henry's law C = α·pO2 (air-saturated "
        "medium ≈ 0.2 mM at 150 mmHg); hepatocyte OCR — Botte et al., Lab Chip "
        "2024, doi:10.1039/d4lc00204k."
    ),
    "pumpless": (
        "Gravity-driven rocking organ-chip perfusion: hydrostatic head "
        "ρgL·sinθ, peak WSS = ρg sinθ·h/2 — Advanced Science 2020, "
        "doi:10.1002/advs.202004856; rocking interval 5 s–60 s, tilt to 25° — "
        "MIMETAS OrganoFlow."
    ),
    "breathing": (
        "Physiological respiratory rate ≈ 0.2 Hz (12 breaths/min) and alveolar "
        "linear strain 5–12 % — Stucki et al., breathing lung-on-chip, "
        "PMC6843435; >20 % strain pathological."
    ),
    "pulsatile": (
        "Womersley α = r√(ωρ/μ) (Womersley 1955); OSI (Ku et al. 1985); PI "
        "(Gosling & King 1974). Heart-on-chip aortic inflow: mean shear 0.59 Pa "
        "at 1.2 Hz, α=0.27, OSI=0.2 — Wang et al., microfluidic flow-profile "
        "generator (Lab Chip)."
    ),
    "scaling": (
        "Organ flow fractions and allometric scale (mass^0.75) for body-on-chip "
        "scaling — Ucciferri, Sbrana & Ahluwalia 2014, doi:10.3389/fbioe.2014.00074; "
        "adult cardiac output ≈ 5 L/min."
    ),
    "gradient": (
        "Diffusion-based source–sink gradients: edge-to-edge spacing ~1 mm, "
        "~200 µm agarose layer — EuropePMC PMC5552394; small-molecule "
        "diffusivity ≈ 5e-10 m²/s (1e-9–1e-10 range)."
    ),
}

#: (id, domain, goal, raw) — raw keys match the block's raw_keys exactly.
_SPECS: list[dict] = [
    # ---- barrier -----------------------------------------------------------
    {
        "id": "barrier-caco2-teer-papp",
        "goal": (
            "Measure barrier integrity of a Caco-2 monolayer on a 1.12 cm² 12-well "
            "Transwell insert. The cell-free blank reads 150 Ω; the seeded insert "
            "reads 900 Ω. With 50 µM of a paracellular probe in the donor and a "
            "steady-state flux of 0.03 nmol/min, report the TEER (Ω·cm²), the "
            "apparent permeability Papp (cm/s) and the clearance (mL/min)."
        ),
        "raw": {"barrier": {
            "cell_type": "Caco-2", "insert_area_cm2": 1.12,
            "resistance_total_ohm": 900.0, "resistance_blank_ohm": 150.0,
            "probe": "paracellular probe", "donor_conc_um": 50.0, "flux_nmol_min": 0.03,
        }},
        "expected": ["teer_ohm_cm2", "papp_cm_s", "clearance_mL_min"],
    },
    {
        "id": "barrier-hcmec-teer",
        "goal": (
            "A blood-brain-barrier model of hCMEC/D3 on a 0.33 cm² 24-well insert: "
            "blank 100 Ω, seeded total 1900 Ω. Lucifer yellow at 20 µM crosses at "
            "0.002 nmol/min. Give the TEER (Ω·cm²) and Papp (cm/s)."
        ),
        "raw": {"barrier": {
            "cell_type": "hCMEC/D3", "insert_area_cm2": 0.33,
            "resistance_total_ohm": 1900.0, "resistance_blank_ohm": 100.0,
            "probe": "Lucifer yellow", "donor_conc_um": 20.0, "flux_nmol_min": 0.002,
        }},
        "expected": ["teer_ohm_cm2", "papp_cm_s"],
    },
    # ---- oxygen ------------------------------------------------------------
    {
        "id": "oxygen-phh-sinusoid",
        "goal": (
            "Primary human hepatocytes perfused at a tissue pO2 of 40 mmHg with a "
            "cell density of 2e8 cells/mL and 200 µm spheroids. Report the "
            "dissolved O2 (mM), the Krogh penetration depth (µm) and the anoxic "
            "core fraction."
        ),
        "raw": {"oxygen": {
            "cell_type": "primary human hepatocytes", "target_po2_mmhg": 40.0,
            "cell_density_cells_ml": 2e8, "spheroid_diameter_um": 200.0,
        }},
        "expected": ["dissolved_o2_mM", "penetration_depth_um", "necrotic_fraction"],
    },
    {
        "id": "oxygen-hepg2-hypoxic-core",
        "goal": (
            "A HepG2 spheroid of 600 µm diameter in air-equilibrated medium "
            "(150 mmHg) at 5e8 cells/mL. Give the penetration depth (µm), the "
            "necrotic core fraction and the O2 demand per 1e6 cells (µmol/min)."
        ),
        "raw": {"oxygen": {
            "cell_type": "HepG2", "target_po2_mmhg": 150.0,
            "cell_density_cells_ml": 5e8, "spheroid_diameter_um": 600.0,
        }},
        "expected": ["dissolved_o2_mM", "penetration_depth_um", "necrotic_fraction",
                     "demand_umol_min"],
    },
    # ---- pumpless ----------------------------------------------------------
    {
        "id": "pumpless-hepg2-rocking",
        "goal": (
            "A rocking-platform chip cultures HepG2 (physiological WSS 0.01–0.05 Pa "
            "sinusoid) on a 700 µm × 150 µm × 30 mm channel tilted 15° with a 20 s "
            "rocking half-period and symmetric rocking. Report the hydrostatic head "
            "(Pa), peak wall shear (Pa), the oscillatory shear index and the rocking "
            "cycles per hour."
        ),
        "raw": {"pumpless": {
            "cell_type": "HepG2", "tilt_angle_deg": 15.0, "channel_length_mm": 30.0,
            "width_um": 700.0, "height_um": 150.0, "rocking_half_period_s": 20.0,
            "backward_shear_fraction": 1.0,
        }},
        "expected": ["hydrostatic_head_pa", "peak_wall_shear_pa",
                     "oscillatory_shear_index", "cycles_per_hour"],
    },
    {
        "id": "pumpless-hcmec-tesla",
        "goal": (
            "An hCMEC/D3 BBB chip rocked on a 600 µm × 50 µm × 20 mm channel at 10° "
            "with a 30 s half-period and unidirectional Tesla-valve flow (backward "
            "shear fraction 0). Give the hydrostatic head (Pa), peak wall shear (Pa), "
            "oscillatory shear index and displaced volume per half-cycle (µL)."
        ),
        "raw": {"pumpless": {
            "cell_type": "hCMEC/D3", "tilt_angle_deg": 10.0, "channel_length_mm": 20.0,
            "width_um": 600.0, "height_um": 50.0, "rocking_half_period_s": 30.0,
            "backward_shear_fraction": 0.0,
        }},
        "expected": ["hydrostatic_head_pa", "peak_wall_shear_pa",
                     "oscillatory_shear_index", "volume_per_half_cycle_ul"],
    },
    # ---- breathing ---------------------------------------------------------
    {
        "id": "breathing-ali-h1299",
        "goal": (
            "An ALI lung-chip breathes at 0.2 Hz with 10 % linear strain on a 300 µm "
            "membrane span. 30 µL of apical medium sits on 1.12 cm². For a 72 h "
            "culture with 1.5 s stretch per 5 s cycle, report the breaths per minute, "
            "membrane stroke (µm), strain rate (/s), total cycles and apical film "
            "thickness (µm)."
        ),
        "raw": {"breathing": {
            "cell_type": "H1299", "frequency_hz": 0.2, "strain_pct": 10.0,
            "membrane_span_um": 300.0, "apical_volume_ul": 30.0, "surface_area_cm2": 1.12,
            "culture_duration_h": 72.0, "stretch_seconds": 1.5, "cycle_seconds": 5.0,
        }},
        "expected": ["breaths_per_minute", "cyclic_displacement_um", "strain_rate_per_s",
                     "total_cycles", "stretch_duty_fraction", "ali_liquid_film_um"],
    },
    {
        "id": "breathing-alveoli-5pct",
        "goal": (
            "Primary alveolar epithelial cells stretched at 0.25 Hz and 5 % strain on "
            "a 400 µm span. 15 µL apical medium on 0.33 cm². Give breaths per minute, "
            "membrane stroke (µm) and the ALI film thickness (µm)."
        ),
        "raw": {"breathing": {
            "cell_type": "primary alveolar epithelial cells", "frequency_hz": 0.25,
            "strain_pct": 5.0, "membrane_span_um": 400.0, "apical_volume_ul": 15.0,
            "surface_area_cm2": 0.33, "culture_duration_h": 24.0,
            "stretch_seconds": 1.2, "cycle_seconds": 4.0,
        }},
        "expected": ["breaths_per_minute", "cyclic_displacement_um", "ali_liquid_film_um"],
    },
    # ---- pulsatile ---------------------------------------------------------
    {
        "id": "pulsatile-aortic-heartchip",
        "goal": (
            "A heart-on-chip perfuses aortic endothelium at 1.2 Hz in a 300 µm-high "
            "channel (blood: 0.0035 Pa·s, 1060 kg/m³). Mean shear 0.20 Pa, amplitude "
            "0.35 Pa; flow peaks at 25 µL/min, dips to 5 µL/min, mean 15 µL/min. "
            "Report the Womersley number, oscillatory shear index, peak shear (Pa) "
            "and Gosling pulsatility index."
        ),
        "raw": {"pulsatile": {
            "cell_type": "aortic endothelium", "frequency_hz": 1.2, "channel_height_um": 300.0,
            "viscosity_pas": 0.0035, "density_kgm3": 1060.0, "shear_mean_pa": 0.20,
            "shear_amplitude_pa": 0.35, "peak_flow_uLmin": 25.0, "minimum_flow_uLmin": 5.0,
            "mean_flow_uLmin": 15.0,
        }},
        "expected": ["womersley_number", "oscillatory_shear_index", "peak_shear_pa",
                     "pulsatility_index"],
    },
    {
        "id": "pulsatile-venous-heartchip",
        "goal": (
            "Venous endothelium on-chip at 1.0 Hz in a 150 µm channel (water: "
            "0.001 Pa·s, 1000 kg/m³), mean shear 0.10 Pa with amplitude 0.08 Pa, "
            "flow 8/2/5 µL/min (peak/min/mean). Give Womersley number, OSI, peak "
            "shear (Pa) and PI."
        ),
        "raw": {"pulsatile": {
            "cell_type": "venous endothelium", "frequency_hz": 1.0, "channel_height_um": 150.0,
            "viscosity_pas": 0.001, "density_kgm3": 1000.0, "shear_mean_pa": 0.10,
            "shear_amplitude_pa": 0.08, "peak_flow_uLmin": 8.0, "minimum_flow_uLmin": 2.0,
            "mean_flow_uLmin": 5.0,
        }},
        "expected": ["womersley_number", "oscillatory_shear_index", "peak_shear_pa",
                     "pulsatility_index"],
    },
    # ---- scaling -----------------------------------------------------------
    {
        "id": "scaling-liver-chip",
        "goal": (
            "A body-on-chip liver compartment is scaled to an adult (70 kg, cardiac "
            "output 5000 mL/min). The chip carries 2e6 cells in 20 µL. To match the "
            "in-vivo liver transit time of 40 s, what chip flow (µL/min) is needed, "
            "and what are the organ flow fraction, organ flow rate (mL/min), "
            "mass-proportional cells in the organ and transit-time match error (s)?"
        ),
        "raw": {"scaling": {
            "organ": "liver", "total_cells_chip": 2e6, "cardiac_output_mlmin": 5000.0,
            "body_mass_g": 70000.0, "chip_volume_ul": 20.0, "flow_rate_uLmin": 30.0,
            "target_transit_s": 40.0,
        }},
        "expected": ["organ_flow_fraction", "organ_flow_rate_mlmin", "cells_in_organ",
                     "transit_time_s", "residence_time_match_error_s"],
    },
    {
        "id": "scaling-kidney-chip",
        "goal": (
            "A kidney compartment on the same body-on-chip (70 kg adult, cardiac "
            "output 5000 mL/min, 5e6 chip cells in 30 µL). In-vivo renal transit "
            "~5 s; pick the chip flow that hits it, then report the organ flow "
            "fraction, flow rate (mL/min), cells-in-organ and transit time (s)."
        ),
        "raw": {"scaling": {
            "organ": "kidneys", "total_cells_chip": 5e6, "cardiac_output_mlmin": 5000.0,
            "body_mass_g": 70000.0, "chip_volume_ul": 30.0, "flow_rate_uLmin": 360.0,
            "target_transit_s": 5.0,
        }},
        "expected": ["organ_flow_fraction", "organ_flow_rate_mlmin", "cells_in_organ",
                     "transit_time_s", "residence_time_match_error_s"],
    },
    # ---- gradient ----------------------------------------------------------
    {
        "id": "gradient-cxcl12-chemotaxis",
        "goal": (
            "Build a 500 µm source–sink CXCL12 gradient: 500 µM source, 0 µM sink, "
            "diffusivity 5e-10 m²/s, 24 h chemotaxis. Report the steepness (µM/mm), "
            "mid-gap concentration (µM), diffusive relaxation time (s) and "
            "steady-state flux (mol/m²/s)."
        ),
        "raw": {"gradient": {
            "chemoattractant": "CXCL12", "source_conc_um": 500.0, "sink_conc_um": 0.0,
            "distance_um": 500.0, "experiment_hours": 24.0, "diffusivity_m2s": 5e-10,
        }},
        "expected": ["steepness_um_per_mm", "midpoint_conc_um", "relaxation_time_s",
                     "flux_mol_m2s"],
    },
    {
        "id": "gradient-fgf8-pattern",
        "goal": (
            "A 2 mm FGF8 morphogen gradient (200 µM source, 0 sink, 5e-10 m²/s) is "
            "used for 48 h patterning. Give the steepness (µM/mm), midpoint "
            "concentration (µM), relaxation time (s) and flux (mol/m²/s)."
        ),
        "raw": {"gradient": {
            "chemoattractant": "FGF8", "source_conc_um": 200.0, "sink_conc_um": 0.0,
            "distance_um": 2000.0, "experiment_hours": 48.0, "diffusivity_m2s": 5e-10,
        }},
        "expected": ["steepness_um_per_mm", "midpoint_conc_um", "relaxation_time_s",
                     "flux_mol_m2s"],
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "gold_new_domains.json"))
    args = ap.parse_args()

    entries: list[dict] = []
    for spec in _SPECS:
        domain = next(k for k in spec["raw"])
        payload = {"goal": spec["goal"], "rationale": "complete-info gold",
                   "caveats": [], **spec["raw"]}
        result = submit_design(payload)
        plan = DesignPlan(**result["design"])
        issues = result["verification"]
        errors = [i for i in issues if i["level"] == "error"]
        if errors:
            print(f"[FAIL] {spec['id']}: verifier errors: {errors}", file=sys.stderr)
            return 2
        expected: dict[str, float] = {}
        claimed = _read(plan, domain)
        for key in spec["expected"]:
            if key not in claimed:
                print(f"[FAIL] {spec['id']}: derived key {key} missing", file=sys.stderr)
                return 2
            expected[key] = claimed[key]
        warnings = [i["message"] for i in issues if i["level"] == "warning"]
        if warnings:
            print(f"[warn] {spec['id']}: {warnings}", file=sys.stderr)
        entries.append({
            "id": spec["id"],
            "goal": spec["goal"],
            "expected": expected,
            "source": _SOURCES[domain],
            "scenario": "complete-info",
        })
        print(f"[ok] {spec['id']}: {', '.join(f'{k}={v:g}' for k, v in expected.items())}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nsaved -> {args.out}  ({len(entries)} entries)")
    return 0


def _read(plan: DesignPlan, domain: str) -> dict[str, float]:
    """Read the block's derived values off the plan (bare keys)."""
    obj = getattr(plan, domain)
    keys = {
        "barrier": ("teer_ohm_cm2", "papp_cm_s", "clearance_mL_min"),
        "oxygen": ("dissolved_o2_mM", "penetration_depth_um", "necrotic_fraction",
                   "demand_umol_min"),
        "pumpless": ("hydrostatic_head_pa", "driven_flow_rate_uLmin", "peak_wall_shear_pa",
                     "volume_per_half_cycle_ul", "oscillatory_shear_index",
                     "cycles_per_hour", "shear_ratio_to_target"),
        "breathing": ("breaths_per_minute", "cyclic_displacement_um", "strain_rate_per_s",
                      "total_cycles", "stretch_duty_fraction", "ali_liquid_film_um"),
        "pulsatile": ("womersley_number", "oscillatory_shear_index", "peak_shear_pa",
                      "pulsatility_index"),
        "scaling": ("organ_flow_fraction", "organ_flow_rate_mlmin", "cells_in_organ",
                    "allometric_scale", "transit_time_s", "residence_time_match_error_s"),
        "gradient": ("steepness_um_per_mm", "midpoint_conc_um", "relaxation_time_s",
                     "flux_mol_m2s"),
    }[domain]
    out: dict[str, float] = {}
    for k in keys:
        v = getattr(obj, k)
        if v is not None and not isinstance(v, bool):
            out[k] = float(v)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
