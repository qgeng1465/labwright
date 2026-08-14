"""Data plumbing for the extractor fine-tune — chat rendering and loss masking.

The extractor maps an experimental *goal* (prose) to the *raw* design inputs
only — never the derived numbers, which always come from the calculators. We
fine-tune a chat-instruct model with a masked LM objective: loss is computed
**only on the assistant's JSON turn**, so everything the model says is scored
against the exact raw-JSON rendering rather than the instruction or the goal.
"""

from __future__ import annotations

import json

#: System instruction for the extractor. Deliberately mirrors Labwright's
#: raw/derived split: the model reports raw inputs and never touches a number
#: the calculators own.
SYSTEM_PROMPT = (
    "You extract raw wet-lab design inputs from an experimental goal. "
    "Return a single JSON object with ONLY the raw input block: chip, flow "
    "and cells for a microfluidic channel design, culture for a plate-culture "
    "design, spheroid for a 3D-spheroid design, pk for a perfused-system "
    "pharmacokinetics design, barrier for a monolayer QC design, oxygen for a "
    "dissolved-pO2 design, pumpless for a gravity-flow rocking platform, "
    "breathing for a lung ALI/stretch design, pulsatile for a cardiac-waveform "
    "design, scaling for a body-on-chip allometry design, or gradient for a "
    "chemotaxis source-sink design. Do NOT compute or report derived numbers "
    "such as wall shear stress, Reynolds number, seed counts, medium volumes, "
    "confluence, spheroid diameter, extraction ratio, clearance, TEER, "
    "penetration depth, Womersley number, organ flow fraction or gradient "
    "steepness — those are always calculated deterministically after you "
    "return. Return ONLY the JSON object."
)

#: The same contract, with the exact key names spelled out. The fine-tuned
#: extractor learned this from its training rows; untrained API baselines are
#: told it here so the comparison tests value extraction, not key-name guessing.
SCHEMA_PROMPT = SYSTEM_PROMPT + (
    "\n\nUse EXACTLY these key names and units in the JSON:\n"
    "  chip (microfluidic channel): width_um (µm), height_um (µm), length_mm (mm)\n"
    "  flow (perfusion): flow_rate_uLmin (µL/min per channel), viscosity_pas (Pa·s), "
    "density_kgm3 (kg/m³)\n"
    "  cells: cell_type, seeding_density_cells_cm2 (cells/cm²), culture_area_cm2 (cm²)\n"
    "  culture (plate design): plate_format "
    "('6-well'|'12-well'|'24-well'|'48-well'|'96-well'), wells (integer), cell_type, "
    "seeding_density_cells_cm2 (cells/cm²), and optionally viability_pct, "
    "confluent_density_cells_cm2, doubling_time_h, culture_duration_h\n"
    "  spheroid (3D-spheroid design): cell_type, spheroid_format "
    "('96-ula'|'384-ula'|'hanging-drop'), spheroid_count (integer), "
    "cells_per_spheroid (integer), cell_diameter_um (µm), and optionally "
    "doubling_time_h, culture_duration_h\n"
    "  pk (perfused-system pharmacokinetics): compound, molecular_weight_g_mol "
    "(g/mol), inlet_concentration_uM (µM), outlet_concentration_uM (µM), "
    "flow_rate_uLmin (µL/min), and optionally system_volume_uL (µL), "
    "dose_interval_h (h)\n"
    "  barrier (monolayer QC): cell_type, insert_area_cm2 (cm²), "
    "resistance_total_ohm (Ω), resistance_blank_ohm (Ω), and optionally probe "
    "(string), donor_conc_um (µM), flux_nmol_min (nmol/min)\n"
    "  oxygen (dissolved pO2): cell_type, target_po2_mmhg (mmHg), and "
    "optionally cell_density_cells_ml (cells/mL), spheroid_diameter_um (µm)\n"
    "  pumpless (gravity-flow rocking): cell_type, tilt_angle_deg (deg), "
    "channel_length_mm (mm), width_um (µm), height_um (µm), "
    "rocking_half_period_s (s), and optionally viscosity_pas (Pa·s), "
    "density_kgm3 (kg/m³), backward_shear_fraction\n"
    "  breathing (lung ALI + stretch): cell_type, frequency_hz (Hz), "
    "strain_pct (%), membrane_span_um (µm), and optionally "
    "culture_duration_h (h), apical_volume_ul (µL), surface_area_cm2 (cm²), "
    "stretch_seconds (s), cycle_seconds (s)\n"
    "  pulsatile (cardiac waveform): cell_type, frequency_hz (Hz), "
    "channel_height_um (µm), shear_mean_pa (Pa), shear_amplitude_pa (Pa), and "
    "optionally peak_flow_uLmin, minimum_flow_uLmin, mean_flow_uLmin (µL/min)\n"
    "  scaling (body-on-chip): organ, total_cells_chip (cells), and optionally "
    "cardiac_output_mlmin (mL/min), body_mass_g (g), chip_volume_ul (µL), "
    "flow_rate_uLmin (µL/min), target_transit_s (s)\n"
    "  gradient (chemotaxis): chemoattractant, source_conc_um (µM), "
    "sink_conc_um (µM), distance_um (µm), and optionally experiment_hours (h), "
    "diffusivity_m2s (m²/s)\n"
    "Emit exactly one design block — never two."
)

#: ChatML assistant-turn opener, as rendered by the Qwen2.5 template. The loss
#: mask boundary is found by locating this marker in the rendered text.
_ASSISTANT_MARKER = "<|im_start|>assistant\n"


def render_chat(goal: str, raw_json_str: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Build the three-turn chat for one (goal → raw) example."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
        {"role": "assistant", "content": raw_json_str},
    ]


def encode_example(
    tokenizer,
    goal: str,
    raw_json_str: str,
    max_len: int = 2048,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict | None:
    """Tokenize one example and build ``input_ids``/``labels`` with masked loss.

    Labels carry ``-100`` for everything before the assistant turn, so the
    model only learns from its own JSON output. Returns ``None`` when the
    assistant content is truncated away entirely (a skip, not a bug).
    """
    msgs = render_chat(goal, raw_json_str, system_prompt=system_prompt)
    full = tokenizer.apply_chat_template(msgs, tokenize=False)
    start = full.find(_ASSISTANT_MARKER)
    if start < 0:
        raise ValueError(
            "assistant-turn marker not found in rendered chat — is the "
            "tokenizer's chat template ChatML-style?"
        )
    start += len(_ASSISTANT_MARKER)

    enc = tokenizer(full, return_offsets_mapping=True, max_length=max_len, truncation=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    cut = len(input_ids)
    for i, (s, _e) in enumerate(offsets):
        if s >= start:
            cut = i
            break
    labels = [-100] * cut + input_ids[cut:]
    if not any(l != -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def raw_to_json(raw: dict) -> str:
    """Deterministic, compact JSON rendering of a raw input block."""
    return json.dumps(raw, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
