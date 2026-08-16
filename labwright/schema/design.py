"""Pydantic models describing a complete wet-lab experimental design.

Units are declared in every field name where they are not implicit.
Derived fields (``DerivedFlowMetrics``) are *always* produced by
:mod:`labwright.calc` — never by the language model — and are cross-checked by
the verifier.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChipGeometry(BaseModel):
    """Microfluidic channel geometry."""

    model_config = ConfigDict(extra="forbid")

    width_um: float = Field(gt=0, description="Channel width (µm), typical OOC: 400-1000")
    height_um: float = Field(gt=0, description="Channel height (µm), typical OOC: 50-200")
    length_mm: float = Field(gt=0, description="Channel length (mm)")
    channel_count: int = Field(default=1, ge=1, description="Number of parallel channels")
    material: str = Field(default="PDMS", description="Device material, e.g. PDMS, glass")


class FlowParams(BaseModel):
    """Perfusion inputs."""

    model_config = ConfigDict(extra="forbid")

    flow_rate_uLmin: float = Field(gt=0, description="Per-channel volumetric flow rate (µL/min)")
    viscosity_pas: float = Field(default=1e-3, gt=0, description="Dynamic viscosity (Pa·s)")
    density_kgm3: float = Field(default=1000, gt=0, description="Fluid density (kg/m³)")


class DerivedFlowMetrics(BaseModel):
    """Flow quantities computed deterministically by the calculators."""

    shear_pa: float = Field(description="Wall shear stress (Pa); ×10 = dyn/cm²")
    reynolds: float = Field(description="Reynolds number (laminar if << 2300)")
    pressure_drop_pa: float = Field(description="Laminar pressure drop over the channel (Pa)")
    residence_time_s: float = Field(description="Mean fluid residence time (s)")
    channel_volume_ul: float = Field(description="Per-channel culture volume (µL)")
    mean_velocity_mms: float = Field(description="Mean flow velocity (mm/s)")


class CellPlan(BaseModel):
    """Seeding and culture plan."""

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type, e.g. HepG2, primary hepatocytes")
    seeding_density_cells_cm2: float = Field(gt=0, description="Seeding density (cells/cm²)")
    culture_area_cm2: float = Field(gt=0, description="Effective culture area (cm²)")
    seed_count: float = Field(gt=0, description="Total cells to seed")
    doubling_time_h: float | None = Field(default=None, description="Doubling time if proliferative (h)")
    culture_duration_h: float | None = Field(default=None, description="Planned culture duration (h)")


class CulturePlan(BaseModel):
    """Plate-based culture plan.

    Derived fields (``seed_per_well``, ``total_seed_count``,
    ``medium_volume_per_well_ml``, ``total_medium_ml``,
    ``expected_confluence_pct``) are *always* computed by
    :mod:`labwright.calc.culture` — never proposed by the LLM — and re-checked
    by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    plate_format: str = Field(description="Plate format: 6/12/24/48/96-well")
    wells: int = Field(default=1, ge=1, description="Number of wells plated")
    cell_type: str = Field(description="Cell type")
    seeding_density_cells_cm2: float = Field(gt=0, description="Seeding density (cells/cm²)")
    seed_per_well: float = Field(gt=0, description="DERIVED: density × well surface area")
    total_seed_count: float = Field(gt=0, description="DERIVED: seed_per_well × wells")
    medium_volume_per_well_ml: float = Field(gt=0, description="DERIVED: standard working volume for the format")
    total_medium_ml: float = Field(gt=0, description="DERIVED: per-well volume × wells")
    viability_pct: float | None = Field(default=None, ge=0, le=100, description="Thawed/passaged viability (%)")
    confluent_density_cells_cm2: float | None = Field(default=None, gt=0, description="Cells/cm² at 100% confluence (cell-type dependent — an input)")
    doubling_time_h: float | None = Field(default=None, gt=0, description="Population doubling time (h)")
    culture_duration_h: float | None = Field(default=None, ge=0, description="Culture duration (h)")
    expected_confluence_pct: float | None = Field(default=None, ge=0, description="DERIVED: predicted confluence at harvest (may exceed 100 for over-confluent cultures)")


class SpheroidPlan(BaseModel):
    """3D spheroid/organoid culture plan.

    Derived fields (``spheroid_volume_ul``, ``expected_diameter_um``,
    ``cells_total``, ``medium_volume_per_spheroid_ul``, ``total_medium_ml``,
    ``expected_cells_after_growth``) are *always* computed by
    :mod:`labwright.calc.spheroid` — never proposed by the LLM — and re-checked
    by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type, e.g. HepG2, primary hepatocytes, tumour cells")
    spheroid_format: str = Field(description="Vessel/format: 96-ula / 384-ula / hanging-drop")
    spheroid_count: int = Field(ge=1, description="Number of spheroids to form")
    cells_per_spheroid: float = Field(gt=0, description="Cells seeded per spheroid")
    cell_diameter_um: float = Field(gt=0, description="Mean single-cell diameter (um), used for volume-to-size estimates")
    spheroid_volume_ul: float = Field(gt=0, description="DERIVED: cells_per_spheroid × single-cell volume (solid-sphere packing)")
    expected_diameter_um: float = Field(gt=0, description="DERIVED: spheroid diameter implied by cells_per_spheroid × cell volume")
    cells_total: float = Field(gt=0, description="DERIVED: spheroid_count × cells_per_spheroid")
    medium_volume_per_spheroid_ul: float = Field(gt=0, description="DERIVED: standard working volume for the vessel format")
    total_medium_ml: float = Field(gt=0, description="DERIVED: per-spheroid volume × spheroid_count")
    doubling_time_h: float | None = Field(default=None, gt=0, description="Population doubling time (h), if proliferative")
    culture_duration_h: float | None = Field(default=None, ge=0, description="Culture duration (h)")
    expected_cells_after_growth: float | None = Field(default=None, ge=0, description="DERIVED: predicted cells per spheroid at harvest, N(t) = N0 * 2^(t/td), when growth inputs are present")


class DosePlan(BaseModel):
    """Compound dosing plan."""

    model_config = ConfigDict(extra="forbid")

    compound: str = Field(description="Compound name")
    molecular_weight_g_mol: float = Field(gt=0, description="Molecular weight (g/mol)")
    stock_mM: float = Field(gt=0, description="Stock concentration (mM)")
    working_mM: float = Field(gt=0, description="Working concentration (mM)")
    dmso_fraction_vv: float = Field(ge=0, description="DMSO volume fraction in medium (v/v)")
    vehicle_control: bool = Field(default=True, description="Include matched vehicle control")
    exposure_h: float | None = Field(default=None, description="Exposure duration (h)")


class StatsPlan(BaseModel):
    """Statistical design of the comparison."""

    model_config = ConfigDict(extra="forbid")

    effect_size: float = Field(gt=0, description="Expected between-group difference (measurement units)")
    std_dev: float = Field(gt=0, description="Expected pooled standard deviation")
    alpha: float = Field(default=0.05, gt=0, lt=1)
    power: float = Field(default=0.80, gt=0, lt=1)
    n_per_group: int = Field(ge=1, description="Biological replicates per group")
    note: str | None = Field(default=None, description="Justification / assumption notes")


class PkPlan(BaseModel):
    """Pharmacokinetics of a drug in a perfused organ-on-chip system.

    Derived fields (``extraction_ratio``, ``clearance_uLmin``, and — when the
    extra inputs are present — ``half_life_h``, ``accumulation_ratio``,
    ``mass_cleared_ug_h``) are *always* computed by :mod:`labwright.calc.pk`
    from the inlet/outlet concentrations and flow rate — never proposed by the
    LLM — and re-checked by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    compound: str = Field(description="Drug name")
    molecular_weight_g_mol: float | None = Field(default=None, gt=0, description="Drug molecular weight (g/mol); needed for mass_cleared_ug_h")
    inlet_concentration_uM: float = Field(gt=0, description="Drug concentration entering the chip (µM)")
    outlet_concentration_uM: float = Field(ge=0, description="Drug concentration leaving the chip (µM)")
    flow_rate_uLmin: float = Field(gt=0, description="Perfusion flow rate (µL/min)")
    extraction_ratio: float = Field(description="DERIVED: 1 − C_out/C_in, the fraction cleared per pass")
    clearance_uLmin: float = Field(description="DERIVED: E × Q, volume of medium fully cleared per minute")
    system_volume_uL: float | None = Field(default=None, gt=0, description="Recirculating medium volume (µL); needed for half_life_h")
    half_life_h: float | None = Field(default=None, gt=0, description="DERIVED: ln2·V/Cl, elimination half-life (h)")
    dose_interval_h: float | None = Field(default=None, gt=0, description="Time between doses (h); needed for accumulation_ratio")
    accumulation_ratio: float | None = Field(default=None, ge=1, description="DERIVED: 1/(1 − e^(−ln2·τ/t½)), steady-state accumulation factor")
    mass_cleared_ug_h: float | None = Field(default=None, ge=0, description="DERIVED: Cl·C_in·MW·6e-5, mass the chip clears per hour (µg/h)")


class BarrierPlan(BaseModel):
    """Epithelial/endothelial barrier QC plan (TEER + permeability).

    Derived fields (``teer_ohm_cm2`` and — when the probe flux and donor
    concentration are present — ``papp_cm_s``, ``clearance_mL_min``) are
    *always* computed by :mod:`labwright.calc.barrier` — never proposed by the
    LLM — and re-checked by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type, e.g. Caco-2 (gut), hCMEC/D3 (BBB), Calu-3 (airway)")
    insert_area_cm2: float = Field(gt=0, description="Membrane growth area (cm²); 24-well Transwell ≈ 0.33, 12-well ≈ 1.12")
    resistance_total_ohm: float = Field(gt=0, description="Measured total resistance across the insert (Ω)")
    resistance_blank_ohm: float = Field(gt=0, description="Cell-free insert (electrode + medium) resistance (Ω)")
    teer_ohm_cm2: float = Field(gt=0, description="DERIVED: (R_total − R_blank) × A")
    probe: str | None = Field(default=None, description="Permeability probe, e.g. FITC-dextran 4 kDa")
    donor_conc_um: float | None = Field(default=None, gt=0, description="Donor-chamber probe concentration (µM)")
    flux_nmol_min: float | None = Field(default=None, ge=0, description="Steady-state probe flux across the monolayer (nmol/min)")
    papp_cm_s: float | None = Field(default=None, ge=0, description="DERIVED: flux/(60·A·C₀), apparent permeability (cm/s)")
    clearance_mL_min: float | None = Field(default=None, ge=0, description="DERIVED: Papp·A·60, permeability-surface-area product (mL/min)")


class OxygenPlan(BaseModel):
    """Dissolved-oxygen control for a physioxic / hypoxic chip culture.

    Derived fields (``dissolved_o2_mM`` always, plus ``penetration_depth_um``
    and ``necrotic_fraction`` when a cell density / spheroid diameter is given)
    are *always* computed by :mod:`labwright.calc.o2` — never proposed by the
    LLM — and re-checked by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type; its O2 consumption rate (OCR) comes from the physiology registry")
    target_po2_mmhg: float = Field(gt=0, description="Target O2 partial pressure in the chip (mmHg); tissue in vivo ≈ 8–104, air-equilibrated medium ≈ 150")
    cell_density_cells_ml: float | None = Field(default=None, gt=0, description="Cell density (cells/mL) for consumption / penetration estimates")
    spheroid_diameter_um: float | None = Field(default=None, gt=0, description="Spheroid diameter (µm) when checking necrotic-core risk")
    dissolved_o2_mM: float = Field(ge=0, description="DERIVED: Henry's law from target pO2")
    penetration_depth_um: float | None = Field(default=None, ge=0, description="DERIVED: Krogh O2 penetration depth (µm)")
    necrotic_fraction: float | None = Field(default=None, ge=0, le=1, description="DERIVED: spheroid anoxic-core volume fraction")
    demand_umol_min: float | None = Field(default=None, ge=0, description="DERIVED: O2 demand per 10⁶ cells at the registry OCR (µmol/min)")


class PumplessPlan(BaseModel):
    """Gravity-driven (rocking/tilting) pumpless perfusion plan.

    Derived fields (``hydrostatic_head_pa``, ``flow_rate_uLmin``,
    ``peak_wall_shear_pa``, ``volume_per_half_cycle_ul``,
    ``oscillatory_shear_index``, ``cycles_per_hour``,
    ``shear_ratio_to_target``) are *always* computed by
    :mod:`labwright.calc.pumpless` — never proposed by the LLM — and re-checked
    by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type; its physiological shear range sets the WSS target")
    tilt_angle_deg: float = Field(gt=0, le=25, description="Platform tilt from horizontal (deg); practical rocker limit 25")
    channel_length_mm: float = Field(gt=0, description="Channel length along the tilt axis (mm)")
    width_um: float = Field(gt=0, description="Channel width (µm)")
    height_um: float = Field(gt=0, description="Channel height (µm)")
    rocking_half_period_s: float = Field(gt=0, description="Duration of one half of a rocking cycle (s); organ chips 5-60")
    viscosity_pas: float = Field(default=1e-3, gt=0, description="Dynamic viscosity (Pa·s)")
    density_kgm3: float = Field(default=1000, gt=0, description="Fluid density (kg/m³)")
    backward_shear_fraction: float = Field(default=1.0, ge=0, le=1, description="Reverse-direction shear as a fraction of forward (1 = symmetric rocking, 0 = unidirectional Tesla-valve)")
    hydrostatic_head_pa: float = Field(gt=0, description="DERIVED: ρ·g·L·sinθ, the sole driving pressure")
    driven_flow_rate_uLmin: float = Field(gt=0, description="DERIVED: Hagen-Poiseuille flow driven by the head (µL/min)")
    peak_wall_shear_pa: float = Field(gt=0, description="DERIVED: peak wall shear during a rocking half-cycle (Pa)")
    volume_per_half_cycle_ul: float = Field(gt=0, description="DERIVED: volume displaced in one half-cycle (µL)")
    oscillatory_shear_index: float = Field(ge=0, le=0.5, description="DERIVED: 0.5 symmetric rocking, 0 unidirectional")
    cycles_per_hour: float = Field(gt=0, description="DERIVED: full rocking cycles per hour")
    shear_ratio_to_target: float | None = Field(default=None, description="DERIVED: chip WSS / physiological target (0.5-2× in range)")


class BreathingPlan(BaseModel):
    """Lung-on-chip ALI + cyclic mechanical stretch plan.

    Derived fields (``breaths_per_minute``, ``cyclic_displacement_um``,
    ``strain_rate_per_s``, ``total_cycles``, ``stretch_duty_fraction``,
    ``ali_liquid_film_um``) are *always* computed by
    :mod:`labwright.calc.breathing` — never proposed by the LLM — and re-checked
    by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type, e.g. alveolar epithelial (A549, primary)")
    frequency_hz: float = Field(gt=0, description="Breathing/actuation frequency (Hz); physiological 0.2-0.25")
    strain_pct: float = Field(gt=0, description="Applied linear strain (%); 5-12 physiological, >20 pathological")
    membrane_span_um: float = Field(default=250, gt=0, description="Membrane span across the stretch axis (µm)")
    apical_volume_ul: float | None = Field(default=None, ge=0, description="Residual apical liquid volume at ALI (µL)")
    surface_area_cm2: float | None = Field(default=None, gt=0, description="Apical epithelial surface area (cm²)")
    culture_duration_h: float | None = Field(default=None, ge=0, description="Stretch application duration (h); needed for total_cycles")
    stretch_seconds: float | None = Field(default=None, ge=0, description="Time at peak stretch per cycle (s); needed for duty fraction")
    cycle_seconds: float | None = Field(default=None, gt=0, description="Full stretch-cycle period (s); needed for duty fraction")
    breaths_per_minute: float = Field(gt=0, description="DERIVED: f × 60")
    cyclic_displacement_um: float = Field(gt=0, description="DERIVED: ε·L, the membrane edge stroke (µm)")
    strain_rate_per_s: float = Field(gt=0, description="DERIVED: (ε/100)·f, linearised strain rate (1/s)")
    total_cycles: float | None = Field(default=None, ge=0, description="DERIVED: hours × 3600 × f, total stretch cycles")
    stretch_duty_fraction: float | None = Field(default=None, ge=0, le=1, description="DERIVED: stretch/cycle time held deformed")
    ali_liquid_film_um: float | None = Field(default=None, ge=0, description="DERIVED: apical residual film thickness (µm)")


class PulsatilePlan(BaseModel):
    """Pulsatile / cardiac-cycle waveform plan for heart-on-chip.

    Derived fields (``womersley_number``, ``oscillatory_shear_index``,
    ``peak_shear_pa``, and — when the flow-waveform inputs are present —
    ``pulsatility_index``) are *always* computed by
    :mod:`labwright.calc.pulsatile` — never proposed by the LLM — and re-checked
    by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(description="Cell type, e.g. endothelial or valvular")
    frequency_hz: float = Field(gt=0, description="Cardiac-cycle frequency (Hz); 0.8-2 ≈ 48-120 bpm")
    channel_height_um: float = Field(gt=0, description="Channel height (µm); half-height is the effective Womersley radius")
    viscosity_pas: float = Field(default=1e-3, gt=0, description="Dynamic viscosity (Pa·s)")
    density_kgm3: float = Field(default=1000, gt=0, description="Fluid density (kg/m³)")
    shear_mean_pa: float = Field(ge=0, description="Time-averaged wall shear of the waveform (Pa)")
    shear_amplitude_pa: float = Field(ge=0, description="Sinusoidal shear amplitude (Pa); reversal when amp > mean")
    peak_flow_uLmin: float | None = Field(default=None, ge=0, description="Peak flow over the cycle (µL/min); needed for pulsatility_index")
    minimum_flow_uLmin: float | None = Field(default=None, ge=0, description="Minimum flow over the cycle (µL/min); needed for pulsatility_index")
    mean_flow_uLmin: float | None = Field(default=None, gt=0, description="Time-averaged flow (µL/min); needed for pulsatility_index")
    womersley_number: float = Field(gt=0, description="DERIVED: flow unsteadiness α = (h/2)·√(ωρ/μ)")
    oscillatory_shear_index: float = Field(ge=0, le=0.5, description="DERIVED: flow-reversal fraction (0 none, 0.5 fully reversing)")
    peak_shear_pa: float = Field(ge=0, description="DERIVED: mean + amplitude, the waveform peak (Pa)")
    pulsatility_index: float | None = Field(default=None, ge=0, description="DERIVED: Gosling (Q_peak − Q_min)/Q_mean")


class ScalingPlan(BaseModel):
    """Multi-organ body-on-chip allometric scaling plan (one organ compartment).

    Derived fields (``organ_flow_fraction``, ``organ_flow_rate_mlmin``,
    ``cells_in_organ``, ``allometric_scale``, and — when the compartment volume
    and perfusion flow are given — ``transit_time_s``,
    ``residence_time_match_error_s``) are *always* computed by
    :mod:`labwright.calc.scaling` from the physiology tables — never proposed by
    the LLM — and re-checked by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    organ: str = Field(description="Organ name: liver, kidneys, brain, heart, gut, skin, muscle, lungs")
    total_cells_chip: float = Field(gt=0, description="Chip-wide cell budget being scaled (cells)")
    cardiac_output_mlmin: float = Field(default=5000, gt=0, description="Cardiac output the chip is scaled to (mL/min); adult ≈ 5000")
    body_mass_g: float = Field(default=70000, gt=0, description="Reference body mass (g); adult ≈ 70000")
    chip_volume_ul: float | None = Field(default=None, ge=0, description="Compartment (channel + chamber) volume (µL); needed for transit time")
    flow_rate_uLmin: float | None = Field(default=None, gt=0, description="Perfusion flow through the compartment (µL/min); needed for transit time")
    target_transit_s: float | None = Field(default=None, ge=0, description="In-vivo organ transit time (s); needed for the match error")
    organ_flow_fraction: float = Field(gt=0, lt=1, description="DERIVED: organ's share of cardiac output (liver 0.27)")
    organ_flow_rate_mlmin: float = Field(gt=0, description="DERIVED: fraction × cardiac output (mL/min)")
    cells_in_organ: float = Field(gt=0, description="DERIVED: (m_organ/m_body) × chip budget (cells)")
    allometric_scale: float = Field(gt=0, description="DERIVED: (m_organ/m_body)^0.75, Kleiber metabolic factor")
    transit_time_s: float | None = Field(default=None, ge=0, description="DERIVED: V/Q·60, perfusate transit (s)")
    residence_time_match_error_s: float | None = Field(default=None, ge=0, description="DERIVED: |transit − target|, flow-side objective residual (s)")


class GradientPlan(BaseModel):
    """Steady concentration-gradient (chemotaxis) plan.

    Derived fields (``steepness_um_per_mm``, ``midpoint_conc_um``,
    ``relaxation_time_s``, ``flux_mol_m2s``) are *always* computed by
    :mod:`labwright.calc.gradient` — never proposed by the LLM — and re-checked
    by the verifier.
    """

    model_config = ConfigDict(extra="forbid")

    chemoattractant: str = Field(description="Solute, e.g. CXCL12, fMLP, EGF")
    source_conc_um: float = Field(gt=0, description="Chemoattractant concentration in the source channel (µM)")
    sink_conc_um: float = Field(ge=0, description="Buffer concentration in the sink channel (µM)")
    distance_um: float = Field(gt=0, description="Source-to-sink gap, the diffusive bridge (µm); classic ~1000")
    experiment_hours: float = Field(gt=0, description="Planned experiment duration (h); stability needs ≥ 10τ")
    diffusivity_m2s: float = Field(default=5e-10, gt=0, description="Solute diffusivity in the bridge (m²/s); small-molecule estimate")
    steepness_um_per_mm: float = Field(gt=0, description="DERIVED: (C_src − C_sink)/L × 1000 (µM/mm)")
    midpoint_conc_um: float = Field(ge=0, description="DERIVED: mid-gap steady-state concentration (µM)")
    relaxation_time_s: float = Field(gt=0, description="DERIVED: L²/D, gradient formation time (s)")
    flux_mol_m2s: float = Field(ge=0, description="DERIVED: D·(C_src − C_sink)/L, steady-state flux (mol/m²/s)")


class BioprintingPlan(BaseModel):
    """Micro-extrusion bioprinting plan (G-code move → deposited ink).

    Derived fields (``extrusion_volume_nl``, ``print_time_s``,
    ``extrusion_rate_nl_min``, ``filament_mass_ug``, ``lines_to_cover``) are
    *always* computed by :mod:`labwright.calc.bioprinting` — never proposed by
    the LLM — and re-checked by the verifier. The nozzle diameter comes from the
    registered nozzle table, an equipment-spec convention.
    """

    model_config = ConfigDict(extra="forbid")

    nozzle_id: str = Field(description="Registered nozzle id / alias, e.g. 'nozzle_3', '3', 'cryo3', 'uv5'")
    travel_distance_um: float = Field(gt=0, description="G-code path travel distance (µm); a 10 mm move = 10000")
    feed_rate_mm_min: float = Field(gt=0, description="Print feed rate (mm/min)")
    density_g_cm3: float = Field(default=1.0, gt=0, description="Ink density (g/cm³); cell-laden hydrogels ≈ 1")
    footprint_width_um: float | None = Field(default=None, gt=0, description="Footprint width to fill with lines (µm); needed for lines_to_cover")
    line_pitch_um: float | None = Field(default=None, gt=0, description="Centre-to-centre fill-line pitch (µm); needed for lines_to_cover")
    extrusion_volume_nl: float = Field(gt=0, description="DERIVED: π(d/2)²·L, ink volume over the path (nL)")
    print_time_s: float = Field(gt=0, description="DERIVED: L/v, traversal time of the move (s)")
    extrusion_rate_nl_min: float = Field(gt=0, description="DERIVED: volume / time, deposition rate (nL/min)")
    filament_mass_ug: float = Field(gt=0, description="DERIVED: volume × ink density (µg)")
    lines_to_cover: float | None = Field(default=None, ge=1, description="DERIVED: ceil(footprint_width / line_pitch), fill-line count")


class CoculturePlan(BaseModel):
    """Two-population co-culture seeding plan (liver-lobule / mixed models).

    Derived fields (``cells_per_well_a``, ``cells_per_well_b``,
    ``total_cells_a``, ``total_cells_b``, ``seeding_ratio_ab``) are *always*
    computed by :mod:`labwright.calc.coculture` — never proposed by the LLM —
    and re-checked by the verifier. The A-fraction is the designer's stated
    choice; the total density is the stated seeding budget.
    """

    model_config = ConfigDict(extra="forbid")

    cell_type_a: str = Field(description="Population A, e.g. HUVEC-T1")
    cell_type_b: str = Field(description="Population B, e.g. HepG2")
    total_density_cells_cm2: float = Field(gt=0, description="Total seeding density across both populations (cells/cm²)")
    area_cm2: float = Field(gt=0, description="Culture surface area per well (cm²)")
    fraction_a: float = Field(gt=0, lt=1, description="Fraction of the total assigned to population A")
    wells: int = Field(default=1, ge=1, description="Number of wells plated")
    cells_per_well_a: float = Field(gt=0, description="DERIVED: f·ρ·A, population A cells per well")
    cells_per_well_b: float = Field(gt=0, description="DERIVED: (1−f)·ρ·A, population B cells per well")
    total_cells_a: float = Field(gt=0, description="DERIVED: per-well A × wells")
    total_cells_b: float = Field(gt=0, description="DERIVED: per-well B × wells")
    seeding_ratio_ab: float = Field(gt=0, description="DERIVED: A cells / B cells")


class EnzymePlan(BaseModel):
    """Competitive-inhibition reaction plan (OA + UDPGA class of question).

    Derived fields (``fractional_activity``, ``percent_inhibition``,
    ``ic50_um``, ``apparent_km_um``, ``inhibitor_substrate_ratio``, and —
    when Vmax is supplied — ``velocity_umol_min``) are *always* computed by
    :mod:`labwright.calc.enzyme` — never proposed by the LLM — and re-checked
    by the verifier. Km/Ki are stated inputs, never invented here.
    """

    model_config = ConfigDict(extra="forbid")

    enzyme: str = Field(description="Enzyme / target protein, e.g. UGT2B7")
    substrate: str = Field(description="Substrate / cofactor, e.g. UDPGA")
    km_um: float = Field(gt=0, description="Michaelis constant (µM)")
    s_conc_um: float = Field(gt=0, description="Substrate concentration in the mix (µM)")
    ki_um: float = Field(gt=0, description="Inhibitor dissociation constant (µM)")
    i_conc_um: float = Field(ge=0, description="Inhibitor concentration in the mix (µM)")
    vmax_umol_min: float | None = Field(default=None, gt=0, description="Maximum reaction velocity (µmol/min); needed for velocity_umol_min")
    fractional_activity: float = Field(gt=0, le=1, description="DERIVED: [S]/(Km(1+[I]/Ki)+[S]), fraction of uninhibited rate remaining")
    percent_inhibition: float = Field(ge=0, le=100, description="DERIVED: (1 − v_i/v_0)·100")
    ic50_um: float = Field(gt=0, description="DERIVED: Ki(1 + [S]/Km), run-condition IC50 (Cheng-Prusoff)")
    apparent_km_um: float = Field(gt=0, description="DERIVED: Km(1 + [I]/Ki)")
    velocity_umol_min: float | None = Field(default=None, ge=0, description="DERIVED: Vmax × fractional activity (µmol/min), when Vmax supplied")
    inhibitor_substrate_ratio: float = Field(ge=0, description="DERIVED: [I]/[S] molar ratio")


class ChampPlan(BaseModel):
    """ChAMP methylation-array batch plan (cohort → BeadChips).

    Derived fields (``n_arrays``, ``n_chips``, ``n_expected_failed_arrays``)
    are *always* computed by :mod:`labwright.calc.bioinformatics` — never
    proposed by the LLM — and re-checked by the verifier. Platform capacities
    are Illumina product conventions (software spec), not literature values.
    """

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(ge=1, description="Cohort size (samples)")
    platform: str = Field(description="BeadChip platform: '450k' or 'epic'")
    fail_rate_pct: float | None = Field(default=None, ge=0, le=100, description="Expected array QC fail rate (%); needed for n_expected_failed_arrays")
    n_arrays: int = Field(ge=1, description="DERIVED: one array per sample")
    n_chips: int = Field(ge=1, description="DERIVED: ceil(n_samples / chip capacity) physical BeadChips")
    n_expected_failed_arrays: float | None = Field(default=None, ge=0, description="DERIVED: n_arrays × fail_rate, expected QC failures")


class PlinkPlan(BaseModel):
    """PLINK genotype-batch plan (cohort × variants → dataset size).

    Derived fields (``bed_size_mb``, ``n_per_chr_files``,
    ``per_chr_bed_size_mb``) are *always* computed by
    :mod:`labwright.calc.bioinformatics` — never proposed by the LLM — and
    re-checked by the verifier. The 2-bits/sample/variant ``.bed`` format and
    the 25 chromosome files are PLINK 1.9 software conventions.
    """

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(ge=1, description="Genotyped sample count")
    n_variants: int = Field(ge=1, description="Variant count across the dataset")
    n_variants_chr: int | None = Field(default=None, ge=1, description="Variant count on one chromosome; needed for per_chr_bed_size_mb")
    bed_size_mb: float = Field(gt=0, description="DERIVED: n_samples·n_variants/4/1e6, binary .bed size (MB)")
    n_per_chr_files: int = Field(ge=1, description="DERIVED: 25 standard per-chromosome files (1–22, X, Y, MT)")
    per_chr_bed_size_mb: float | None = Field(default=None, ge=0, description="DERIVED: one chromosome's .bed size (MB), when n_variants_chr given")


class SolventPlan(BaseModel):
    """Hanging-drop / multi-well solvent-evaporation plan.

    Derived fields (``evaporation_rate_ul_hr``, ``residual_volume_ul``,
    ``edge_evaporation_factor``) are *always* computed by
    :mod:`labwright.calc.solvent` — never proposed by the LLM — and re-checked
    by the verifier. The edge effect is a documented-range parameter, not a
    hidden measurement.
    """

    model_config = ConfigDict(extra="forbid")

    drop_volume_ul: float = Field(gt=0, description="Initial hanging-drop volume (µL)")
    hours: float = Field(ge=0, description="Elapsed evaporation time (h)")
    temp_c: float = Field(ge=0, le=50, description="Ambient temperature (°C)")
    rh: float = Field(ge=0, le=1, description="Relative humidity (fraction, 0–1)")
    well_row: str = Field(description="96-well row (A–H)")
    well_col: int = Field(ge=1, le=12, description="96-well column (1–12)")
    edge_factor: float | None = Field(default=None, ge=1, le=3, description="Plate-edge evaporation factor override (documented range 1.4–2.0); default 1.5")
    evaporation_rate_ul_hr: float = Field(gt=0, description="DERIVED: interior Langmuir rate × edge factor (µL/hr)")
    residual_volume_ul: float = Field(ge=0, description="DERIVED: d²-law residual after the stated time (µL)")
    edge_evaporation_factor: float = Field(ge=1, description="DERIVED: 1.5× for A/H rows or 1/12 columns, else 1.0")


class DesignPlan(BaseModel):
    """Top-level output of the Labwright agent."""

    goal: str = Field(description="Restatement of the experimental goal")
    rationale: str = Field(description="Why this design; key assumptions")
    chip: ChipGeometry | None = Field(default=None, description="Channel geometry (absent for plate-only culture designs)")
    flow: FlowParams | None = Field(default=None, description="Perfusion inputs (absent for plate-only culture designs)")
    derived: DerivedFlowMetrics | None = Field(default=None, description="Deterministically computed flow metrics")
    cells: CellPlan | None = Field(default=None, description="Channel-based cell plan (absent for plate-only culture designs)")
    culture: CulturePlan | None = Field(default=None, description="Plate-based culture plan (only when plating on multi-well plates)")
    spheroid: SpheroidPlan | None = Field(default=None, description="3D spheroid/organoid culture plan (only when forming spheroids)")
    dosing: DosePlan | None = None
    stats: StatsPlan | None = None
    pk: PkPlan | None = Field(default=None, description="Perfused-system pharmacokinetics plan (only when studying drug clearance)")
    barrier: BarrierPlan | None = Field(default=None, description="Epithelial/endothelial barrier QC plan (only when measuring a monolayer)")
    oxygen: OxygenPlan | None = Field(default=None, description="Dissolved-oxygen control plan (only when O2 is a design lever)")
    pumpless: PumplessPlan | None = Field(default=None, description="Gravity-driven pumpless perfusion plan (only on a rocking platform)")
    breathing: BreathingPlan | None = Field(default=None, description="Lung ALI + cyclic stretch plan (only for a breathing lung chip)")
    pulsatile: PulsatilePlan | None = Field(default=None, description="Pulsatile cardiac-waveform plan (only for a heart-on-chip)")
    scaling: ScalingPlan | None = Field(default=None, description="Multi-organ allometric scaling plan (only in a body-on-chip)")
    gradient: GradientPlan | None = Field(default=None, description="Concentration-gradient chemotaxis plan (only for a gradient generator)")
    bioprinting: BioprintingPlan | None = Field(default=None, description="Micro-extrusion bioprinting plan (only when printing ink along G-code paths)")
    coculture: CoculturePlan | None = Field(default=None, description="Two-population co-culture seeding plan (only when plating mixed populations)")
    enzyme: EnzymePlan | None = Field(default=None, description="Competitive-inhibition reaction plan (only when a Km/Ki competition is quantified)")
    champ: ChampPlan | None = Field(default=None, description="ChAMP methylation batch plan (only when sizing methylation BeadChips)")
    plink: PlinkPlan | None = Field(default=None, description="PLINK genotype batch plan (only when sizing binary .bed datasets)")
    solvent: SolventPlan | None = Field(default=None, description="Hanging-drop / solvent-evaporation plan (only when tracking evaporation)")
    caveats: list[str] = Field(default_factory=list, description="Things to verify in the lab")
