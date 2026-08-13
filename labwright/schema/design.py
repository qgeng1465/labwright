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
    caveats: list[str] = Field(default_factory=list, description="Things to verify in the lab")
