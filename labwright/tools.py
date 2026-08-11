"""Tool registry — the only way Labwright's agent reaches the calculators.

The registry is deliberately tiny. A :class:`Tool` binds three things:

- a ``pydantic`` parameter model (validates input and produces the JSON Schema
  the LLM needs for tool calling),
- a pure function from :mod:`labwright.calc` (the actual math),
- prose that tells the LLM when to call it.

**Extending Labwright** (this is the documented extension point): write a
calculator in :mod:`labwright.calc`, then add a ``Tool`` here. Nothing else
needs to change. The agent, the verifier and the demo all read the same
registry, so a new calculator is instantly callable, verifiable and
demonstrable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from labwright.calc import cell, dosing, microfluidics as mf, stats
from labwright.published import verify_published_protocol

# ---------------------------------------------------------------------------
# Registry primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A callable calculator exposed to the agent."""

    name: str
    description: str
    params_model: type[BaseModel]
    func: Callable[..., Any]
    category: str
    units_out: str = ""

    @property
    def schema(self) -> dict[str, Any]:
        """OpenAI-style function-tool schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_model.model_json_schema(),
            },
        }

    def call(self, **kwargs: Any) -> Any:
        """Validate arguments against the model, then run the calculator."""
        return self.func(**self.params_model(**kwargs).model_dump())


REGISTRY: dict[str, Tool] = {}


def register_tool(cls: type[BaseModel], name: str, description: str, func: Callable, category: str, units_out: str = ""):
    """Declare a tool in the global registry."""
    t = Tool(name=name, description=description, params_model=cls, func=func, category=category, units_out=units_out)
    REGISTRY[name] = t
    return t


# ---------------------------------------------------------------------------
# Parameter models
# ---------------------------------------------------------------------------


class ShearStressParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min")
    width_um: float = Field(gt=0, description="Channel width in micrometres")
    height_um: float = Field(gt=0, description="Channel height in micrometres")
    viscosity_pas: float = Field(gt=0, description="Dynamic viscosity in Pa·s (water ≈ 1e-3, culture medium ≈ 0.9-1.1e-3)")


class FlowForShearParams(BaseModel):
    target_shear_pa: float = Field(gt=0, description="Target wall shear stress in Pa (physiological microvascular ≈ 0.01-0.1 Pa)")
    width_um: float = Field(gt=0, description="Channel width in micrometres")
    height_um: float = Field(gt=0, description="Channel height in micrometres")
    viscosity_pas: float = Field(gt=0, description="Dynamic viscosity in Pa·s")


class ReynoldsParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min")
    width_um: float = Field(gt=0, description="Channel width in micrometres")
    height_um: float = Field(gt=0, description="Channel height in micrometres")
    viscosity_pas: float = Field(gt=0, description="Dynamic viscosity in Pa·s")


class PressureDropParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min")
    width_um: float = Field(gt=0, description="Channel width in micrometres")
    height_um: float = Field(gt=0, description="Channel height in micrometres")
    length_mm: float = Field(gt=0, description="Channel length in millimetres")
    viscosity_pas: float = Field(gt=0, description="Dynamic viscosity in Pa·s")


class ResidenceTimeParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min")
    width_um: float = Field(gt=0, description="Channel width in micrometres")
    height_um: float = Field(gt=0, description="Channel height in micrometres")
    length_mm: float = Field(gt=0, description="Channel length in millimetres")


class O2DeliveryParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min")
    o2_in_mol_L: float = Field(gt=0, description="Dissolved O2 concentration entering channel in mol/L (air-equilibrated ≈ 0.2e-3)")
    o2_out_mol_L: float = Field(ge=0, default=0.0, description="Dissolved O2 leaving channel in mol/L")


class SeedCountParams(BaseModel):
    density_cells_cm2: float = Field(gt=0, description="Seeding density in cells/cm^2 (HepG2 ≈ 5e4-1e5, primary hepatocytes ≈ 1e5)")
    area_cm2: float = Field(gt=0, description="Culture area in cm^2 (a 400 µm × 20 mm × 100 µm channel ≈ 0.08 cm^2)")


class GrowthParams(BaseModel):
    seed_count: float = Field(gt=0, description="Cells seeded at t=0")
    doubling_time_h: float = Field(gt=0, description="Population doubling time in hours (HepG2 ≈ 30-40 h)")
    elapsed_h: float = Field(ge=0, description="Elapsed time in hours")


class ConfluenceTimeParams(BaseModel):
    seed_count: float = Field(gt=0, description="Seeded cell count")
    confluence_count: float = Field(gt=0, description="Cell count at desired confluence")
    doubling_time_h: float = Field(gt=0, description="Doubling time in hours")


class MolarityParams(BaseModel):
    mass_mg: float = Field(gt=0, description="Mass of compound in mg")
    molecular_weight_g_mol: float = Field(gt=0, description="Molecular weight in g/mol")
    volume_ml: float = Field(gt=0, description="Final volume in mL")


class DilutionParams(BaseModel):
    stock_mM: float = Field(gt=0, description="Stock concentration in mM")
    target_mM: float = Field(gt=0, description="Desired working concentration in mM")
    target_volume_ml: float = Field(gt=0, description="Final working volume in mL")


class DmsoParams(BaseModel):
    stock_dmso_mM: float = Field(gt=0, description="Compound concentration in pure DMSO stock, mM")
    working_mM: float = Field(gt=0, description="Desired working concentration in medium, mM")


class SampleSizeParams(BaseModel):
    effect_size: float = Field(gt=0, description="Expected difference between group means (measurement units)")
    std_dev: float = Field(gt=0, description="Expected pooled standard deviation (same units as effect_size)")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Type-I error rate")
    power: float = Field(default=0.80, gt=0, lt=1, description="Target statistical power")
    two_sided: bool = Field(default=True, description="Two-sided test (recommended)")


class ReplicatesParams(BaseModel):
    cv_pct: float = Field(gt=0, description="Assay coefficient of variation in percent")
    precision_pct: float = Field(gt=0, description="Desired relative precision (95% CI half-width) in percent")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Confidence level")


class ChipClaimParams(BaseModel):
    width_um: float = Field(gt=0, description="Channel width in micrometres, as reported")
    height_um: float = Field(gt=0, description="Channel height in micrometres, as reported")
    length_mm: float = Field(gt=0, description="Channel length in millimetres, as reported")


class FlowClaimParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min, as reported")
    viscosity_pas: float = Field(default=1e-3, gt=0, description="Dynamic viscosity in Pa·s")
    density_kgm3: float = Field(default=1000, gt=0, description="Fluid density in kg/m³")


class VerifyProtocolParams(BaseModel):
    chip: ChipClaimParams = Field(description="Channel geometry the paper reports")
    flow: FlowClaimParams = Field(description="Flow inputs the paper reports")
    claimed: dict[str, float] = Field(
        description='Derived values the paper asserts, e.g. {"shear_pa": 0.05, "reynolds": 0.3, '
        '"channel_volume_ul": 0.8}'
    )
    reference: str = Field(description="DOI / journal citation of the paper being checked (required)")


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------

register_tool(
    ShearStressParams,
    "wall_shear_stress",
    "Wall shear stress on cells in a wide rectangular microchannel: tau = 6·mu·Q/(w·h^2). "
    "Call whenever the experiment must match a physiological shear range, or to compute shear "
    "from an intended flow rate.",
    mf.wall_shear_stress,
    "microfluidics",
    units_out="Pa",
)

register_tool(
    FlowForShearParams,
    "flow_rate_for_shear_stress",
    "Inverse calculation: what volumetric flow rate (µL/min) achieves a target wall shear stress "
    "in a given rectangular channel. Use to set a syringe-pump flow from a physiological shear target.",
    mf.flow_rate_for_shear_stress,
    "microfluidics",
    units_out="µL/min",
)

register_tool(
    ReynoldsParams,
    "reynolds_number",
    "Reynolds number in a rectangular channel; sanity check that flow is laminar (microfluidics: << 2300).",
    mf.reynolds_number,
    "microfluidics",
    units_out="dimensionless",
)

register_tool(
    PressureDropParams,
    "pressure_drop",
    "Laminar pressure drop along a rectangular channel, dP = 12·mu·Q·L/(w·h^3). Check the pump can "
    "supply the needed pressure, or estimate pressure-driven flow in a gravity setup.",
    mf.pressure_drop,
    "microfluidics",
    units_out="Pa",
)

register_tool(
    ResidenceTimeParams,
    "residence_time",
    "Mean fluid residence time in the channel (volume / flow). Relevant for dosing and medium-exchange "
    "frequency in perfused culture.",
    mf.residence_time,
    "microfluidics",
    units_out="s",
)

register_tool(
    O2DeliveryParams,
    "o2_delivery_rate",
    "Oxygen delivery rate to cells by perfusion, n_dot = Q·(C_in - C_out). First-order check that "
    "perfusion meets cell O2 demand (air-equilibrated medium ≈ 0.2 mM O2).",
    mf.o2_delivery_rate,
    "microfluidics",
    units_out="µmol/min",
)

register_tool(
    SeedCountParams,
    "seeding_cell_count",
    "Cells to seed onto a culture area at a given density: N = density × area. Plan chip seeding.",
    cell.seeding_cell_count,
    "cell",
    units_out="cells",
)

register_tool(
    GrowthParams,
    "cell_count_after_time",
    "Exponential growth prediction N(t) = N0·2^(t/td). Estimate cell numbers at harvest time.",
    cell.cell_count_after_time,
    "cell",
    units_out="cells",
)

register_tool(
    ConfluenceTimeParams,
    "time_to_confluence",
    "Hours until culture reaches a target confluent cell count, given seeding count and doubling time.",
    cell.time_to_confluence,
    "cell",
    units_out="h",
)

register_tool(
    MolarityParams,
    "molarity_from_mass",
    "Molar concentration from dissolving a weighed mass: C = m/(MW·V). Prepare a compound stock.",
    dosing.molarity_from_mass,
    "dosing",
    units_out="mM",
)

register_tool(
    DilutionParams,
    "dilution_volume",
    "Volume of a more-concentrated stock to add to reach a target working concentration (C1·V1 = C2·V2).",
    dosing.dilution_volume,
    "dosing",
    units_out="mL",
)

register_tool(
    DmsoParams,
    "dmso_fraction",
    "Volume fraction of DMSO in the final medium at a working dose (stock in DMSO / working conc). "
    "Flag if above ~0.1-0.5% v/v (solvent toxicity).",
    dosing.dmso_fraction,
    "dosing",
    units_out="v/v fraction",
)

register_tool(
    SampleSizeParams,
    "sample_size_per_group",
    "Biological replicates per group for a two-sample t-test to detect a given effect at target power. "
    "Call for every controlled comparison; underpowered designs are the top cause of non-reproducible results.",
    stats.sample_size_per_group,
    "stats",
    units_out="n per group",
)

register_tool(
    ReplicatesParams,
    "technical_replicates",
    "Technical replicates per condition to bound the relative standard error of the mean (assay precision).",
    stats.technical_replicates,
    "stats",
    units_out="n",
)

register_tool(
    VerifyProtocolParams,
    "verify_published_protocol",
    "Sanity-check a *published* protocol: recompute the derived numbers (shear, Reynolds, pressure drop, "
    "residence time, channel volume, velocity) from the geometry and flow a paper reports, and flag any "
    "claimed value that does not follow from the paper's own inputs. Use before copying a chip design "
    "or a reported shear/flow pair from the literature.",
    verify_published_protocol,
    "published",
    units_out="verdict per field",
)


def list_tools() -> list[Tool]:
    """All registered tools, grouped by category."""
    return sorted(REGISTRY.values(), key=lambda t: (t.category, t.name))


def tools_for_llm() -> list[dict[str, Any]]:
    """Registry serialized for the LLM function-calling API."""
    return [t.schema for t in list_tools()]


__all__ = ["Tool", "REGISTRY", "register_tool", "list_tools", "tools_for_llm"]
