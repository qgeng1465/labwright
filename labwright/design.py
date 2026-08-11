"""Design construction — the boundary between the LLM and the math.

The agent proposes **raw inputs** (geometry, flow, cells, dosing, statistics
assumptions). It never writes a derived number. :func:`build_design` derives
every computed quantity through the calculators, then :func:`submit_design`
runs the verifier.

This split is what makes Labwright's numbers trustworthy: an LLM that asserts
``shear_pa=0.25`` can be wrong; a design whose ``shear_pa`` was produced by
:mod:`labwright.calc` cannot.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from labwright.calc import cell as calc_cell
from labwright.calc import microfluidics as mf
from labwright.schema.design import (
    CellPlan,
    ChipGeometry,
    DerivedFlowMetrics,
    DesignPlan,
    DosePlan,
    FlowParams,
    StatsPlan,
)
from labwright.verify.checker import format_issues, verify_design


# ---------------------------------------------------------------------------
# Raw input the LLM is allowed to propose
# ---------------------------------------------------------------------------


class DesignInput(BaseModel):
    """Everything the agent is allowed to choose. No derived numbers here."""

    goal: str = Field(description="Experimental goal in one sentence")
    rationale: str = Field(description="Why this design; assumptions and references")
    chip: ChipGeometry
    flow: FlowParams
    cells: dict[str, Any] = Field(
        description="cell_type, seeding_density_cells_cm2, culture_area_cm2, "
        "doubling_time_h, culture_duration_h (no seed_count)"
    )
    dosing: dict[str, Any] | None = Field(
        default=None,
        description="compound, molecular_weight_g_mol, stock_mM, working_mM, "
        "vehicle_control, exposure_h (no dmso_fraction_vv)",
    )
    stats: dict[str, Any] | None = Field(
        default=None,
        description="effect_size, std_dev, alpha, power (no n_per_group; it is computed)",
    )
    caveats: list[str] = Field(default_factory=list, description="What must be checked in the lab")


# ---------------------------------------------------------------------------
# Derivation (the math)
# ---------------------------------------------------------------------------


def build_design(inp: DesignInput) -> DesignPlan:
    """Derive every computed field from the agent's raw inputs."""
    # Flow metrics — all recomputed here, never accepted from the LLM.
    derived = DerivedFlowMetrics(
        shear_pa=mf.wall_shear_stress(
            inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um, inp.flow.viscosity_pas
        ),
        reynolds=mf.reynolds_number(
            inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um,
            inp.flow.viscosity_pas, inp.flow.density_kgm3,
        ),
        pressure_drop_pa=mf.pressure_drop(
            inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um,
            inp.chip.length_mm, inp.flow.viscosity_pas,
        ),
        residence_time_s=mf.residence_time(
            inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um, inp.chip.length_mm
        ),
        channel_volume_ul=mf.channel_volume(
            inp.chip.width_um, inp.chip.height_um, inp.chip.length_mm
        ),
        mean_velocity_mms=mf.mean_velocity(
            inp.flow.flow_rate_uLmin, inp.chip.width_um, inp.chip.height_um
        ),
    )

    cells = CellPlan(
        **inp.cells,
        seed_count=calc_cell.seeding_cell_count(
            inp.cells["seeding_density_cells_cm2"], inp.cells["culture_area_cm2"]
        ),
    )

    dosing = None
    if inp.dosing is not None:
        d = dict(inp.dosing)
        # DMSO fraction is derived from stock/working, never asserted.
        from labwright.calc import dosing as calc_dosing

        d["dmso_fraction_vv"] = calc_dosing.dmso_fraction(d["stock_mM"], d["working_mM"])
        dosing = DosePlan(**d)

    stats = None
    if inp.stats is not None:
        from labwright.calc import stats as calc_stats

        s = dict(inp.stats)
        s["n_per_group"] = calc_stats.sample_size_per_group(s["effect_size"], s["std_dev"], s["alpha"], s["power"])
        stats = StatsPlan(**s)

    plan = DesignPlan(
        goal=inp.goal,
        rationale=inp.rationale,
        chip=inp.chip,
        flow=inp.flow,
        derived=derived,
        cells=cells,
        dosing=dosing,
        stats=stats,
        caveats=inp.caveats,
    )
    return plan


def submit_design(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate raw input, derive everything, verify, and report.

    This is the final tool the agent calls. The result carries the complete
    design plus the verifier's report, so a design with unresolved errors is
    visible to the agent and can be corrected in a follow-up turn.
    """
    inp = DesignInput(**input_dict)
    plan = build_design(inp)
    issues = verify_design(plan)
    return {
        "design": plan.model_dump(mode="json"),
        "verification": [i.__dict__ for i in issues],
        "verification_summary": format_issues(issues),
        "status": "ok" if not issues else "review_required",
    }


__all__ = ["DesignInput", "build_design", "submit_design"]
