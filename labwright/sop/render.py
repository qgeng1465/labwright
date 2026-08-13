"""Render a verified :class:`DesignPlan` as a Markdown SOP.

Every number in the output is copied from the plan's *derived* fields (which
came from the calculators) or from the user's raw inputs — never from the
language model. The SOP is deterministic given the plan.

Each bolded derived number is audited in the trailing *Computation
provenance* section (:mod:`labwright.sop.provenance`): formula, input values
with units, output unit, code version and the verifier's verdict.
"""

from __future__ import annotations

from labwright.schema.design import DesignPlan
from labwright.sop.provenance import sop_provenance_section
from labwright.verify.checker import Issue, has_errors


def design_to_sop(plan: DesignPlan, issues: list[Issue] | None = None) -> str:
    """Full markdown protocol for a verified design plan.

    ``issues`` is the verifier's report (``AgentResult.verification``). A design
    with unresolved *errors* is refused entirely — no protocol to follow. A
    design with *warnings* renders with a prominent "check before the bench"
    banner and per-field verdicts in the provenance block.
    """
    if issues is not None and has_errors(issues):
        return _error_sop(plan, issues)
    lines: list[str] = [
        f"# SOP: {plan.goal}",
        "",
        f"_{plan.rationale}_",
        "",
    ]
    section = 1

    if plan.chip is not None and plan.flow is not None and plan.derived is not None and plan.cells is not None:
        d = plan.derived
        lines += [
            f"## {section}. Device & channel",
            "",
            f"- Geometry: {plan.chip.width_um:.0f} µm wide × {plan.chip.height_um:.0f} µm high × "
            f"{plan.chip.length_mm:.0f} mm long, material {plan.chip.material}",
            f"- Culture volume per channel: **{d.channel_volume_ul:.2f} µL**",
            "",
            f"## {section + 1}. Perfusion",
            "",
            f"- Flow rate: **{plan.flow.flow_rate_uLmin:.2f} µL/min** per channel",
            f"- Wall shear stress: **{d.shear_pa:.3f} Pa** ({d.shear_pa * 10:.2f} dyn/cm²)",
            f"- Reynolds number: {d.reynolds:.2f} {_reynolds_note(d.reynolds)}",
            f"- Pressure drop: {d.pressure_drop_pa:.1f} Pa — verify the pump can hold this",
            f"- Mean residence time: {d.residence_time_s:.1f} s",
            f"- Mean velocity: {d.mean_velocity_mms:.2f} mm/s",
            "",
            f"## {section + 2}. Cell seeding",
            "",
            f"- Cell type: {plan.cells.cell_type}",
            f"- Seeding density: {plan.cells.seeding_density_cells_cm2:g} cells/cm² over "
            f"{plan.cells.culture_area_cm2:.3f} cm²",
            f"- **Seed {plan.cells.seed_count:g} cells** per channel",
        ]
        if plan.cells.doubling_time_h:
            lines.append(
                f"- Doubling time {plan.cells.doubling_time_h:g} h; culture duration "
                f"{plan.cells.culture_duration_h or '—'} h",
            )
        section += 3

    if plan.culture is not None:
        cu = plan.culture
        lines += [
            f"## {section}. Plate culture",
            "",
            f"- Format: **{cu.plate_format}-well plate**, {cu.wells} well(s), {cu.cell_type}",
            f"- Seeding density: {cu.seeding_density_cells_cm2:g} cells/cm²",
            f"- **Seed {cu.seed_per_well:g} cells per well** ({cu.total_seed_count:g} total)",
            f"- Medium: **{cu.medium_volume_per_well_ml:g} mL per well** "
            f"({cu.total_medium_ml:g} mL total)",
        ]
        if cu.viability_pct is not None:
            lines.append(f"- Post-thaw/passage viability: {cu.viability_pct:g}%")
        if cu.expected_confluence_pct is not None:
            lines.append(f"- Predicted confluence at harvest: **{cu.expected_confluence_pct:.1f}%**"
                         + (" ⚠ over-confluent" if cu.expected_confluence_pct > 100 else ""))
        elif cu.doubling_time_h is not None and cu.confluent_density_cells_cm2 is not None:
            lines.append("- Confluence prediction: need culture_duration_h (add it to plan the harvest day)")
        section += 1

    if plan.dosing is not None:
        dos = plan.dosing
        lines += [
            "",
            f"## {section}. Compound dosing",
            "",
            f"- Compound: {dos.compound} (MW {dos.molecular_weight_g_mol:g} g/mol)",
            f"- Stock: {dos.stock_mM:g} mM",
            f"- Working dose: **{dos.working_mM:g} mM**",
            f"- DMSO carry-over: {dos.dmso_fraction_vv * 100:.2f}% v/v"
            + (" ⚠ above 0.5% — solvent-toxicity risk" if dos.dmso_fraction_vv > 0.005 else ""),
            f"- Matched vehicle control: {'yes' if dos.vehicle_control else 'no'}",
        ]
        if dos.exposure_h:
            lines.append(f"- Exposure: {dos.exposure_h:g} h")
        section += 1

    if plan.stats is not None:
        st = plan.stats
        lines += [
            "",
            f"## {section}. Statistical design",
            "",
            f"- Assumed effect {st.effect_size:g} (σ = {st.std_dev:g}); α = {st.alpha:g}, target power {st.power:g}",
            f"- **{st.n_per_group} biological replicates per group**",
        ]
        if st.note:
            lines.append(f"- Note: {st.note}")
        section += 1

    if plan.caveats:
        lines += ["", f"## {section}. Caveats to check in the lab"] + [f"- {c}" for c in plan.caveats]

    warnings = [i for i in (issues or []) if i.level == "warning"]
    if warnings:
        lines += ["", f"## {section}. Verification warnings — check before the bench"]
        lines += [f"- **{i.field}**: {i.message}" for i in warnings]
        section += 1

    # Provenance of every bolded derived number (formula / inputs / units /
    # code version / verification status) — appended before the footer so a
    # bench scientist can audit each number without leaving the protocol.
    lines.append(sop_provenance_section(plan, issues))

    lines += [
        "---",
        f"*Generated by Labwright (qgeng1465). All computed numbers were produced and verified by "
        "deterministic calculators; the language model proposed only the raw inputs.*",
    ]
    return "\n".join(lines)


def _error_sop(plan: DesignPlan, issues: list[Issue]) -> str:
    """A design with unresolved verification errors is not a protocol."""
    lines = [
        f"# SOP: {plan.goal}",
        "",
        "**⛔ Not verified — this design has unresolved verification errors and must not be used.**",
        "",
        "The verifier found errors the design did not correct:",
        "",
    ]
    for i in issues:
        if i.level != "error":
            continue
        detail = (
            f" (expected {i.expected:.6g}, found {i.found:.6g})"
            if i.expected is not None and i.found is not None
            else ""
        )
        lines.append(f"- **{i.field}**: {i.message}{detail}")
    lines += [
        "",
        "*Generated by Labwright. Do not follow this SOP — fix the errors and re-verify.*",
    ]
    return "\n".join(lines)


def _reynolds_note(re: float) -> str:
    """Laminar claim derived from the computed Reynolds number, not asserted."""
    if re >= 2300:
        return "⚠ Re ≥ 2300 — flow may be turbulent; the parallel-plate laminar shear formula no longer applies"
    if re >= 100:
        return f"(laminar, Re {re:.0f} < 2300)"
    return "(laminar, Re << 2300)"


__all__ = ["design_to_sop"]
