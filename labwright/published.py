"""Verify the internal consistency of a *reported* protocol.

The reproducibility crisis is partly a numbers crisis: a paper states a
channel geometry, a flow rate and a wall-shear claim, and no reader has a
cheap way to check that the three agree. ``verify_published_protocol``
recomputes every derived number from the raw inputs a paper reports and
returns a per-field verdict.

It is the *reverse* of :func:`labwright.design.submit_design`. Design asks
"given raw inputs, what are the outputs?"; this asks "do the outputs this
paper reports follow from its own inputs?" — the check that turns a design
copilot into a literature sanity-checker.

Since WS3 the checker also understands plate-culture protocols: pass the
reported ``culture`` block (plate_format, seeding_density_cells_cm2, optional
wells/viability/confluent_density/doubling/culture_duration) and it recomputes
seed-per-well, per-well medium volume and (when the growth inputs are reported)
expected confluence. Chip/flow and culture may be given together or alone.
"""

from __future__ import annotations

from typing import Any

from labwright.calc import microfluidics as mf
from labwright.design import derive_culture
from labwright.schema.design import ChipGeometry, FlowParams

#: Default tolerance for a claimed number to count as "consistent" with the
#: recomputed value. Literature numbers are rarely published to more precision.
DEFAULT_TOLERANCE = 0.05  # ±5 %

#: Derived quantities the checker recomputes and compares.
METRICS: dict[str, Any] = {
    "shear_pa": mf.wall_shear_stress,
    "reynolds": mf.reynolds_number,
    "pressure_drop_pa": mf.pressure_drop,
    "residence_time_s": mf.residence_time,
    "channel_volume_ul": mf.channel_volume,
    "mean_velocity_mms": mf.mean_velocity,
}

#: Derived culture quantities the checker recomputes and compares. The three
#: numbers a culture paper is most likely to state; ``total_seed_count`` /
#: ``total_medium_ml`` follow from seed_per_well * wells but are stated less
#: often. ``expected_confluence_pct`` is only recomputable when the paper also
#: reports doubling time, confluent density and culture duration.
CULTURE_METRICS = ("seed_per_well", "medium_volume_per_well_ml", "expected_confluence_pct")


def _check(name: str, computed: float | None, claimed_value: float | None, tolerance: float) -> dict[str, Any]:
    """Build one per-field check record.

    ``computed`` is ``None`` when the reported inputs are insufficient to
    recompute the quantity (e.g. a confluence claim without growth inputs) — in
    that case a present claim is ``unverifiable`` rather than silently accepted.
    """
    if computed is None:
        return {
            "field": name,
            "computed": None,
            "claimed": claimed_value,
            "relative_error": None,
            "verdict": "not_claimed" if claimed_value is None else "unverifiable",
        }
    if claimed_value is None:
        return {
            "field": name,
            "computed": round(float(computed), 6),
            "claimed": None,
            "relative_error": None,
            "verdict": "not_claimed",
        }
    claimed_value = float(claimed_value)
    rel = abs(computed - claimed_value) / abs(computed) if computed else float("inf")
    return {
        "field": name,
        "computed": round(float(computed), 6),
        "claimed": claimed_value,
        "relative_error": round(float(rel), 4),
        "verdict": "consistent" if rel <= tolerance else "discrepancy",
    }


def verify_published_protocol(
    chip: dict[str, float] | None = None,
    flow: dict[str, float] | None = None,
    culture: dict[str, float] | None = None,
    claimed: dict[str, float] | None = None,
    reference: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Check that a paper's claimed derived numbers follow from its own inputs.

    Parameters
    ----------
    chip : dict, optional
        Raw channel geometry as reported: ``width_um``, ``height_um``,
        ``length_mm``.
    flow : dict, optional
        Raw flow inputs as reported: ``flow_rate_uLmin``, ``viscosity_pas``,
        ``density_kgm3``.
    culture : dict, optional
        Raw plate-culture inputs as reported: ``plate_format``,
        ``seeding_density_cells_cm2``, and any of ``wells``, ``viability_pct``,
        ``confluent_density_cells_cm2``, ``doubling_time_h``,
        ``culture_duration_h``.
    claimed : dict
        The derived values the paper asserts, e.g. ``{"shear_pa": 0.05,
        "reynolds": 0.3, "channel_volume_ul": 0.8}`` or ``{"seed_per_well":
        3200}``. Keys not in :data:`METRICS` / :data:`CULTURE_METRICS` are
        ignored.
    reference : str
        DOI / journal / patent the claims come from — mandatory for provenance.
    tolerance : float
        Relative tolerance below which a claim is ``consistent``.

    Returns
    -------
    dict
        ``{"status": "ok" | "review_required" | "unverifiable" | "validation_error",
        "reference", "checks": [...]}``. Each check has ``field``, ``computed``,
        ``claimed``, ``relative_error`` and ``verdict`` in
        ``{"consistent", "discrepancy", "not_claimed", "unverifiable"}``.

        ``unverifiable`` means at least one *present* claim could not be
        recomputed from the reported inputs (e.g. a confluence figure with no
        growth inputs), so it is neither confirmed nor contradicted.
    """
    if not reference:
        return {"status": "validation_error", "error": "reference (DOI/journal) is required"}
    claimed = claimed or {}
    checks: list[dict[str, Any]] = []

    # Fluid domain — chip + flow both reported.
    fluid = bool(chip) and bool(flow)
    if fluid:
        try:
            chip_model = ChipGeometry(**chip)
            flow_model = FlowParams(**flow)
        except Exception as exc:  # noqa: BLE001 - report to the caller/LLM
            return {"status": "validation_error", "error": str(exc)}

        args = {
            "shear_pa": (flow_model.flow_rate_uLmin, chip_model.width_um, chip_model.height_um, flow_model.viscosity_pas),
            "reynolds": (
                flow_model.flow_rate_uLmin,
                chip_model.width_um,
                chip_model.height_um,
                flow_model.viscosity_pas,
                flow_model.density_kgm3,
            ),
            "pressure_drop_pa": (
                flow_model.flow_rate_uLmin,
                chip_model.width_um,
                chip_model.height_um,
                chip_model.length_mm,
                flow_model.viscosity_pas,
            ),
            "residence_time_s": (
                flow_model.flow_rate_uLmin,
                chip_model.width_um,
                chip_model.height_um,
                chip_model.length_mm,
            ),
            "channel_volume_ul": (chip_model.width_um, chip_model.height_um, chip_model.length_mm),
            "mean_velocity_mms": (flow_model.flow_rate_uLmin, chip_model.width_um, chip_model.height_um),
        }
        for name, calc in METRICS.items():
            checks.append(_check(name, calc(*args[name]), claimed.get(name), tolerance))

    # Culture domain.
    if culture:
        try:
            derived = derive_culture(culture)
        except Exception as exc:  # noqa: BLE001 - report to the caller/LLM
            return {"status": "validation_error", "error": str(exc)}
        for name in CULTURE_METRICS:
            checks.append(_check(name, derived.get(name), claimed.get(name), tolerance))

    if not checks:
        return {
            "status": "validation_error",
            "error": "no inputs: report chip+flow (fluidics) and/or culture (plate)",
        }

    discrepancies = [c for c in checks if c["verdict"] == "discrepancy"]
    unverifiable = [c for c in checks if c["verdict"] == "unverifiable"]
    if discrepancies:
        status = "review_required"
    elif unverifiable:
        status = "unverifiable"
    else:
        status = "ok"
    return {
        "status": status,
        "reference": reference,
        "tolerance_pct": tolerance * 100,
        "checks": checks,
        "n_discrepancies": len(discrepancies),
    }


__all__ = ["DEFAULT_TOLERANCE", "METRICS", "CULTURE_METRICS", "verify_published_protocol"]
