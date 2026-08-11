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
"""

from __future__ import annotations

from typing import Any

from labwright.calc import microfluidics as mf
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


def verify_published_protocol(
    chip: dict[str, float],
    flow: dict[str, float],
    claimed: dict[str, float],
    reference: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Check that a paper's claimed derived numbers follow from its own inputs.

    Parameters
    ----------
    chip : dict
        Raw channel geometry as reported: ``width_um``, ``height_um``,
        ``length_mm``.
    flow : dict
        Raw flow inputs as reported: ``flow_rate_uLmin``, ``viscosity_pas``,
        ``density_kgm3``.
    claimed : dict
        The derived values the paper asserts, e.g. ``{"shear_pa": 0.05,
        "reynolds": 0.3, "channel_volume_ul": 0.8}``.
    reference : str
        DOI / journal / patent the claims come from — mandatory for provenance.
    tolerance : float
        Relative tolerance below which a claim is ``consistent``.

    Returns
    -------
    dict
        ``{"status": "ok" | "review_required", "reference", "checks": [...]}``.
        Each check has ``field``, ``computed``, ``claimed``, ``relative_error``
        and ``verdict`` in ``{"consistent", "discrepancy", "not_claimed"}``.
    """
    if not reference:
        return {"status": "validation_error", "error": "reference (DOI/journal) is required"}
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

    checks: list[dict[str, Any]] = []
    for name, calc in METRICS.items():
        computed = calc(*args[name])
        if name not in claimed:
            checks.append(
                {
                    "field": name,
                    "computed": round(float(computed), 6),
                    "claimed": None,
                    "relative_error": None,
                    "verdict": "not_claimed",
                }
            )
            continue
        claimed_value = float(claimed[name])
        rel = abs(computed - claimed_value) / abs(computed) if computed else float("inf")
        checks.append(
            {
                "field": name,
                "computed": round(float(computed), 6),
                "claimed": claimed_value,
                "relative_error": round(float(rel), 4),
                "verdict": "consistent" if rel <= tolerance else "discrepancy",
            }
        )

    discrepancies = [c for c in checks if c["verdict"] == "discrepancy"]
    return {
        "status": "review_required" if discrepancies else "ok",
        "reference": reference,
        "tolerance_pct": tolerance * 100,
        "checks": checks,
        "n_discrepancies": len(discrepancies),
    }


__all__ = ["DEFAULT_TOLERANCE", "METRICS", "verify_published_protocol"]
