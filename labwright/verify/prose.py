"""Prose-number gate — the last place a hallucinated number could hide.

The arithmetic verifier re-derives every *field* of a design, but a design also
carries free-text prose (``rationale``, ``caveats``) in which the agent could
*assert* a derived number that contradicts the calculators — "shear ≈ 0.05 Pa"
in the rationale while the derived shear is 0.12 Pa. Nothing would flag it.

This layer extracts every number-with-unit written in prose and cross-checks it
against the plan's own values in the same physical dimension. An asserted number
that matches no raw or derived value in that dimension is a warning: it cannot
be reproduced from this design, so it is either a hallucination or an unsupported
assumption the user must be told about.

The gate is deliberately conservative — it only judges dimensions the plan
actually carries (no pressure fields -> "0.05 Pa" in prose is not judged) and it
emits warnings, never errors, so an honest design is never blocked by it. It is
the safety property that turns the benchmark's ``hallucination_rate == 0.000``
from "0 by construction" into "0 and attack-tested": a number typed in prose is
now checked, not trusted.
"""

from __future__ import annotations

import re
from typing import Any

from labwright.schema.design import DesignPlan
from labwright.verify.checker import Issue

#: Relative tolerance for a prose number to count as "this design's value".
#: A prose restatement of a derived value is not expected to reproduce it
#: exactly, but it should be within ~5% (the same tolerance the benchmark uses
#: to score a recovered target).
_PROSE_MATCH_TOL = 0.05

#: Physical dimension of every numeric raw/derived field. A prose number is
#: judged only in dimensions the plan actually carries.
_FIELD_DIMENSIONS: dict[str, str] = {
    # flow
    "derived.shear_pa": "pressure",
    "derived.pressure_drop_pa": "pressure",
    "derived.residence_time_s": "time_s",
    "derived.channel_volume_ul": "volume_ul",
    "derived.mean_velocity_mms": "velocity",
    "flow_rate_uLmin": "flow_rate",
    "viscosity_pas": "viscosity",
    "density_kgm3": "density",
    # geometry
    "width_um": "length_um",
    "height_um": "length_um",
    "length_mm": "length_mm",
    # cells
    "cells.seed_count": "count",
    "cells.seeding_density_cells_cm2": "density_cm2",
    "cells.culture_area_cm2": "area_cm2",
    "cells.doubling_time_h": "time_h",
    "cells.culture_duration_h": "time_h",
    # culture
    "culture.seed_per_well": "count",
    "culture.total_seed_count": "count",
    "culture.medium_volume_per_well_ml": "volume_ml",
    "culture.total_medium_ml": "volume_ml",
    "culture.seeding_density_cells_cm2": "density_cm2",
    "culture.expected_confluence_pct": "pct",
    "culture.viability_pct": "pct",
    "culture.doubling_time_h": "time_h",
    "culture.culture_duration_h": "time_h",
    # spheroid
    "spheroid.cells_per_spheroid": "count",
    "spheroid.spheroid_count": "count",
    "spheroid.cell_diameter_um": "length_um",
    "spheroid.expected_diameter_um": "length_um",
    "spheroid.spheroid_volume_ul": "volume_ul",
    "spheroid.medium_volume_per_spheroid_ul": "volume_ul",
    "spheroid.total_medium_ml": "volume_ml",
    "spheroid.cells_total": "count",
    "spheroid.expected_cells_after_growth": "count",
    # dosing
    "dosing.stock_mM": "conc_mm",
    "dosing.working_mM": "conc_mm",
    "dosing.molecular_weight_g_mol": "gmol",
    "dosing.exposure_h": "time_h",
    # stats
    "stats.n_per_group": "count",
}

#: Canonical (pint) unit of each dimension. Prose numbers are converted into it
#: before comparison, so "0.5 dyn/cm²" is judged against shear_pa just like
#: "0.05 Pa" is.
_DIM_CANONICAL: dict[str, str] = {
    "pressure": "Pa",
    "time_s": "s",
    "volume_ul": "uL",
    "volume_ml": "mL",
    "velocity": "mm/s",
    "flow_rate": "uL/min",
    "viscosity": "Pa*s",
    "density": "kg/m**3",
    "length_um": "um",
    "length_mm": "mm",
    "count": "cells",
    "density_cm2": "cells/cm**2",
    "area_cm2": "cm**2",
    "time_h": "h",
    "pct": "%",
    "conc_mm": "mM",
    "gmol": "g/mol",
}

#: Prose unit tokens -> the dimension they express, most-specific first so
#: "mm/s" beats "mm", "µL/min" beats "µL", "Pa·s" beats "Pa".
_UNIT_TO_DIM: list[tuple[str, str]] = [
    (r"mm/s", "velocity"),
    (r"[µμ]L/min|uL/min|ul/min", "flow_rate"),
    (r"[µμ]m|um", "length_um"),
    (r"[µμ]L|uL|ul", "volume_ul"),
    (r"mL|ml", "volume_ml"),
    (r"dyn/cm²|dyn/cm2|dyn/cm\^2|dyn/cm", "pressure"),
    (r"Pa·s|Pa\.s|Pa s\b|Pa\*s", "viscosity"),
    (r"Pa", "pressure"),
    (r"cells/cm²|cells/cm2|cells/cm\^2|cells/cm", "density_cm2"),
    (r"cm²|cm2|cm\^2", "area_cm2"),
    (r"mM", "conc_mm"),
    (r"g/mol", "gmol"),
    (r"kg/m", "density"),
    (r"cells", "count"),
    (r"mm", "length_mm"),
    (r"hours?|hrs?|h\b", "time_h"),
    (r"s\b", "time_s"),
]

_NUM_UNIT_RE = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*("
    + "|".join(p for p, _ in _UNIT_TO_DIM)
    + r")"
)

#: Words that mark a number as a *threshold* rather than an assertion of this
#: design's value ("above 400 µm", "up to 24 h", "keep ≤ 0.1 %"). A bound is a
#: piece of domain knowledge, not a claimed design number, so it is not judged.
_THRESHOLD_RE = re.compile(
    r"(above|below|over|under|exceed(?:ing|s|ed)?|less\s+than|greater\s+than|"
    r"more\s+than|up\s+to|at\s+least|at\s+most|max|min|<|>|≤|≥)"
)


def _dimension_for_token(token: str) -> str | None:
    """Map a captured unit token back to its dimension (regex-first-match)."""
    for pattern, dim in _UNIT_TO_DIM:
        if re.fullmatch(pattern, token):
            return dim
    return None


def _is_threshold_bound(text: str, start: int) -> bool:
    """True when the prose right before the number marks it as a bound."""
    window = text[max(0, start - 12):start]
    return bool(_THRESHOLD_RE.search(window))


def _pint_unit(token: str) -> str | None:
    """Normalise a prose unit token into something pint can convert from."""
    t = token.replace("μ", "u").replace("µ", "u")
    t = t.replace("²", "2").replace("^2", "2").replace("·", "*")
    if t in ("hr", "hrs", "hours"):
        return "h"
    if t == "ul":
        return "uL"
    if t == "ul/min":
        return "uL/min"
    if t.endswith("2"):
        t = t[:-1] + "**2"
    return t


def _to_canonical(number: float, token: str, dim: str) -> float | None:
    """Convert a prose number into the dimension's canonical unit.

    Returns ``None`` when the unit is not convertible — the number is then
    skipped (conservative: we judge only numbers we can compare).
    """
    canonical = _DIM_CANONICAL.get(dim)
    if canonical is None:
        return None
    unit = _pint_unit(token)
    if unit is None or unit == canonical:
        return number
    from labwright.verify.units import convert

    try:
        return convert(number, unit, canonical)
    except Exception:  # noqa: BLE001 - unknown/ambiguous unit, cannot judge
        return None


def _matches_any(number: float, values: list[float]) -> bool:
    return any(
        abs(number - v) <= _PROSE_MATCH_TOL * max(abs(v), 1e-12) for v in values
    )


def _numeric_fields(plan: DesignPlan) -> list[tuple[str, str, float]]:
    """(field, dimension, value) for every numeric raw/derived field present."""
    out: list[tuple[str, str, float]] = []

    def add(field: str, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if v != v or v in (float("inf"), float("-inf")):  # not finite
            return
        dim = _FIELD_DIMENSIONS.get(field)
        if dim is None:
            return
        out.append((field, dim, v))

    if plan.derived is not None:
        for f in (
            "shear_pa", "pressure_drop_pa", "residence_time_s",
            "channel_volume_ul", "mean_velocity_mms",
        ):
            add(f"derived.{f}", getattr(plan.derived, f))
    if plan.chip is not None:
        add("width_um", plan.chip.width_um)
        add("height_um", plan.chip.height_um)
        add("length_mm", plan.chip.length_mm)
    if plan.flow is not None:
        add("flow_rate_uLmin", plan.flow.flow_rate_uLmin)
        add("viscosity_pas", plan.flow.viscosity_pas)
        add("density_kgm3", plan.flow.density_kgm3)
    if plan.cells is not None:
        for f in (
            "seed_count", "seeding_density_cells_cm2", "culture_area_cm2",
            "doubling_time_h", "culture_duration_h",
        ):
            add(f"cells.{f}", getattr(plan.cells, f))
    if plan.culture is not None:
        c = plan.culture
        for f in (
            "seed_per_well", "total_seed_count", "medium_volume_per_well_ml",
            "total_medium_ml", "seeding_density_cells_cm2", "expected_confluence_pct",
            "viability_pct", "doubling_time_h", "culture_duration_h",
        ):
            add(f"culture.{f}", getattr(c, f))
    if plan.spheroid is not None:
        s = plan.spheroid
        for f in (
            "cells_per_spheroid", "spheroid_count", "cell_diameter_um",
            "expected_diameter_um", "spheroid_volume_ul",
            "medium_volume_per_spheroid_ul", "total_medium_ml", "cells_total",
            "expected_cells_after_growth", "doubling_time_h", "culture_duration_h",
        ):
            add(f"spheroid.{f}", getattr(s, f))
    if plan.dosing is not None:
        for f in ("stock_mM", "working_mM", "molecular_weight_g_mol", "exposure_h"):
            add(f"dosing.{f}", getattr(plan.dosing, f))
    if plan.stats is not None:
        add("stats.n_per_group", plan.stats.n_per_group)
    return out


def check_prose_numbers(plan: DesignPlan, issues: list[Issue]) -> None:
    """Flag a number-with-unit written in the design's prose that no calculator produced.

    Cross-checks ``rationale`` and every ``caveats`` entry against the plan's own
    numeric fields (raw and derived). A number whose dimension the plan carries
    but whose value matches none of the plan's values in that dimension is a
    warning — it is asserted but not reproducible from this design.
    """
    fields = _numeric_fields(plan)
    values_by_dim: dict[str, list[float]] = {}
    for _field, dim, value in fields:
        values_by_dim.setdefault(dim, []).append(value)

    texts: list[str] = []
    if plan.rationale:
        texts.append(plan.rationale)
    texts.extend(plan.caveats or [])

    for text in texts:
        for match in _NUM_UNIT_RE.finditer(text):
            if _is_threshold_bound(text, match.start()):
                continue  # "above 400 µm" is a bound, not a claimed value
            number = float(match.group(1))
            token = match.group(2)
            dim = _dimension_for_token(token)
            if dim is None or dim not in values_by_dim:
                continue  # plan carries no value in this dimension -> cannot judge
            canonical = _to_canonical(number, token, dim)
            if canonical is None:
                continue
            if not _matches_any(canonical, values_by_dim[dim]):
                issues.append(
                    Issue(
                        level="warning",
                        field="prose",
                        message=(
                            f"number {number:g} ({token}) in the design text matches no "
                            "value in this design — an asserted number the calculators "
                            "did not produce"
                        ),
                        expected=None,
                        found=number,
                    )
                )


__all__ = ["check_prose_numbers"]
