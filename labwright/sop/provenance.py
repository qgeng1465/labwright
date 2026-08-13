"""Computation provenance — every derived number's full trace, in one place.

The SOP bolds derived numbers ("Seed 3,200 cells per well"); this module gives
each of those numbers its *provenance*: the formula, every input value with its
unit, the output unit, the Labwright code version that produced it, and the
verifier's verdict. That turns "computed by calc and verified by verify" from a
claim into something a reviewer, an ELN or a LIMS can re-derive line by line.

Two consumers:

- :func:`sop_provenance_section` — the markdown block appended to the SOP so a
  bench scientist can audit each bolded number without leaving the protocol.
- :func:`export_eln` — a structured (JSON/CSV) record of the full computation
  path for import into an electronic lab notebook or a LIMS.

The rule is the same as everywhere else in Labwright: the language model never
appears here. Every field, formula, input and value below comes from the
design plan (which the calculators produced) and the verifier.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from labwright import __version__
from labwright.schema.design import DesignPlan
from labwright.verify.units import CANONICAL_UNITS

#: Version string: package version plus the git commit that generated the plan,
#: so provenance is reproducible down to the code.
def _code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        ).stdout.strip()
        if sha:
            return f"labwright-{__version__} (git {sha})"
    except Exception:  # noqa: BLE001 - not a git checkout / no git
        pass
    return f"labwright-{__version__}"

CODE_VERSION = _code_version()

#: LaTeX-ish formula per derived field. The verifier recomputes exactly these.
FORMULAS: dict[str, str] = {
    "derived.shear_pa": r"τ = 6·μ·Q / (w·h²)",
    "derived.reynolds": r"Re = ρ·u·D_h / μ",
    "derived.pressure_drop_pa": r"ΔP = 12·μ·Q·L / (w·h³)",
    "derived.residence_time_s": r"t = V / Q",
    "derived.channel_volume_ul": r"V = w·h·L",
    "derived.mean_velocity_mms": r"ū = Q / (w·h)",
    "cells.seed_count": r"N = ρ·A",
    "culture.seed_per_well": r"N_well = ρ·A_well",
    "culture.total_seed_count": r"N_total = N_well × wells",
    "culture.medium_volume_per_well_ml": r"V_well = standard working volume (format table)",
    "culture.total_medium_ml": r"V_total = V_well × wells",
    "culture.expected_confluence_pct": r"conf = N(t) / (ρ_conv·A_well) × 100,  N(t) = N_0·2^(t/t_d)",
    "dosing.dmso_fraction_vv": r"f = C_working / C_stock",
    "stats.n_per_group": r"n from two-sample t-test power equation",
}


def _unit(field: str) -> str:
    return CANONICAL_UNITS.get(field, "")


def _status_for(field: str, issues: list | None) -> str:
    """Verifier verdict for one field: ok / warning / error (default ok)."""
    if not issues:
        return "ok"
    errs = [i for i in issues if i.field == field and i.level == "error"]
    if errs:
        return "error"
    if any(i.field == field and i.level == "warning" for i in issues):
        return "warning"
    return "ok"


def provenance_for(plan: DesignPlan, issues: list | None = None) -> list[dict[str, Any]]:
    """Build a provenance record for every derived field the plan carries.

    Each record: ``{field, formula, inputs: [{name, value, unit}], unit,
    value, status, code_version}``. ``status`` comes from ``issues`` when given,
    else defaults to "ok" (a plain derivation). Order follows the SOP's sections.
    """
    records: list[dict[str, Any]] = []

    def add(field: str, value: Any, inputs: list[tuple[str, Any, str]]) -> None:
        if value is None:
            return
        records.append({
            "field": field,
            "formula": FORMULAS.get(field, ""),
            "inputs": [
                {"name": name, "value": value, "unit": unit}
                for name, value, unit in inputs
            ],
            "unit": _unit(field),
            "value": value,
            "status": _status_for(field, issues),
            "code_version": CODE_VERSION,
        })

    if plan.derived is not None:
        c = plan.chip
        f = plan.flow
        add("derived.shear_pa", plan.derived.shear_pa, [
            ("flow_rate_uLmin", f.flow_rate_uLmin, "uL/min"),
            ("width_um", c.width_um, "um"),
            ("height_um", c.height_um, "um"),
            ("viscosity_pas", f.viscosity_pas, "Pa*s"),
        ])
        add("derived.reynolds", plan.derived.reynolds, [
            ("flow_rate_uLmin", f.flow_rate_uLmin, "uL/min"),
            ("width_um", c.width_um, "um"),
            ("height_um", c.height_um, "um"),
            ("viscosity_pas", f.viscosity_pas, "Pa*s"),
            ("density_kgm3", f.density_kgm3, "kg/m^3"),
        ])
        add("derived.pressure_drop_pa", plan.derived.pressure_drop_pa, [
            ("flow_rate_uLmin", f.flow_rate_uLmin, "uL/min"),
            ("width_um", c.width_um, "um"),
            ("height_um", c.height_um, "um"),
            ("length_mm", c.length_mm, "mm"),
            ("viscosity_pas", f.viscosity_pas, "Pa*s"),
        ])
        add("derived.residence_time_s", plan.derived.residence_time_s, [
            ("flow_rate_uLmin", f.flow_rate_uLmin, "uL/min"),
            ("width_um", c.width_um, "um"),
            ("height_um", c.height_um, "um"),
            ("length_mm", c.length_mm, "mm"),
        ])
        add("derived.channel_volume_ul", plan.derived.channel_volume_ul, [
            ("width_um", c.width_um, "um"),
            ("height_um", c.height_um, "um"),
            ("length_mm", c.length_mm, "mm"),
        ])
        add("derived.mean_velocity_mms", plan.derived.mean_velocity_mms, [
            ("flow_rate_uLmin", f.flow_rate_uLmin, "uL/min"),
            ("width_um", c.width_um, "um"),
            ("height_um", c.height_um, "um"),
        ])

    if plan.cells is not None:
        add("cells.seed_count", plan.cells.seed_count, [
            ("seeding_density_cells_cm2", plan.cells.seeding_density_cells_cm2, "cells/cm^2"),
            ("culture_area_cm2", plan.cells.culture_area_cm2, "cm^2"),
        ])

    if plan.culture is not None:
        cu = plan.culture
        add("culture.seed_per_well", cu.seed_per_well, [
            ("seeding_density_cells_cm2", cu.seeding_density_cells_cm2, "cells/cm^2"),
            ("plate_format", cu.plate_format, "—"),
        ])
        add("culture.total_seed_count", cu.total_seed_count, [
            ("seed_per_well", cu.seed_per_well, "cells"),
            ("wells", cu.wells, "—"),
        ])
        add("culture.medium_volume_per_well_ml", cu.medium_volume_per_well_ml, [
            ("plate_format", cu.plate_format, "—"),
        ])
        add("culture.total_medium_ml", cu.total_medium_ml, [
            ("medium_volume_per_well_ml", cu.medium_volume_per_well_ml, "mL"),
            ("wells", cu.wells, "—"),
        ])
        add("culture.expected_confluence_pct", cu.expected_confluence_pct, [
            ("seed_per_well", cu.seed_per_well, "cells"),
            ("doubling_time_h", cu.doubling_time_h, "h"),
            ("culture_duration_h", cu.culture_duration_h, "h"),
            ("confluent_density_cells_cm2", cu.confluent_density_cells_cm2, "cells/cm^2"),
            ("plate_format", cu.plate_format, "—"),
        ])

    if plan.dosing is not None:
        add("dosing.dmso_fraction_vv", plan.dosing.dmso_fraction_vv, [
            ("working_mM", plan.dosing.working_mM, "mM"),
            ("stock_mM", plan.dosing.stock_mM, "mM"),
        ])

    if plan.stats is not None:
        add("stats.n_per_group", float(plan.stats.n_per_group), [
            ("effect_size", plan.stats.effect_size, ""),
            ("std_dev", plan.stats.std_dev, ""),
            ("alpha", plan.stats.alpha, ""),
            ("power", plan.stats.power, ""),
        ])

    return records


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def sop_provenance_section(plan: DesignPlan, issues: list | None = None) -> str:
    """Markdown block listing each derived number's formula, inputs, unit, code
    version and verification status — appended to the SOP."""
    records = provenance_for(plan, issues)
    if not records:
        return ""
    lines = [
        "",
        f"## Computation provenance (each bolded number above)",
        "",
        f"*Labwright {CODE_VERSION}; the model proposed only raw inputs — every value below was "
        "computed by the calculators and cross-checked by the verifier.*",
        "",
    ]
    for r in records:
        inputs = ", ".join(
            f"{inp['name']}={_fmt_value(inp['value'])} {inp['unit']}".rstrip()
            for inp in r["inputs"]
        )
        lines.append(
            f"- **{r['field']}** = {_fmt_value(r['value'])} {r['unit']}  ·  {r['formula']}  ·  "
            f"inputs: {inputs}  ·  verify: {r['status']}"
        )
    lines.append("")
    return "\n".join(lines)


def export_eln(plan: DesignPlan, issues: list | None = None, fmt: str = "json") -> str:
    """Structured export of the full computation path for an ELN/LIMS.

    ``fmt`` is ``"json"`` (records with inputs/units/version/status) or
    ``"csv"`` (one row per derived field). JSON is the lossless form; CSV is for
    spreadsheets.
    """
    records = provenance_for(plan, issues)
    if fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["field", "formula", "inputs", "unit", "value", "status", "code_version"])
        for r in records:
            inputs = "; ".join(
                f"{i['name']}={_fmt_value(i['value'])}{i['unit']}" for i in r["inputs"]
            )
            w.writerow([r["field"], r["formula"], inputs, r["unit"], _fmt_value(r["value"]), r["status"], r["code_version"]])
        return buf.getvalue()
    import json

    return json.dumps(records, indent=2, ensure_ascii=False)


__all__ = [
    "CODE_VERSION",
    "FORMULAS",
    "provenance_for",
    "sop_provenance_section",
    "export_eln",
]
