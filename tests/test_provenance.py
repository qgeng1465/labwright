"""Tests for computation provenance — every derived number's full trace.

The contract: a provenance record carries formula, every input (name, value,
unit), the output unit, the value, the verifier's status and the Labwright code
version — nothing invented by a language model. The SOP section renders that
trace, and the ELN export serializes it losslessly.
"""

import json
import re

import pytest

from labwright.design import DesignInput, build_design
from labwright.schema.design import ChipGeometry, FlowParams
from labwright.sop.provenance import (
    CODE_VERSION,
    FORMULAS,
    export_eln,
    provenance_for,
    sop_provenance_section,
)
from labwright.verify.checker import verify_design


def _full_plan():
    return build_design(DesignInput(
        goal="Model drug-induced liver injury in a perfused liver chip",
        rationale="test plan",
        chip=ChipGeometry(width_um=400, height_um=100, length_mm=20, channel_count=1),
        flow=FlowParams(flow_rate_uLmin=10, viscosity_pas=1e-3, density_kgm3=1000),
        cells=dict(cell_type="HepG2", seeding_density_cells_cm2=1e5, culture_area_cm2=0.08),
        dosing=dict(compound="Acetaminophen", molecular_weight_g_mol=151.16,
                    stock_mM=100, working_mM=0.1, vehicle_control=True),
        stats=dict(effect_size=1.0, std_dev=1.0, alpha=0.05, power=0.80),
        culture=dict(
            plate_format="96", wells=4, cell_type="HepG2",
            seeding_density_cells_cm2=1e4, viability_pct=90,
            confluent_density_cells_cm2=1e6, doubling_time_h=30,
            culture_duration_h=72,
        ),
    ))


def test_code_version_is_versioned():
    assert re.match(r"^labwright-\d+\.\d+\.\d+", CODE_VERSION)


def test_every_derived_field_has_formula():
    plan = _full_plan()
    for field in (
        "derived.shear_pa", "derived.reynolds", "derived.pressure_drop_pa",
        "derived.residence_time_s", "derived.channel_volume_ul",
        "derived.mean_velocity_mms", "cells.seed_count",
        "culture.seed_per_well", "culture.total_seed_count",
        "culture.medium_volume_per_well_ml", "culture.total_medium_ml",
        "culture.expected_confluence_pct", "dosing.dmso_fraction_vv",
        "stats.n_per_group",
    ):
        assert FORMULAS.get(field), f"no formula for {field}"


def test_provenance_records_full_trace():
    plan = _full_plan()
    issues = verify_design(plan)
    records = provenance_for(plan, issues)
    fields = {r["field"] for r in records}
    for expected in ("derived.shear_pa", "culture.total_seed_count", "dosing.dmso_fraction_vv"):
        assert expected in fields
    shear = next(r for r in records if r["field"] == "derived.shear_pa")
    assert shear["formula"] == FORMULAS["derived.shear_pa"]
    assert shear["unit"] == "Pa"
    assert "flow_rate_uLmin" in {i["name"] for i in shear["inputs"]}
    for inp in shear["inputs"]:
        assert inp["value"] is not None
        assert inp["unit"]
    assert shear["status"] == "ok"
    assert shear["code_version"] == CODE_VERSION


def test_corrupt_field_status_is_error():
    plan = _full_plan()
    plan.derived.shear_pa *= 10  # a hallucinated shear stress
    records = provenance_for(plan, verify_design(plan))
    shear = next(r for r in records if r["field"] == "derived.shear_pa")
    assert shear["status"] == "error"


def test_provenance_has_no_llm_artifacts():
    records = provenance_for(_full_plan())
    blob = json.dumps(records).lower()
    for banned in ("gpt", "claude", "deepseek", "model-"):
        assert banned not in blob, f"provenance leaked {banned!r}"


def test_sop_section_renders_trace():
    plan = _full_plan()
    md = sop_provenance_section(plan, verify_design(plan))
    assert "## Computation provenance" in md
    assert CODE_VERSION in md
    assert "**derived.shear_pa**" in md
    assert "inputs:" in md


def test_eln_json_roundtrip():
    plan = _full_plan()
    data = json.loads(export_eln(plan, verify_design(plan), fmt="json"))
    assert isinstance(data, list) and data
    fields = {r["field"] for r in data}
    assert "derived.shear_pa" in fields
    for r in data:
        assert r["code_version"] == CODE_VERSION
        assert "formula" in r and "inputs" in r and "unit" in r and "status" in r


def test_eln_csv_has_header_and_rows():
    plan = _full_plan()
    csv_text = export_eln(plan, verify_design(plan), fmt="csv")
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("field,formula,inputs,unit,value,status,code_version")
    assert len(lines) > 1
