"""Tests for the supplementary traceability-log builder."""

from __future__ import annotations

import json

from eval.make_traceability_log import build, iter_design_records, write_readme


def _prov_rec(field, status="ok"):
    return {
        "field": field, "formula": f"formula({field})",
        "inputs": [{"name": "x", "value": 1.0, "unit": "u"}],
        "unit": "u", "value": 1.0, "status": status,
        "code_version": "labwright-0.7 (git test)",
    }


def _entry(entry_id, plan=True, prov=True):
    lw = {
        "valid": bool(plan), "recovery": {"x": 0.0}, "hallucination_rate": 0.0,
        "no_plan": not plan, "tool_trace": ["calc_x", "submit_design"],
    }
    if plan:
        lw["plan"] = {"goal": entry_id, "derived": {"x": 1.0}}
    if prov:
        lw["provenance"] = [_prov_rec("derived.x"), _prov_rec("derived.y", "warning")]
    return {"id": entry_id, "gold": {"goal": f"goal for {entry_id}"},
            "labwright": lw}


def _results():
    return {
        "model": "test-model",
        "per_entry": [
            _entry("t1", plan=True, prov=True),
            _entry("t2", plan=False, prov=False),       # refused / no plan
            _entry("t3", plan=True, prov=False),        # legacy: plan without provenance
        ],
    }


def test_build_writes_per_entry_logs(tmp_path):
    summary = build(_results(), tmp_path)
    per = tmp_path / "test-model" / "t1__labwright.json"
    data = json.loads(per.read_text())
    assert data["entry_id"] == "t1"
    assert data["system"] == "labwright"
    assert data["goal"] == "goal for t1"
    assert data["plan"]["derived"]["x"] == 1.0
    assert len(data["provenance"]) == 2
    assert data["provenance"][0]["field"] == "derived.x"
    assert data["provenance"][1]["status"] == "warning"
    assert data["tool_trace"] == ["calc_x", "submit_design"]


def test_build_counts_coverage_honestly(tmp_path):
    summary = build(_results(), tmp_path)
    # Only t1 is fully traceable; t2 no plan, t3 plan-without-provenance are
    # both counted separately so coverage is honest.
    assert summary["entries_with_provenance"] == 1
    assert summary["entries_no_plan"] == 1
    assert summary["entries_plan_without_prov"] == 1
    assert summary["provenance_records"] == 2
    assert summary["derived_fields_covered"] == 2
    assert summary["status_counts"] == {"ok": 1, "warning": 1}
    assert summary["tool_usage"] == {"calc_x": 1, "submit_design": 1}


def test_index_and_readme_written(tmp_path):
    s = build(_results(), tmp_path)
    write_readme(tmp_path, [s])
    assert (tmp_path / "INDEX.json").exists()
    readme = (tmp_path / "README.md").read_text()
    assert "**1** entries" in readme
    assert "traceability" in readme.lower()


def test_iter_design_records_skips_non_design():
    results = {"per_entry": [
        {"id": "a", "gold": {"goal": "g"}, "bare": {"reported": {}}},
    ]}
    got = list(iter_design_records(results))
    assert got == []
