"""Tests for the Gradio demo wiring — the trust surface a bench scientist sees.

These regression-tests the review finding that the "trace every number" panel
was dead on arrival: ``_run`` returned a 5-tuple while the click handler
declared 4 outputs, so the provenance markdown (output 5) was silently dropped
and the panel rendered the bare status string ``"ok"`` instead.
"""

from __future__ import annotations

import pytest

# ``labwright.ui`` imports gradio eagerly, and gradio lives in the ``[ui]``
# optional extra — so skip the whole module (not just fail) on installs
# without it. CI installs ``[dev,agent,ui]`` and runs these tests for real.
pytest.importorskip("gradio")

from labwright.agent.agent import AgentResult
from labwright.design import submit_design
from labwright.schema.design import DesignPlan
from labwright.ui import app
from labwright.verify.checker import Issue

_RAW = {
    "goal": "Perfused liver-chip model of drug-induced injury",
    "rationale": "Sinusoidal shear target 0.05 Pa; HepG2 at 1e5/cm2",
    "chip": {"width_um": 400, "height_um": 100, "length_mm": 20, "channel_count": 1},
    "flow": {"flow_rate_uLmin": 2, "viscosity_pas": 0.001, "density_kgm3": 1000},
    "cells": {"cell_type": "HepG2", "seeding_density_cells_cm2": 100000,
              "culture_area_cm2": 0.08},
    "dosing": {"compound": "Acetaminophen", "molecular_weight_g_mol": 151.16,
               "stock_mM": 100, "working_mM": 0.1, "vehicle_control": True},
    "stats": {"effect_size": 1.0, "std_dev": 1.0, "alpha": 0.05, "power": 0.80},
    "caveats": ["confirm shear from literature"],
}


def _plan_and_issues() -> tuple[DesignPlan, list[Issue]]:
    sub = submit_design(_RAW)
    return DesignPlan(**sub["design"]), [Issue(**i) for i in sub["verification"]]


def _patch_demo(monkeypatch, result: AgentResult):
    """Stub both the LLM client (no API key in tests) and the agent loop."""

    class FakeLLM:
        def __init__(self, *a, **k):  # noqa: ARG002 - scripted stand-in
            pass

    class FakeAgent:
        def __init__(self, llm):  # noqa: ARG002 - scripted stand-in
            pass

        def run(self, goal):  # noqa: ARG002 - scripted stand-in
            return result

    monkeypatch.setattr(app, "LLMClient", FakeLLM)
    monkeypatch.setattr(app, "DesignAgent", FakeAgent)


def test_run_returns_four_outputs_and_trace_lands_in_panel_4(monkeypatch):
    """The trace table must be output #4 — the fix for the silently-dropped panel."""
    plan, issues = _plan_and_issues()
    result = AgentResult(design=plan, status="ok", verification_summary="✓ all verified",
                         verification=issues)
    _patch_demo(monkeypatch, result)

    sop, js, badge, prov = app._run("goal", "", "", "")
    assert isinstance(sop, str) and isinstance(js, str)
    assert isinstance(badge, str) and isinstance(prov, str)
    # the provenance markdown lands in the 4th output (the trace panel)
    assert "derived.shear_pa" in prov
    assert "| field | formula | inputs | value | verify | code |" in prov
    # the badge is two-axis, not a binary "verified"
    assert "verified by calculators" in badge
    assert "model-proposed" in badge


def test_run_status_branch_is_two_axis(monkeypatch):
    plan, issues = _plan_and_issues()
    result = AgentResult(design=plan, status="review_required",
                         verification_summary="[WARNING] prose: ...",
                         verification=[Issue(level="warning", field="prose",
                                             message="unmatched number")])
    _patch_demo(monkeypatch, result)

    sop, js, badge, prov = app._run("goal", "", "", "")
    assert "review required" in badge
    assert "model-proposed" in badge
    # the SOP renders (warnings only) but the warning reaches the provenance table
    assert "prose" in prov or "warning" in sop.lower()
