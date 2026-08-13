"""Adversarial gate-breaking — the attack tests that turn the benchmark's
``hallucination_rate == 0.000`` from a definitional artifact into a verified
safety property.

A number may enter a Labwright design by exactly one path: a **raw input**, then
derived and re-proved by the calculators. These tests attack every other path and
prove it is closed:

1. A derived number typed in prose (the agent's final answer) is refused.
2. A derived field smuggled into ``submit_design`` — top-level, as a ``derived``
   block, or nested inside a raw block — is rejected with a validation error,
   never silently dropped.
3. A derived field tampered with in a finished plan is caught by the verifier.
4. A derived number *asserted* in the design's own prose (``rationale`` /
   ``caveats``) that contradicts the calculators is flagged as a warning.

Together they make "no hallucinated number can ship" a tested claim, not an
architectural hope.
"""

import pytest

from labwright.agent.agent import DesignAgent
from labwright.design import submit_design
from labwright.schema.design import DesignPlan
from labwright.verify.checker import has_errors, verify_design

# A valid raw-only design, shared across tests. Derived shear from these inputs
# is ≈ 0.05 Pa (see test_agent.py).
_RAW = {
    "goal": "Perfused liver-chip model of drug-induced injury",
    "rationale": "Sinusoidal shear target 0.05 Pa; HepG2 at 1e5/cm2",
    "chip": {"width_um": 400, "height_um": 100, "length_mm": 20,
             "channel_count": 1, "material": "PDMS"},
    "flow": {"flow_rate_uLmin": 2, "viscosity_pas": 0.001, "density_kgm3": 1000},
    "cells": {"cell_type": "HepG2", "seeding_density_cells_cm2": 100000,
              "culture_area_cm2": 0.08, "doubling_time_h": 35,
              "culture_duration_h": 72},
    "dosing": {"compound": "Acetaminophen", "molecular_weight_g_mol": 151.16,
               "stock_mM": 100, "working_mM": 0.1, "vehicle_control": True,
               "exposure_h": 24},
    "stats": {"effect_size": 1.0, "std_dev": 1.0, "alpha": 0.05, "power": 0.80},
    "caveats": ["confirm shear from literature"],
}


def _prose_warnings(result: dict) -> list[str]:
    return [i["message"] for i in result["verification"] if i["field"] == "prose"]


# ---------------------------------------------------------------------------
# 1. A derived number typed in prose is refused
# ---------------------------------------------------------------------------


def test_prose_only_answer_is_refused():
    """The agent's prose answer never becomes a design; a tool call is required."""

    class ProseLLM:
        def chat(self, messages, tools=None, **kwargs):
            return type("Msg", (), {"content": "The shear is 0.25 Pa, done!",
                                    "tool_calls": None})()

    result = DesignAgent(llm=ProseLLM(), max_iterations=2).run("design")
    assert result.status == "error"
    assert result.design is None
    assert any(s.get("type") == "prose-refused" for s in result.steps)


# ---------------------------------------------------------------------------
# 2. A derived field smuggled into submit_design is rejected explicitly
# ---------------------------------------------------------------------------


def test_submit_rejects_top_level_derived_field():
    with pytest.raises(ValueError, match="shear_pa"):
        submit_design({**_RAW, "shear_pa": 0.05})


def test_submit_rejects_derived_block():
    with pytest.raises(ValueError, match="derived"):
        submit_design({**_RAW, "derived": {"shear_pa": 0.05, "reynolds": 1.0}})


def test_submit_rejects_derived_field_in_culture_block():
    payload = {**_RAW, "cells": None,
               "culture": {"plate_format": "96-well", "wells": 1,
                           "cell_type": "HepG2",
                           "seeding_density_cells_cm2": 100000,
                           "seed_per_well": 1234}}
    with pytest.raises(ValueError, match="culture.seed_per_well"):
        submit_design(payload)


def test_submit_rejects_derived_field_in_spheroid_block():
    payload = {**_RAW, "cells": None,
               "spheroid": {"cell_type": "HepG2", "spheroid_format": "96-ula",
                            "spheroid_count": 1, "cells_per_spheroid": 1000,
                            "cell_diameter_um": 20, "expected_diameter_um": 999}}
    with pytest.raises(ValueError, match="spheroid.expected_diameter_um"):
        submit_design(payload)


def test_agent_recovers_when_derived_field_rejected():
    """A smuggle attempt is fed back as a tool error; the agent resubmits clean."""
    import json
    from types import SimpleNamespace

    def call(name, arguments):
        return SimpleNamespace(id=name,
                               function=SimpleNamespace(name=name,
                                                        arguments=arguments))

    class SmugglerLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                bad = dict(_RAW)
                bad["shear_pa"] = 0.25  # try to force the answer
                return SimpleNamespace(
                    content=None,
                    tool_calls=[call("submit_design", json.dumps(bad))],
                )
            return SimpleNamespace(
                content=None,
                tool_calls=[call("submit_design", json.dumps(_RAW))],
            )

    result = DesignAgent(llm=SmugglerLLM(), max_iterations=5).run("design")
    assert result.is_verified
    assert any(s.get("type") == "submit_rejected" for s in result.steps)
    assert result.design is not None
    # the smuggled 0.25 never reached the design
    assert abs(result.design.derived.shear_pa - 0.25) > 1e-6


# ---------------------------------------------------------------------------
# 3. A derived field tampered with in a finished plan is caught by the verifier
# ---------------------------------------------------------------------------


def test_tampered_derived_field_caught_by_verifier():
    result = submit_design(_RAW)
    plan = DesignPlan(**result["design"])
    plan.derived.shear_pa *= 10.0  # a hand-edited or injected value
    issues = verify_design(plan)
    assert has_errors(issues)
    assert any(i.field == "derived.shear_pa" and i.level == "error" for i in issues)


def test_tampered_spheroid_field_caught_by_verifier():
    payload = {**_RAW, "cells": None,
               "spheroid": {"cell_type": "HepG2", "spheroid_format": "96-ula",
                            "spheroid_count": 4, "cells_per_spheroid": 1000,
                            "cell_diameter_um": 20}}
    plan = DesignPlan(**submit_design(payload)["design"])
    plan.spheroid.cells_total = 99999
    issues = verify_design(plan)
    assert any(i.field == "spheroid.cells_total" and i.level == "error" for i in issues)


def test_hallucination_rate_zero_only_for_clean_design():
    from eval.benchmark import hallucination_rate

    clean = DesignPlan(**submit_design(_RAW)["design"])
    assert hallucination_rate(clean) == 0.0

    tampered = DesignPlan(**submit_design(_RAW)["design"])
    tampered.derived.shear_pa *= 10.0
    assert hallucination_rate(tampered) > 0.0


# ---------------------------------------------------------------------------
# 4. A number asserted in the design's own prose is cross-checked
# ---------------------------------------------------------------------------


def test_prose_contradiction_is_flagged():
    bad = dict(_RAW)
    bad["rationale"] = "Sinusoidal shear will be 0.5 Pa"  # derived is ~0.05
    result = submit_design(bad)
    assert _prose_warnings(result), "contradicting prose number must be flagged"
    assert not any(i["level"] == "error" for i in result["verification"]), \
        "a warning, not an error"


def test_prose_restating_derived_value_is_fine():
    result = submit_design(_RAW)  # rationale says 0.05 Pa; derived is ~0.05
    assert not _prose_warnings(result)


def test_prose_unit_alias_is_normalised():
    # 0.5 dyn/cm² == 0.05 Pa — a unit alias, not a contradiction.
    ok = dict(_RAW)
    ok["rationale"] = "Sinusoidal shear 0.5 dyn/cm²"
    result = submit_design(ok)
    assert not _prose_warnings(result)


def test_prose_number_skipped_when_dimension_absent():
    # A plate-culture design carries no pressure fields: "0.05 Pa" in prose
    # cannot be judged against a pressure value that does not exist.
    payload = {**_RAW, "cells": None,
               "culture": {"plate_format": "96-well", "wells": 4,
                           "cell_type": "HepG2",
                           "seeding_density_cells_cm2": 100000},
               "rationale": "Some note about 0.05 Pa shear we cannot check"}
    result = submit_design(payload)
    assert not _prose_warnings(result)
    assert result["status"] == "ok"


def test_prose_number_matching_raw_input_is_fine():
    # A raw-input number restated in prose (the flow the design actually uses).
    ok = dict(_RAW)
    ok["rationale"] = "Perfuse at 2 µL/min for a liver sinusoid"
    result = submit_design(ok)
    assert not _prose_warnings(result)


def test_prose_scanner_rejects_no_legitimate_design():
    # Sanity: a realistic, fully-populated rationale produces no false warnings.
    realistic = dict(_RAW)
    realistic["rationale"] = (
        "HepG2 seeded at 1e5 cells/cm2 on 0.08 cm2; perfuse at 2 µL/min "
        "(shear ≈ 0.05 Pa, laminar Re < 1); 100 mM stock diluted to 0.1 mM "
        "(0.1% DMSO); 16 per group for power 0.8; harvest at 72 h."
    )
    result = submit_design(realistic)
    assert result["status"] == "ok"
    assert not _prose_warnings(result)


def test_spheroid_prose_anchors_are_fine():
    # The system-prompt spheroid anchors restate raw/derived values the design
    # carries, so a well-built spheroid design must not be flagged.
    payload = {**_RAW, "cells": None,
               "spheroid": {"cell_type": "primary hepatocytes",
                            "spheroid_format": "96-ula", "spheroid_count": 24,
                            "cells_per_spheroid": 1000, "cell_diameter_um": 20},
               "rationale": "One spheroid per 96-ULA well (100 µL medium); "
                            "1000 cells/spheroid ≈ 200 µm, above 400 µm necrotic."}
    result = submit_design(payload)
    assert not _prose_warnings(result), result["verification_summary"]
