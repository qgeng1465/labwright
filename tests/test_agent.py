"""Tests for the design agent ReAct loop, using a scripted fake LLM."""

from types import SimpleNamespace

import pytest

from labwright.agent.agent import DesignAgent
from labwright.verify.checker import Issue


class FakeLLM:
    """Scripted stand-in for an OpenAI-compatible model."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        # First call: the model asks the calculator for a flow rate to hit 0.05 Pa.
        if self.calls == 1:
            return self._msg(
                tool_calls=[
                    self._call(
                        "flow_rate_for_shear_stress",
                        '{"target_shear_pa": 0.05, "width_um": 400, "height_um": 100, "viscosity_pas": 0.001}',
                    )
                ]
            )
        # Second call: submit the design with raw inputs.
        if self.calls == 2:
            return self._msg(
                tool_calls=[
                    self._call(
                        "submit_design",
                        _RAW_INPUT_JSON,
                    )
                ]
            )
        raise AssertionError(f"unexpected extra call #{self.calls}")

    @staticmethod
    def _call(name, arguments):
        return SimpleNamespace(id=f"call-{name}", function=SimpleNamespace(name=name, arguments=arguments))

    @staticmethod
    def _msg(tool_calls):
        return SimpleNamespace(content=None, tool_calls=tool_calls)


_RAW_INPUT_JSON = (
    '{"goal":"Perfused liver-chip model of drug-induced injury",'
    '"rationale":"Sinusoidal shear target 0.05 Pa; HepG2 at 1e5/cm2",'
    '"chip":{"width_um":400,"height_um":100,"length_mm":20,"channel_count":1,"material":"PDMS"},'
    '"flow":{"flow_rate_uLmin":2,"viscosity_pas":0.001,"density_kgm3":1000},'
    '"cells":{"cell_type":"HepG2","seeding_density_cells_cm2":100000,"culture_area_cm2":0.08,'
    '"doubling_time_h":35,"culture_duration_h":72},'
    '"dosing":{"compound":"Acetaminophen","molecular_weight_g_mol":151.16,"stock_mM":100,'
    '"working_mM":0.1,"vehicle_control":true,"exposure_h":24},'
    '"stats":{"effect_size":1.0,"std_dev":1.0,"alpha":0.05,"power":0.80},'
    '"caveats":["confirm shear from literature"]}'
)


def test_agent_calls_tool_then_submits():
    fake = FakeLLM()
    agent = DesignAgent(llm=fake, max_iterations=5)
    result = agent.run("Design a liver chip at 0.05 Pa shear")

    assert fake.calls == 2
    assert result.is_verified
    assert result.design is not None
    assert result.status == "ok"
    # Derived numbers came from the calculators, not the model
    assert round(result.design.derived.shear_pa, 4) == pytest.approx(0.05, abs=1e-3)
    assert result.design.stats.n_per_group == 16
    assert result.design.cells.seed_count == pytest.approx(8000)
    assert result.design.dosing.dmso_fraction_vv == pytest.approx(0.001)
    # Tool execution is recorded
    assert any(s["tool"] == "flow_rate_for_shear_stress" for s in result.steps)
    assert any(s["tool"] == "submit_design" for s in result.steps)
    # The verifier's real findings travel with the result as Issue objects, so
    # the UI/SOP can render per-field verdicts instead of a hardcoded "ok".
    assert result.verification == []
    assert isinstance(result.verification, list)


def test_agent_result_carries_issue_objects():
    """A design submitted with a verification warning returns Issue objects, not dicts."""
    bad = _RAW_INPUT_JSON.replace(
        '"rationale":"Sinusoidal shear target 0.05 Pa; HepG2 at 1e5/cm2"',
        '"rationale":"Sinusoidal shear will be 0.5 Pa"',  # contradicts the derived ~0.05
    )

    class WarnLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id=f"c{self.calls}",
                    function=SimpleNamespace(name="submit_design", arguments=bad),
                )],
            )

    result = DesignAgent(llm=WarnLLM(), max_iterations=3).run("design")
    assert result.status == "review_required"
    assert result.verification, "a warning-producing design must carry findings"
    assert all(isinstance(i, Issue) for i in result.verification)
    assert any(i.field == "prose" and i.level == "warning" for i in result.verification)


def test_agent_refuses_prose_answer():
    class ProseLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content="The shear stress is 0.25 Pa, done!", tool_calls=None)
            if self.calls == 2:
                return SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="submit_design", arguments=_RAW_INPUT_JSON),
                        )
                    ],
                )
            raise AssertionError("unexpected call")

    agent = DesignAgent(llm=ProseLLM(), max_iterations=5)
    result = agent.run("design")
    assert result.is_verified
    assert any(s["type"] == "prose-refused" for s in result.steps)
    # The prose number was never accepted anywhere
    assert "0.25" not in result.verification_summary


def test_agent_self_corrects_on_validation_error():
    """A malformed submit_design must be fed back, not crash the loop."""

    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            bad = '{"goal":"g","rationale":"r","chip":{"width_um":400,"height_um":100,"length_mm":20},' \
                  '"flow":{"flow_rate_uLmin":2,"viscosity_pas":0.001},"cells":{"cell_type":"X",' \
                  '"seeding_density_cells_cm2":100000,"culture_area_cm2":0.08},' \
                  '"dosing":{"compound":"A","molecular_weight_g_mol":151,"stock_mM":100,"working_mM":0.1,' \
                  '"vehicle_control":"0.1% v/v DMSO in medium"}}'  # vehicle_control is a string -> invalid
            if self.calls == 1:
                return SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(id="bad", function=SimpleNamespace(name="submit_design", arguments=bad))],
                )
            return SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="good",
                        function=SimpleNamespace(name="submit_design", arguments=_RAW_INPUT_JSON),
                    )
                ],
            )

    agent = DesignAgent(llm=FlakyLLM(), max_iterations=5)
    result = agent.run("design")
    assert result.is_verified
    assert any(s.get("type") == "submit_rejected" for s in result.steps)
    assert result.design is not None


def test_agent_gives_up_without_submit():
    class StubbornLLM:
        def chat(self, messages, tools=None, **kwargs):
            return SimpleNamespace(content=None, tool_calls=None)

    agent = DesignAgent(llm=StubbornLLM(), max_iterations=3)
    result = agent.run("design")
    assert result.status == "error"
    assert "did not call submit_design" in (result.error or "")
