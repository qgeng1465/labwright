"""The design agent — a ReAct tool loop that ends in a verified design.

Flow
----
1. The user states a wet-lab goal.
2. The agent may call calculator tools (:mod:`labwright.tools`) to reason over
   numbers (e.g. "which flow gives 0.05 Pa?").
3. The agent must finish by calling ``submit_design`` with **raw inputs only**.
4. Labwright derives every computed number, runs the verifier, and returns the
   verified design plus the verification report.

The agent therefore cannot emit a single computed number from memory; the only
path to a derived number is the calculators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from labwright.agent.llm import LLMClient
from labwright.design import submit_design
from labwright.physiology import physiology_anchor_text
from labwright.schema.design import DesignPlan
from labwright.tools import REGISTRY, list_tools

SYSTEM_PROMPT = """You are Labwright, a wet-lab experimental design copilot for organ-on-chip and \
perfused cell-culture experiments. You help design reproducible experiments with *correct numbers*.

Hard rules:
1. You NEVER invent a computed number (shear stress, Reynolds, pressure, residence time, volume, \
seed count, DMSO fraction, sample size, ...). Every computed value must come from calling the \
provided calculator tools.
2. If the goal is a PURE CALCULATION (e.g. "what shear does 10 µL/min give in a 400×100 µm \
channel?"), call the corresponding calculator directly and report its result — never write a \
number yourself, even "obvious" ones.
3. Before calling `submit_design`, lay out a short plan: which target values the goal needs, \
which calculator answers each one, and which raw inputs you will submit. Then work through the \
calculators step by step.
4. Use the calculator tools to reason: e.g. to hit a physiological shear target, call \
`flow_rate_for_shear_stress`; to size replicates, call `sample_size_per_group`.
5. When the design is settled, call `submit_design` exactly once, with ONLY raw inputs \
(geometry, flow, cell/dose/stat *assumptions*). Derived fields are computed for you.
6. If `submit_design` returns `status: review_required`, read the verification report and fix \
the design, then call `submit_design` again with corrected raw inputs. Fix ONLY the raw inputs \
you proposed; never hand-write a derived number to silence a check. Warnings (out-of-range, \
safety hints) are acceptable if you explain them in `caveats`; only errors must be corrected.
7. `submit_design` field types matter: `dosing.vehicle_control` is a JSON boolean (`true`/`false`); \
do NOT put prose there. Never include derived fields (`dmso_fraction_vv`, `seed_count`, \
`n_per_group`, any flow metric) in `submit_design` — they are computed for you.

Common physiological anchors (verify against literature before relying on them):
- Hepatic sinusoidal shear ≈ 0.05-0.15 Pa (0.5-1.5 dyn/cm²); lung alveolar-capillary ≈ 0.03 Pa;
  microvascular endothelium ≈ 0.1-1 Pa.
- Culture medium viscosity ≈ 1e-3 Pa·s (water-like).
- DMSO vehicle ≥ ~0.5% v/v can be cytotoxic; keep ≤ 0.1% when possible.
- Spheroids: one spheroid per well in a 96-well ULA plate (≈ 100 µL working volume/well);
  primary hepatocyte spheroids ≈ 1000 cells/spheroid ≈ 200 µm (20 µm cells, dense packing);
  spheroids above ~400 µm develop necrotic cores (oxygen diffuses ~200 µm from the surface).
- Perfused PK: drug extraction in a recirculating liver-on-chip is reported as the fraction
  cleared per pass (E = 1 − C_out/C_in). Hepatic extraction classes (Rowland & Tozer):
  low (E < 0.3, e.g. antipyrine — capacity-limited, reflects enzyme activity), intermediate
  (0.3–0.7), high (E > 0.7, e.g. propranolol — flow-limited, reflects perfusion). Clearance
  Cl = E·Q; the perfused volume to clear is the system volume (reservoir + chip + tubing).

Cell physiology (literature ranges with sources — call `cell_physiology` for the full per-cell entry):
""" + physiology_anchor_text() + """

Be explicit about assumptions in `rationale` and list what the user must check in the lab in `caveats`."""

#: Appended to the system prompt when the agent is configured with
#: ``elicit=True`` (the boundary/adversarial evaluation only). It switches the
#: agent's default for a missing input from "assume a standard value" to "ask".
#: The default shipped agent (``elicit=False``) is deliberately unchanged so the
#: 562 existing benchmark tests and the committed results are not perturbed.
_ELICIT_INSTRUCTION = """
Elicitation rule (boundary mode):
- If the goal is missing a parameter the calculators need, call `request_info` to ask the
  user for it — do NOT guess or assume a "standard" value. Report which parameter is missing
  and why it is needed (e.g. the channel height for wall shear).
- If the goal proposes a physically impossible or lethal condition (e.g. a shear above
  cell viability, an impossible geometry, a lethal DMSO load), do NOT compute or report a
  number for it. Submit the design as stated through `submit_design` so the verifier can
  reject it, or refuse if the goal is internally contradictory.
- Never fabricate a number to satisfy a goal that cannot be satisfied with the given
  information."""

def _request_info_schema() -> dict[str, Any]:
    """Schema for the elicitation tool (registered only when ``elicit=True``).

    ``request_info`` is a conversation tool: it forwards a question to the user
    and returns no computed value. Calling it (rather than guessing) is what the
    boundary evaluation counts as *elicitation* — the fail-safe response to a
    goal that is missing a calculator input.
    """
    return {
        "type": "function",
        "function": {
            "name": "request_info",
            "description": (
                "Ask the user for a parameter the goal did not state but a calculator "
                "needs. Use this instead of guessing a 'standard' value when the goal is "
                "under-determined. Returns 'question forwarded' — the user's answer will "
                "arrive as new information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "description": "Name of the missing parameter, e.g. 'height_um'.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The precise question to ask the user.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Why the parameter is needed and which calculator "
                        "consumes it.",
                    },
                },
                "required": ["parameter", "question"],
            },
        },
    }


def _input_schema() -> dict[str, Any]:
    from labwright.design import DesignInput

    return DesignInput.model_json_schema()


def _submit_tool_schema(verify_gate: bool = True) -> dict[str, Any]:
    if verify_gate:
        description = (
            "Final action. Submit the settled design as RAW inputs only "
            "(no derived numbers — they are computed and verified for you). "
            "Returns the complete verified design. On `review_required`, fix ONLY "
            "your raw inputs and resubmit — never hand-write a derived number."
        )
    else:
        description = (
            "Final action. Submit the settled design as design inputs. "
            "Returns the complete design with every computed number."
        )
    return {
        "type": "function",
        "function": {
            "name": "submit_design",
            "description": description,
            "parameters": _input_schema(),
        },
    }


#: The *no-gate ablation* system prompt. The calculator tools, the field-type
#: rules and the physiological anchors are unchanged; everything that depends on
#: the verifier is removed — the "never invent a computed number" rule, the
#: "derived fields are computed and verified for you" contract, and the
#: ``review_required`` fix loop. The benchmark runs this prompt (with the
#: verifier switched off in ``submit_design``) as the ``tool_no_gate`` system to
#: isolate what the verification layer adds beyond the calculators themselves.
NO_VERIFY_SYSTEM_PROMPT = """You are a wet-lab experimental design assistant for organ-on-chip and \
perfused cell-culture experiments. You help design reproducible experiments.

You have calculator tools that compute standard assay quantities (shear stress, Reynolds number, \
pressure, residence time, volumes, seed counts, DMSO fraction, sample size, ...). Use them whenever \
you need a number for the design.

When the design is settled, call `submit_design` with the design fields. `submit_design` field types \
matter: `dosing.vehicle_control` is a JSON boolean (`true`/`false`); do NOT put prose there.

Common physiological anchors (verify against literature before relying on them):
- Hepatic sinusoidal shear ≈ 0.05-0.15 Pa (0.5-1.5 dyn/cm²); lung alveolar-capillary ≈ 0.03 Pa;
  microvascular endothelium ≈ 0.1-1 Pa.
- Culture medium viscosity ≈ 1e-3 Pa·s (water-like).
- DMSO vehicle ≥ ~0.5% v/v can be cytotoxic; keep ≤ 0.1% when possible.
- Spheroids: one spheroid per well in a 96-well ULA plate (≈ 100 µL working volume/well);
  primary hepatocyte spheroids ≈ 1000 cells/spheroid ≈ 200 µm (20 µm cells, dense packing);
  spheroids above ~400 µm develop necrotic cores (oxygen diffuses ~200 µm from the surface).
- Perfused PK: drug extraction in a recirculating liver-on-chip is reported as the fraction
  cleared per pass (E = 1 − C_out/C_in). Hepatic extraction classes (Rowland & Tozer):
  low (E < 0.3, e.g. antipyrine — capacity-limited, reflects enzyme activity), intermediate
  (0.3–0.7), high (E > 0.7, e.g. propranolol — flow-limited, reflects perfusion). Clearance
  Cl = E·Q; the perfused volume to clear is the system volume (reservoir + chip + tubing).

Cell physiology (literature ranges with sources — call `cell_physiology` for the full per-cell entry):
""" + physiology_anchor_text() + """

Be explicit about assumptions in `rationale` and list what the user must check in the lab in `caveats`."""


@dataclass
class AgentResult:
    """Outcome of a design session."""

    design: DesignPlan | None = None
    #: The verifier's findings as ``Issue`` objects — carried so every consumer
    #: (UI, SOP, provenance) renders the *real* verdict, never a hardcoded "ok".
    verification: list[Any] = field(default_factory=list)
    verification_summary: str = ""
    status: str = "ok"
    error: str | None = None
    steps: list[dict] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.design is not None and self.status == "ok"


class DesignAgent:
    """Run the ReAct loop against a pluggable LLM.

    ``verify_gate=False`` selects the *no-gate ablation*: the same calculators
    and loop, but the verifier is switched off — ``submit_design`` always
    accepts, never reports ``review_required``, and the system prompt carries no
    verification discipline. The benchmark runs this as the ``tool_no_gate``
    system and scores the resulting plans post-hoc with the identical rules, so
    the only difference from ``labwright`` is the verification layer.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_iterations: int = 12,
        max_tool_calls_per_turn: int = 8,
        verify_gate: bool = True,
        max_submission_attempts: int = 1,
        elicit: bool = False,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.verify_gate = verify_gate
        self.max_submission_attempts = max_submission_attempts
        self.elicit = elicit
        self.system_prompt = SYSTEM_PROMPT if verify_gate else NO_VERIFY_SYSTEM_PROMPT
        if elicit:
            self.system_prompt += _ELICIT_INSTRUCTION
        self._tools = [t.schema for t in list_tools()] + [_submit_tool_schema(verify_gate)]
        if elicit:
            self._tools.append(_request_info_schema())

    # -- plumbing ----------------------------------------------------------

    def _execute_tool(self, name: str, arguments: str) -> str:
        if name == "request_info":
            # Elicitation is a conversation turn, not a computation: the
            # question is forwarded to the user and the agent waits. The tool
            # call is already recorded in ``steps`` (the boundary evaluation
            # counts it as the fail-safe elicitation response). It deliberately
            # returns no numeric value — the whole point is that the missing
            # input is not yet available.
            try:
                q = json.loads(arguments)
            except Exception:  # noqa: BLE001 - feed malformed args back to the model
                return json.dumps({"status": "invalid", "error": "request_info expects a JSON object"})
            return json.dumps({
                "status": "question_forwarded",
                "parameter": q.get("parameter"),
                "message": "Ask the user: " + str(q.get("question", "")),
            }, ensure_ascii=False)
        if name == "submit_design":
            try:
                result = submit_design(json.loads(arguments), verify=self.verify_gate)
            except Exception as exc:  # noqa: BLE001 - feed the schema error back to the model
                result = {"status": "validation_error", "error": str(exc)}
            return json.dumps(result, ensure_ascii=False, default=str)
        if name not in REGISTRY:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps({"result": REGISTRY[name].call(**json.loads(arguments))}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - report to the model so it can self-correct
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # -- main loop ---------------------------------------------------------

    def run(self, goal: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": goal},
        ]
        result = AgentResult()
        final_submission: dict[str, Any] | None = None
        submission_attempts = 0

        for _ in range(self.max_iterations):
            assistant = self.llm.chat(messages, tools=self._tools)
            messages.append({"role": "assistant", "content": assistant.content or "", **self._tool_call_dump(assistant)})

            if not assistant.tool_calls:
                # Model tried to answer in prose. With the gate on, refuse to
                # accept prose numbers (they are unverifiable); in the no-gate
                # ablation, still steer it back to the tool interface so it
                # stays in the calculator loop rather than drifting to prose.
                refusal = (
                    "You must call `submit_design` with the settled raw inputs. Do not output a "
                    "design in prose — numbers you type are not trusted."
                    if self.verify_gate
                    else "You must call `submit_design` with the design inputs."
                )
                messages.append({"role": "user", "content": refusal})
                result.steps.append({"type": "prose-refused"})
                continue

            turn_steps: list[dict] = []
            for call in assistant.tool_calls[: self.max_tool_calls_per_turn]:
                output = self._execute_tool(call.function.name, call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                turn_steps.append({"tool": call.function.name, "output": output[:500]})
                if call.function.name == "submit_design":
                    sub = json.loads(output)
                    status = sub.get("status")
                    if status == "ok":
                        final_submission = sub
                    elif status == "review_required":
                        submission_attempts += 1
                        if submission_attempts < self.max_submission_attempts:
                            # The verifier found real problems and the report is
                            # already on the wire as this tool result; the agent
                            # reads it and fixes the flagged fields (the prompt
                            # promises this loop — honour it when configured).
                            result.steps.append(
                                {"type": "review_required", "attempt": submission_attempts}
                            )
                            messages.append({"role": "user", "content": (
                                "The verifier returned `review_required` "
                                f"(attempt {submission_attempts}/{self.max_submission_attempts}). "
                                "Read the verification report above, fix ONLY the flagged "
                                "fields, and call `submit_design` again with the corrected "
                                "raw inputs."
                            )})
                        else:
                            # Attempt budget spent: accept the honest verdict so
                            # the final plan still records the real (dirty) state.
                            final_submission = sub
                    else:
                        # Schema/validation failure: the model sees the error as a
                        # tool result and can correct the submission next turn.
                        result.steps.append({"type": "submit_rejected", "output": output[:300]})
                    break
            result.steps.extend(turn_steps)

            if final_submission is not None:
                return self._finalize(final_submission, result)

        result.status = "error"
        result.error = f"agent did not call submit_design within {self.max_iterations} iterations"
        return result

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _tool_call_dump(assistant) -> dict[str, Any]:
        """Serialize assistant tool_calls into the wire format for the next turn."""
        if not assistant.tool_calls:
            return {}
        return {
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in assistant.tool_calls
            ]
        }

    def _finalize(self, submission: dict[str, Any], result: AgentResult) -> AgentResult:
        result.status = submission.get("status", "ok")
        result.verification_summary = submission.get("verification_summary", "")
        # Rebuild the verifier's Issue objects (submit_design returns dicts) so
        # the UI and the SOP provenance block can render per-field verdicts.
        from labwright.verify.checker import Issue

        result.verification = [
            Issue(**v) if isinstance(v, dict) else v
            for v in submission.get("verification", [])
        ]
        try:
            result.design = DesignPlan(**submission["design"])
        except Exception as exc:  # noqa: BLE001
            result.status = "error"
            result.error = f"failed to parse returned design: {exc}"
        return result


__all__ = ["AgentResult", "DesignAgent", "SYSTEM_PROMPT", "NO_VERIFY_SYSTEM_PROMPT"]
