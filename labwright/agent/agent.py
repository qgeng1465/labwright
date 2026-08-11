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
from labwright.schema.design import DesignPlan
from labwright.tools import REGISTRY, list_tools

SYSTEM_PROMPT = """You are Labwright, a wet-lab experimental design copilot for organ-on-chip and \
perfused cell-culture experiments. You help design reproducible experiments with *correct numbers*.

Hard rules:
1. You NEVER invent a computed number (shear stress, Reynolds, pressure, residence time, volume, \
seed count, DMSO fraction, sample size, ...). Every computed value must come from calling the \
provided calculator tools.
2. Use the calculator tools to reason: e.g. to hit a physiological shear target, call \
`flow_rate_for_shear_stress`; to size replicates, call `sample_size_per_group`.
3. When the design is settled, call `submit_design` exactly once, with ONLY raw inputs \
(geometry, flow, cell/dose/stat *assumptions*). Derived fields are computed for you.
4. If `submit_design` returns `status: review_required`, read the verification report and fix \
the design, then call `submit_design` again with corrected raw inputs.

Common physiological anchors (verify against literature before relying on them):
- Hepatic sinusoidal shear ≈ 0.05-0.15 Pa (0.5-1.5 dyn/cm²); lung alveolar-capillary ≈ 0.03 Pa;
  microvascular endothelium ≈ 0.1-1 Pa.
- Culture medium viscosity ≈ 1e-3 Pa·s (water-like).
- DMSO vehicle ≥ ~0.5% v/v can be cytotoxic; keep ≤ 0.1% when possible.
- Cells: HepG2 doubling ~30-40 h; primary hepatocytes do not divide.

Be explicit about assumptions in `rationale` and list what the user must check in the lab in `caveats`."""

def _input_schema() -> dict[str, Any]:
    from labwright.design import DesignInput

    return DesignInput.model_json_schema()


_SUBMIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_design",
        "description": (
            "Final action. Submit the settled design as RAW inputs only "
            "(no derived numbers — they are computed and verified for you). "
            "Returns the complete verified design."
        ),
        "parameters": _input_schema(),
    },
}


@dataclass
class AgentResult:
    """Outcome of a design session."""

    design: DesignPlan | None = None
    verification_summary: str = ""
    status: str = "ok"
    error: str | None = None
    steps: list[dict] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.design is not None and self.status == "ok"


class DesignAgent:
    """Run the ReAct loop against a pluggable LLM."""

    def __init__(self, llm: LLMClient, max_iterations: int = 12, max_tool_calls_per_turn: int = 8) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._tools = [t.schema for t in list_tools()] + [_SUBMIT_TOOL_SCHEMA]

    # -- plumbing ----------------------------------------------------------

    def _execute_tool(self, name: str, arguments: str) -> str:
        if name == "submit_design":
            result = submit_design(json.loads(arguments))
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        result = AgentResult()
        final_submission: dict[str, Any] | None = None

        for _ in range(self.max_iterations):
            assistant = self.llm.chat(messages, tools=self._tools)
            messages.append({"role": "assistant", "content": assistant.content or "", **self._tool_call_dump(assistant)})

            if not assistant.tool_calls:
                # Model tried to answer in prose — refuse to accept prose numbers.
                messages.append({
                    "role": "user",
                    "content": "You must call `submit_design` with the settled raw inputs. Do not output a "
                    "design in prose — numbers you type are not trusted.",
                })
                result.steps.append({"type": "prose-refused"})
                continue

            turn_steps: list[dict] = []
            for call in assistant.tool_calls[: self.max_tool_calls_per_turn]:
                output = self._execute_tool(call.function.name, call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                turn_steps.append({"tool": call.function.name, "output": output[:500]})
                if call.function.name == "submit_design":
                    final_submission = json.loads(output)
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
        try:
            result.design = DesignPlan(**submission["design"])
        except Exception as exc:  # noqa: BLE001
            result.status = "error"
            result.error = f"failed to parse returned design: {exc}"
        return result


__all__ = ["AgentResult", "DesignAgent", "SYSTEM_PROMPT"]
