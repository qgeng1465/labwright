"""Gradio demo: type a wet-lab goal, get a verified design + SOP.

The demo is intentionally thin: everything after the goal is the same
``DesignAgent`` + verifier pipeline anyone can run from the CLI or the API.
"""

from __future__ import annotations

import json

import gradio as gr

from labwright.agent import DesignAgent, LLMClient
from labwright.sop import design_to_sop

_EXAMPLE_GOALS = [
    "Design a perfused liver-chip experiment to model drug-induced liver injury, "
    "targeting sinusoidal shear and enough replicates to see a 1σ effect",
    "Set up a lung-on-chip culture at alveolar-capillary shear (~0.03 Pa) with "
    "HepG2 as a permeability barrier surrogate; include a vehicle control",
    "Plan a 3D tumor-spheroid-on-chip dosing study with 5-point serial dilution "
    "and powered comparison vs control",
]


def _run(goal: str, api_key: str, model: str, base_url: str) -> tuple[str, str, str, str]:
    if not goal.strip():
        return "Please describe an experimental goal.", "", "", "idle"
    try:
        llm = LLMClient(
            api_key=api_key.strip() or None,
            model=model.strip() or None,
            base_url=base_url.strip() or None,
        )
    except ValueError as exc:
        return str(exc), "", "", "error"

    result = DesignAgent(llm).run(goal)
    if result.status == "error":
        return f"**Error:** {result.error}", "", "", "error"

    sop = design_to_sop(result.design)
    js = json.dumps(result.design.model_dump(mode="json"), indent=2, ensure_ascii=False)
    badge = (
        "🟢 **verified** — every number computed by the calculators"
        if result.status == "ok"
        else "🟠 **review required** — see verification report"
    )
    return sop, js, badge + "\n\n" + result.verification_summary, result.status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Labwright — the AI bench copilot that gets your numbers right") as demo:
        gr.Markdown(
            "# 🧪 Labwright\n\n"
            "**The AI bench copilot that gets your numbers right.**\n\n"
            "Describe a wet-lab experiment and Labwright returns a **verified** design: "
            "chip geometry, flow & shear stress, seeding, dosing and replicates. "
            "The language model **proposes** the inputs; the **calculators** produce and "
            "verify every number, so nothing is hallucinated.\n\n"
            "You need an API key for the LLM brain (DeepSeek by default, OpenAI-compatible). "
            "Leave the fields empty to use `LABWRIGHT_API_KEY` / `DEEPSEEK_API_KEY` env vars."
        )
        with gr.Row():
            goal = gr.Textbox(
                label="Experimental goal",
                placeholder="e.g. liver-chip model of drug-induced injury at sinusoidal shear...",
                lines=3,
                scale=4,
            )
            with gr.Column(scale=1):
                run_btn = gr.Button("Design experiment", variant="primary")
        with gr.Accordion("Model configuration (optional)", open=False):
            api_key = gr.Textbox(label="API key", type="password", placeholder="sk-... (or use env var)")
            model = gr.Textbox(label="Model", placeholder="deepseek-chat")
            base_url = gr.Textbox(label="Base URL", placeholder="https://api.deepseek.com")
        with gr.Row():
            sop_out = gr.Markdown(label="SOP")
            design_out = gr.Code(label="Design (JSON)", language="json")
        status_out = gr.Markdown(label="Verification")

        run_btn.click(
            _run,
            inputs=[goal, api_key, model, base_url],
            outputs=[sop_out, design_out, status_out],
        )
        gr.Examples(examples=_EXAMPLE_GOALS, inputs=goal, label="Try one of these")
        gr.Markdown(
            "---\n*Labwright — every computed number is produced and verified by deterministic "
            "calculators. The model only proposes raw inputs. [Apache-2.0](https://github.com/qgeng1465/labwright)*"
        )
    return demo


def launch(**kwargs) -> None:
    build_app().launch(**kwargs)


__all__ = ["build_app", "launch"]
