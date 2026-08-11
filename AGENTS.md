# Labwright — development guide

## Project identity

Labwright is a **verifiable wet-lab experimental-design copilot**: an LLM agent
proposes raw inputs; deterministic calculators in `labwright/calc/` compute
every derived number; `labwright/verify/` re-checks them. The project's thesis
(and its paper) is that this fixes LLM hallucination of wet-lab numbers.

- **Author only**: qgeng1465. No attribution to any external assistant anywhere in the
  repo, code, or CI.
- **Real significance rule**: no experiments/training run "for their own
  sake". The `eval/` benchmark is a *reserved window* — run only after the code
  and demo have traction, and only with DOI-verifiable gold data.
- Target: real use + a good paper (scGPT-style: usable tool + benchmark +
  preprint). Stars come from people being able to use it.

## Commands

- venv: `/data/qiushuogeng/projects/labwright/.venv/bin/python` (absolute path;
  shell cwd resets between calls)
- tests: `.venv/bin/python -m pytest tests/`
- run: `.venv/bin/python -m labwright.cli design "..."` (needs API key)
- pip index: TUNA mirror (`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`)
- GitHub: no `gh` CLI — use curl + token at `/home/qiushuogeng/token.txt`
- HF downloads: `HF_ENDPOINT=https://hf-mirror.com`

## Architecture

```
labwright/
  calc/         deterministic calculators (microfluidics, cell, dosing, stats)
  tools.py      tool registry — the single extension point (register_tool)
  schema/       pydantic contracts; design.py splits RAW inputs vs DERIVED
  design.py     build_design(): derives everything; submit_design(): + verify
  verify/       checker.py: recomputes every derived number, rejects mismatch
  agent/        llm.py (OpenAI-compatible, DeepSeek default) + agent.py (ReAct)
  sop/          design_to_sop(): deterministic markdown protocol
  ui/           gradio demo; cli.py commands: design / tools
eval/           benchmark harness + gold experiments (reserved window)
```

## Golden rule

**The model proposes; calculators compute; verifier proves.** Never add a path
that lets the LLM emit a derived number. Add calculators via `register_tool`,
never by hardcoding LLM output.
