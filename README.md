# 🧪 Labwright

**The AI bench copilot that gets your numbers right.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-68%20passing-brightgreen)]()
![Status](https://img.shields.io/badge/status-alpha-yellow)

Tell a wet-lab LLM agent what experiment you want. It **proposes** the design.
Labwright **computes and verifies** every number — shear stress, flow rate,
seeding, DMSO carry-over, replicate counts — with deterministic calculators,
so nothing is hallucinated.

**Built for organ-on-chip and perfused cell culture first. General wet-lab by design.**

---

## Why this exists

More than half of published life-science results can't be reproduced
([reproducibility crisis, ~$28B/yr](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537370/)).
A big driver: experiment designs with wrong numbers — an unphysiological shear
stress, an underpowered replicate count, a cytotoxic DMSO concentration — that
survive peer review because nobody checks the arithmetic.

LLMs make this worse. Asked to design a perfusion experiment, a frontier model
will confidently write "shear stress 0.25 Pa" whether or not that follows from
the geometry it chose. **An LLM that writes numbers from memory is a
hallucination engine.**

Labwright inverts the responsibility:

> **The model proposes raw inputs. The calculators produce and verify every
> derived number. The model cannot write a number the calculators didn't check.**

## What you get

| | Without Labwright | With Labwright |
|---|---|---|
| "shear stress" | guessed from memory | `6·μQ/(w·h²)` — recomputed from your geometry |
| "n per group" | made up | power analysis from your effect size & σ |
| DMSO carry-over | "negligible" | `working/stock`, flagged if > 0.5% v/v |
| internal consistency | unverifiable | every derived field re-checked by the verifier |

## Demo

```
$ labwright design "liver-chip model of drug-induced injury at sinusoidal shear"
✓ all derived numbers verified against the calculators

# SOP: Model drug-induced liver injury in a perfused liver-chip at sinusoidal shear

## 2. Perfusion
- Flow rate: **2.00 µL/min** per channel
- Wall shear stress: **0.050 Pa** (0.50 dyn/cm²)
- Reynolds number: 0.13 (laminar, Re << 2300)
- Pressure drop: 20.0 Pa — verify the pump can hold this

## 3. Cell seeding
- Seeding density: 100000 cells/cm² over 0.080 cm²
- **Seed 8000 cells** per channel

## 4. Compound dosing
- Working dose: **0.1 mM** (Acetaminophen)
- DMSO carry-over: 0.10% v/v  ✅

## 5. Statistical design
- **16 biological replicates per group** (α=0.05, power=0.80, effect=1σ)
```

The model chose the goal, the geometry and the assumptions. Every bolded number
was computed by `labwright.calc` and passed `labwright.verify`.

## Quickstart

```bash
pip install -e .[agent]        # or: pip install labwright
export DEEPSEEK_API_KEY=sk-... # any OpenAI-compatible API works
labwright design "lung-on-chip at alveolar-capillary shear (~0.03 Pa)"
```

3 seconds to a verified design. `labwright tools` lists every calculator the
agent can call; `labwright design "..." --output sop` prints just the protocol.

## How it works

```
                    ┌──────────────────────────────┐
   "goal" ────────► │  LLM agent (proposes inputs) │
                    └──────────────┬───────────────┘
                                   │ raw inputs only
                    ┌──────────────▼───────────────┐
                    │  labwright.calc  (computes)  │  shear, Re, dP,
                    │  deterministic calculators   │  residence, seeding,
                    │  + submit_design             │  DMSO fraction, n
                    └──────────────┬───────────────┘
                                   │ verified plan
                    ┌──────────────▼───────────────┐
                    │  labwright.verify (checks)   │  every number re-derived
                    └──────────────┬───────────────┘
                                   ▼
                        SOP + design JSON
```

- **`calc/`** — pure, unit-tested engineering math (microfluidics, cell, dosing,
  stats). The moat: an LLM cannot compute these reliably, but a calculator can.
- **`agent/`** — a ReAct loop over the tool registry. It may call any
  calculator and must finish by calling `submit_design`. Prose answers are
  refused: *"numbers you type are not trusted."*
- **`verify/`** — re-runs every governing equation on the agent's own inputs and
  rejects designs that don't match. This is what makes the "no hallucinated
  numbers" claim checkable, not just asserted.
- **`schema/domains/`** — the extension point. `ooc/` is the first domain
  package; a new domain (cell culture, molecular biology, …) is a folder, not a
  fork.

## Why not just use X?

| Existing approach | What it can't do | What Labwright adds |
|---|---|---|
| **Bare ChatGPT / Claude** | writes numbers from memory; no arithmetic check | verifier recomputes every number; prose refused |
| **Thoth (ICLR'26) & protocol LLMs** | generate *text* protocols, numbers unverified | every derived number computed + checked |
| **MMFT OoC Designer / 3DuF** | static GUI; no natural language, no agent, no benchmark | LLM-driven, natural-language goals, verifiable, benchmarked |
| **Generic agent frameworks (LangGraph etc.)** | no domain calculators, no verification | tool registry + verifier + wet-lab schemas built in |

## Extending Labwright

Adding a calculator is the whole integration story:

```python
# 1. write the math in labwright/calc
# 2. declare it in labwright/tools.py
@register_tool(MyParams, "my_calculator", "what it does", my_calc, "my_domain")
```

The agent, verifier and demo all read the same registry — a new calculator is
instantly callable, verifiable and demonstrable. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Benchmark (reserved experiment window)

`eval/` contains a two-system benchmark — **bare LLM vs Labwright** — on a
gold-standard set of real organ-on-chip experiments, measuring
parameter-recovery accuracy and the **hallucination rate** (fraction of derived
numbers that fail the verifier). Provenance rules forbid unverifiable
literature numbers. Results will be reported here when the code has traction —
per the project's rule that experiments must have real significance, not run
for their own sake. See [`eval/README.md`](eval/README.md).

## Roadmap

- [x] Calculators: microfluidics, cell, dosing, statistics (68 tests)
- [x] ReAct agent + deterministic verifier + SOP + CLI + Gradio demo
- [ ] Curate 20–30 gold organ-on-chip experiments (DOIs required)
- [ ] Run bare-LLM vs Labwright benchmark; publish hallucination-rate numbers
- [ ] Domain package #2 (cell culture); fine-tune a small extractor on the V100
- [ ] Preprint (bioRxiv) with the benchmark + methodology

## License & citation

Apache-2.0. Built and maintained by [qgeng1465](https://github.com/qgeng1465).

```bibtex
@software{labwright,
  author = {Q., Geng},
  title = {Labwright: the AI bench copilot that gets your numbers right},
  year = {2026},
  url = {https://github.com/qgeng1465/labwright},
  license = {Apache-2.0}
}
```

**Disclaimer:** Labwright is an experimental-design aid, not medical device
software. Always verify generated protocols against your own lab's
standard operating procedures and safety regulations.
