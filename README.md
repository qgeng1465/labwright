# 🧪 Labwright

**The AI bench copilot that gets your numbers right.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-71%20passing-brightgreen)]()
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

**Sanity-check a published protocol** — the reverse of design. Given a paper's
geometry, flow and claimed shear/Re/n, Labwright recomputes every number and
flags any that don't follow from the paper's own inputs:

```bash
labwright verify-protocol examples/verify_protocol.json
# shear_pa   computed 0.05  claimed 0.5  rel.err 9.000  discrepancy
# → 1 claimed value(s) do not follow from the reported inputs.
```

**Models.** Default brain is `deepseek-v4-flash` (cheap, thinking disabled —
the arithmetic lives in the calculators, not the model). Any OpenAI-compatible
model works via `LABWRIGHT_MODEL`; both `deepseek-v4-flash` and
`deepseek-v4-pro` are benchmarked in `results/`.

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

## Related work & differentiation

We are not the first to put LLMs on wet-lab design — and we say so plainly.
Three directly related projects define the space:

| Project | What it does | The gap Labwright fills |
|---|---|---|
| **Thoth** (ICLR 2026) | 8B reasoning model that generates biological protocol *text*; trained with a SCORE structured-reward mechanism over 12k+ real protocols | Verification is a *learned* reward inside the model — soft and model-internal. It cannot **prove** a number: a protocol that says "shear 0.25 Pa" can score well and still not follow from its own geometry. |
| **BPL-COGEN** (bioRxiv 2026) | A formal protocol *language* plus a compiler; 95.1% fidelity on 300 Nature Protocols | The compiler checks protocol *structure* (type-safety), not physics. If a protocol asserts "shear 1 Pa", the compiler cannot tell you the stated geometry and flow imply 0.05. |
| **MMFT OoC Designer** (IEEE TCAD 2025) | Automated organ-chip *geometry* design, validated with CFD + fabrication | Optimizes geometry only — no LLM, no natural-language goals, and no cell biology, dosing or statistics. |

Labwright's claim is narrower and sharper: **no number enters a design unless a
deterministic calculator computed it and the verifier re-proved it.** The LLM
proposes raw inputs and a coherent biological narrative — the one thing it is
genuinely good at — while every computed value is exiled to unit-tested code.
That is a *hard gate*, not a soft reward:

> **Thoth learns to write plausible numbers. BPL checks they are well-typed.
> Labwright refuses numbers the physics doesn't support.**

Two capabilities the above don't have:

1. **Reverse verification of published protocols** — `labwright verify-protocol`
   takes a paper's reported geometry, flow and claimed shear / Reynolds / n,
   recomputes them from the paper's *own* inputs, and flags any number that does
   not follow. A literature sanity-checker, not just a design generator.
2. **A benchmark with a reproducibility yardstick** — `eval/` measures both
   parameter recovery *and* the fraction of derived numbers that fail the
   verifier (see below).

## Extending Labwright

Adding a calculator is the whole integration story:

```python
# 1. write the math in labwright/calc
# 2. declare it in labwright/tools.py
@register_tool(MyParams, "my_calculator", "what it does", my_calc, "my_domain")
```

The agent, verifier and demo all read the same registry — a new calculator is
instantly callable, verifiable and demonstrable. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Benchmark

Can an LLM write a wet-lab design without hallucinating the numbers? We measure
it. `eval/` runs two systems — **bare LLM** (the model writes every number from
memory) vs **Labwright** (the model proposes, calculators compute, the verifier
re-proves) — on 12 gold-standard organ-on-chip design goals with pinned
sources (kidney PTEC shear from Jang 2013, liver sinusoid CFD, etc.).

A *usable* design is internally consistent **and** hits every physiological
target within ±5 %. The bare model gets the easy road: retries, per-goal
prompts, and a ±5 % consistency tolerance, against Labwright's 1e-6 verifier.
Full protocol and per-entry records: [`eval/README.md`](eval/README.md),
`results/eval_flash.json`, `results/eval_pro.json`.

```
$ python -m eval.report results/eval_flash.json

metric                          bare-LLM     Labwright
------------------------------------------------------
self-consistent rate                 67%          100%
usable rate                           0%          100%
hallucination rate                 0.333         0.000
```

| model | system | self-consistent | usable | hallucination rate |
|---|---|---|---|---|
| `deepseek-v4-flash` | bare-LLM | 67 % | 0 % | 0.333 |
| `deepseek-v4-flash` | **Labwright** | **100 %** | **100 %** | **0.000** |
| `deepseek-v4-pro` | bare-LLM | 42 % | 8 % | 0.583 |
| `deepseek-v4-pro` | **Labwright** | **100 %** | **100 %** | **0.000** |

Read the numbers honestly. The bare model is not dumb — on several goals it
reports a *self-consistent* design that is still physiologically wrong: asked
for the canonical kidney 0.02 Pa, it confidently built a clean 0.1 Pa chip
(5× off), because the same 1000×100 µm @ 10 µL/min "default chip" satisfies
every shear target at once. Self-consistency does not save you when the target
was wrong to begin with. And on goals where no geometry is reported, its
numbers are simply untrustworthy (hallucination 1.0). Notably, the *larger*
reasoning model (`pro`) does **worse** bare (58 % vs 33 % hallucination):
bigger model, more confident arithmetic — none of it grounded.

Labwright's 100 % usable / 0.000 hallucination is **by construction**, not by
tuning: derived numbers are computed by unit-tested code and re-proved by the
verifier before the design is accepted. The benchmark exists to make that claim
checkable, and to give the reproducibility crisis a concrete, reproducible
number instead of a slogan.

## Roadmap

- [x] Calculators: microfluidics, cell, dosing, statistics (71 tests)
- [x] ReAct agent + deterministic verifier + SOP + CLI + Gradio demo
- [x] Bare-LLM vs Labwright benchmark on 12 gold entries (flash + pro) — 0.000 hallucination vs 33–58 %; see *Benchmark*
- [ ] Curate 20–30 gold organ-on-chip experiments (DOIs required); pin `needs_doi` anchors
- [ ] Reverse-verification of a batch of published protocols (the literature sanity-checker)
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
