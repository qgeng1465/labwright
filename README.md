# 🧪 Labwright

**The AI bench copilot that gets your numbers right.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen)]()
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

**Run it in your browser, no setup:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qgeng1465/labwright/blob/main/colab/labwright_demo.ipynb)
— [colab/labwright_demo.ipynb](colab/labwright_demo.ipynb) installs Labwright,
designs a perfused liver-chip, and reverse-verifies a protocol's numbers.

Web demo (Hugging Face Space): [`hf_space/`](hf_space/) — see
[`hf_space/PUBLISH.md`](hf_space/PUBLISH.md) to deploy.

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
| **MMFT OoC Designer** (IEEE TCAD 2024) | Automated organ-chip *geometry* design, validated with CFD + fabrication | Optimizes geometry only — no LLM, no natural-language goals, and no cell biology, dosing or statistics. |

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
   [`eval/run_verify_batch.py`](eval/run_verify_batch.py) runs it over a set of
   published protocols + explicitly-labelled synthetic controls
   ([`eval/published_protocols/`](eval/published_protocols/)).
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
re-proves) — on two gold sets:

1. **24 "reading" goals** (`eval/gold_experiments.json`) — every goal states
   the answer (geometry, flow, or the physiological target number). This tests
   whether the pipeline extracts the stated numbers and drives the calculators
   to them. It deliberately does *not* test domain knowledge.
2. **12 "recall" goals** (`eval/gold_blind.json`) — the goal states no number
   ("recapitulate physiological venular wall shear"); the model must supply the
   canonical target itself. Nine are `cold` (answer nowhere); three are
   `prompt-backed` (the system prompt lists a range, but the model must still
   pick the right value).

Four systems are compared. The three LLM-memory systems (bare-LLM, soft-gate,
self-verify) write numbers from memory and are scored by *identical* rules —
only the prompt/stage structure differs. Labwright adds the calculators and
the verifier. On top of the two DeepSeek models, the open-weights 8B protocol
generator **thoth-8b** (the one runnable competitor from the related work) is
run through the same three memory systems as a prompt-only comparison — it has
no Labwright mode — and scores 0 % usable on every row (see the `thoth-8b`
rows below).

A *usable* design is internally consistent **and** hits every target within
±5 %. This is an *ablation*, not an equal-resource race: Labwright's
iteration budget, tools and anchor prompt are the treatment under test; the
one asymmetry favouring bare is a ±5 % tolerance and 3 retries. Full protocol,
fairness notes and per-entry records: [`eval/README.md`](eval/README.md).

```
$ python -m eval.report results/eval_flash.json

metric                          bare-LLM     Labwright
------------------------------------------------------
self-consistent rate                  0%           88%
usable rate                           0%           88%
hallucination rate                 1.000         0.125
```

| set | model | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| 24-reading | `flash` | bare-LLM | 0 % | 0 % | 1.000 |
| 24-reading | `flash` | soft-gate | 12 % | 12 % | 0.875 |
| 24-reading | `flash` | self-verify | 0 % | 0 % | 0.792 |
| 24-reading | `flash` | **Labwright** | **88 %** | **88 %** | **0.125** |
| 24-reading | `pro` | bare-LLM | 12 % | 12 % | 0.875 |
| 24-reading | `pro` | soft-gate | 8 % | 8 % | 0.917 |
| 24-reading | `pro` | self-verify | 0 % | 0 % | 0.750 |
| 24-reading | `pro` | **Labwright** | **100 %** | **100 %** | **0.000** |
| 12-blind | `flash` | bare-LLM | 8 % | 0 % | 0.917 |
| 12-blind | `flash` | soft-gate | 8 % | 0 % | 0.917 |
| 12-blind | `flash` | self-verify | 0 % | 0 % | 0.750 |
| 12-blind | `flash` | **Labwright** | **100 %** | **25 %** | **0.000** |
| 12-blind | `pro` | bare-LLM | 8 % | 0 % | 0.917 |
| 12-blind | `pro` | soft-gate | 0 % | 0 % | 1.000 |
| 12-blind | `pro` | self-verify | 0 % | 0 % | 0.806 |
| 12-blind | `pro` | **Labwright** | **100 %** | **33 %** | **0.000** |
| 24-reading | `thoth-8b` | bare-LLM | 0 % | 0 % | 1.000 |
| 24-reading | `thoth-8b` | soft-gate | 0 % | 0 % | 1.000 |
| 24-reading | `thoth-8b` | self-verify | 0 % | 0 % | 1.000 |
| 12-blind | `thoth-8b` | bare-LLM | 0 % | 0 % | 1.000 |
| 12-blind | `thoth-8b` | soft-gate | 0 % | 0 % | 1.000 |
| 12-blind | `thoth-8b` | self-verify | 0 % | 0 % | 1.000 |

*All memory-system rows come from a single re-run at temperature 0.2 after a
prompt regression that dropped the goal text was found and fixed (see the
transparency note in [`eval/README.md`](eval/README.md)); Labwright rows are
the committed run, preserved verbatim — Labwright's agent always receives the
goal, so the bug never touched it. The only usable memory-system entries on
either set are the three single-arithmetic-step goals on the 24-reading set
(12 % for `pro` bare / `flash` soft-gate): goals with no design choice. A point
or two between memory systems is sampling noise; the qualitative ordering is
not. The `thoth-8b` rows are the prompt-only comparison against the open-weights
8B protocol generator (why Thoth is the only runnable competitor, and why MMFT
and BPL-COGEN are not, is on the ground in [`eval/README.md`](eval/README.md#benchmarking-scope-why-these-systems-and-not-the-named-ones)): it has no
Labwright mode (its native output is protocol prose), and run through the
identical harness it produces nothing checkable — 0 % usable on every memory
system on both sets.*

Read the numbers honestly — and the boundary of what they mean.

- **"0.000 hallucination" is an architectural guarantee, not a measured win.**
  Labwright's derived numbers come from the calculators, and the verifier
  recomputes them from the *same* calculators, so a submitted design always
  verifies. What the number actually says: **no number entered a design unless
  a calculator produced it and the verifier re-proved it.** That is the whole
  claim — and it is a strong one. It does **not** say "every design is
  physiologically correct".
- **Recovery ≈ 0 on the 24-reading set is by construction**: the goals hand
  over the answers, and the self-consistent anchors are computed from the same
  equations. The real signal there is number-extraction and tool-calling — a
  genuine capability (bare reaches usable > 0 only on the three
  single-arithmetic-step goals, and only as 12 %; on every goal that requires
  choosing geometry and flow it is 0 % on both models).
- **The two naive fixes do not work.** `soft-gate` (a "re-check yourself"
  prompt) occasionally completes a single-arithmetic-step goal but never
  rescues a design — being told to be careful does not make an LLM's arithmetic
  checkable. `self-verify` (using a second LLM pass as its own verifier) is
  *worse* than nothing: handed its own raw inputs, the model recomputes them
  wrong, so the verifier pass overwrites correct numbers with confident wrong
  ones — 0 % self-consistent on both sets, both models. Only the deterministic
  calculators + verifier reach usable > 0 % on design goals.
- **The blind set is where target selection is actually tested — and Labwright
  drops.** `flash` 88 % → 25 %, `pro` 100 % → 33 %. The gate held: every plan
  was internally verified, hallucination 0.000. But the designs aimed at the
  wrong physiology. On the 12 goals `flash` recovers 3 (arterial 1.5 Pa, lung
  0.03 Pa, BBB 1.0 Pa), `pro` 4 (venular 0.3 Pa, lung 0.03 Pa, HepG2 seeding
  4000 cells per channel, BBB 1.0 Pa); both miss the liver (0.10 Pa vs the 0.05
  Pa convention) and the kidney (`flash` 0.50 Pa, `pro` 0.20 Pa — 24× / 9× off
  the 0.02 Pa target, the `pro` error reading dyn/cm² as Pa). **The gate stops
  fabricated numbers; it cannot supply domain knowledge the model does not
  have.** That boundary is the honest headline, and it is exactly what a
  wet-lab user must not forget: verify the target, not just the arithmetic.

The bare model's own numbers are worse than the earliest commits reported, for
two reasons, both reported honestly. First, the earliest figures (62 %/50 %
self-consistent) counted unverifiable answers — geometry and flow with no
derived numbers to check — as consistent; under the same rule Labwright uses
for a run that never submits (unverifiable = 1.0) the honest reading-set
figures drop to 0 %/12 % for `flash`/`pro`. Second, a prompt regression briefly
dropped the goal text from the bare-family prompts; it is caught by three
regression tests (`tests/test_benchmark_prompts.py`), and **all memory-system
numbers here are from a single post-fix re-run** while the Labwright numbers
are the committed run, preserved verbatim (Labwright's agent always received
the goal through a separate path). The recorded `reported` values are
unchanged; only the honest scoring rule and the prompt fix move the
headlines.

Labwright's residual error on `flash` (88 % usable, not 100 %) is *silence*,
not fabrication: the three goals it missed were pure-calculation goals
(Reynolds check, pressure-drop target, power analysis) where the agent
produced **no design at all** (`plan: false`; hallucination 1.0 is scored as
"no usable output"). The committed reading-set results mark these with
`plan: false`; the later blind-set runs additionally record the agent's own
failure reason, so the claim is auditable. It never wrote a number the
calculators didn't check.

## Roadmap

- [x] Calculators: microfluidics, cell, dosing, statistics
- [x] ReAct agent + deterministic verifier + SOP + CLI + Gradio demo
- [x] Bare-LLM vs Labwright benchmark on 24 gold entries (flash + pro); see *Benchmark*
- [x] Reverse-verification batch: `eval/run_verify_batch.py` over published protocols + labelled controls
- [x] Colab notebook + HF Space scaffolding (`colab/`, `hf_space/`)
- [x] Preprint drafted (kept local-only while in submission; the benchmark evidence and figures ship in this repo)
- [x] Pin all gold anchors (real DOIs / explicit self-consistent labels); render the paper figure (`paper/fig_benchmark.py`)
- [ ] Publish the HF Space (needs a HF token; see `hf_space/PUBLISH.md`)
- [ ] Domain package #2 (cell culture); fine-tune a small extractor on the V100
- [ ] Submit the preprint to bioRxiv

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
