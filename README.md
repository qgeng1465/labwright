# 🧪 Labwright

**The AI bench copilot that gets your numbers right.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![CI](https://github.com/qgeng1465/labwright/actions/workflows/tests.yml/badge.svg)](https://github.com/qgeng1465/labwright/actions)
[![Tests](https://img.shields.io/badge/tests-280%20passing-brightgreen)]()
![Status](https://img.shields.io/badge/status-alpha-yellow)

Frontier LLMs hallucinate ~100 % of the derived numbers in a wet-lab design.
Labwright doesn't let them: the model proposes **raw inputs**, deterministic
calculators compute every derived number, and a verifier **re-proves each one**
before the design is accepted. One yardstick, one rule for every row
([`eval/`](eval/README.md)):

| system | how derived numbers are produced | usable designs (24 reading goals) | hallucination |
|---|---|---|---|
| bare frontier LLM (the status quo) | written from memory | **0–12 %** | ~0.9–1.0 |
| "check yourself" / LLM-as-verifier | self-derived (soft-gate, self-verify) | **0 %** on design goals — only a few single-step arithmetic goals ever reach 8–12 %; the second pass actively corrupts the first | ~0.75–1.0 |
| **Labwright** | calculators compute; verifier re-proves | **88–100 %** | **0.000** |

*Snapshot — the **24-reading set only**. The 15-goal blind set and the 15-goal
3D-spheroid set are in the full table below.*

*hallucination = the fraction of a plan's `derived` fields the verifier
rejects (error-level), averaged over goals; a run that submits no design
scores 1.0 (the denominator is derived fields per plan, not goals — see
Definitions below). Labwright's hallucination is **0.000 on most sets**; the
few non-zero cells are silence or a single rejected field, never a fabricated
number: `flash` **0.125** on the 24-reading set (three pure-calculation goals
where the agent produced no design — silence, not wrong numbers), `flash`
**0.011** / `pro` **0.067** on the 15-3D-spheroid set (one goal each: a single
rejected spheroid field / a no-submit silence), and `flash` **0.071** / `pro`
**0.043** on the 14-plate-culture set (one silence / two goals with a single
rejected field). Per-goal detail is in the benchmark section below.*

![Labwright graphical abstract: goal → LLM proposes raw inputs → deterministic calculators → verifier re-proves every number → SOP + design JSON; naive alternatives rejected at the hard gate](paper/fig_abstract.png)

**Built for organ-on-chip and perfused cell culture first. General wet-lab by design.**

👉 **Try it:** `pip install -e .[agent]` (PyPI release pending) ·
[Open in Colab](https://colab.research.google.com/github/qgeng1465/labwright/blob/main/colab/labwright_demo.ipynb) ·
[Web demo](hf_space/) · reverse-verify a *published* protocol:
`labwright verify-protocol examples/verify_protocol.json`

**Who this is for.**
- *Bench scientists* — paste a paper's geometry/flow/shear and get a
  discrepancy check in 3 seconds (`labwright verify-protocol`), or describe an
  experiment and get a verified SOP.
- *AI-for-science researchers* — a hard-gate agent architecture with a
  reproducible benchmark and an honestly stated boundary ([`eval/`](eval/README.md)).
- *Contributors* — adding a domain is a folder, not a fork ([CONTRIBUTING.md](CONTRIBUTING.md)).

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

## The gap today's wet-lab LLMs haven't closed

An LLM can write you a beautiful protocol. But **every number in it** — shear
stress, flow rate, seeding density, DMSO carry-over, replicate count — is a
*derived* quantity: it only exists once you choose a geometry, a flow, a cell
density. Models write these from memory, and memory can't do arithmetic.

We looked at every closely-related system we could run. None of them closes
this gap — Labwright is the one that does:

| System | How it handles protocol numbers | Can it *prove* a number follows from its own inputs? |
|---|---|---|
| **Thoth** (ICLR 2026) | 8B model trained on 12k+ real protocols with a structured reward to write *plausible* protocol text | No — verification is a learned, model-internal reward |
| **BPL-COGEN** (bioRxiv 2026) | compiler gives 95.1% *type* fidelity on 300 Nature Protocols | No — checks structure, not physics |
| **ChemCrow** (Nature Mach. Intell.) | LLM agent for chemistry; verification delegated to the LLM as judge | No — the judge can't be trusted for arithmetic |
| **LLM self-check** ("check yourself") | the model re-derives its own numbers | No — we measured it: the second pass actively corrupts the first |
| **MMFT OoC Designer** (IEEE TCAD 2024) | deterministic organ-chip *geometry* synthesis | No LLM, no natural language, no cell/dosing/stats layer |
| **Labwright** (this repo) | LLM proposes **raw inputs**; deterministic calculators compute every derived number; the verifier **re-derives each one** | **Yes — a hard gate.** No number enters a design unless a calculator produced it and the verifier re-proved it |

**Labwright inverts the responsibility: the model cannot type a number the
calculators didn't check — a hard gate, not a soft reward.** And the *same*
calculators run backwards: paste a published paper's geometry, flow and claimed shear, and
Labwright recomputes the claims and flags anything that doesn't follow from the
paper's own inputs — a reproducibility checker in three seconds
([`labwright verify-protocol`](#quickstart)).

**Measured on one yardstick.** None of the systems above publishes a measure of
whether its output numbers follow from its own inputs — we do (the table at the
top; full protocol in [`eval/`](eval/README.md)). The two that could
conceivably be run are not runnable here (BPL's released pipeline needs ~60 GB
of GPU memory; MMFT is a deterministic geometry synthesizer, not an LLM), so we
state that plainly instead of claiming a head-to-head. One honest boundary,
stated in [the benchmark](eval/README.md): verification is *necessary, not
sufficient*. Labwright proves numbers are internally consistent; it cannot
supply physiology the model doesn't know. On the 15-goal blind set the usable
rate collapses from 88–100 % on the reading set to **40–47 %** (`flash` 6/15,
`pro` 7/15). Restricting to the eight genuinely **cold** goals — the answer is
in neither the goal nor the system prompt — each model recovers only **3
(38 %**, 95 % Wilson CI 14–69 %); the other five are prompt-backed, the answer
sitting inside a range in the prompt. That boundary is the real research
frontier, and closing it is where this project is headed.

## What you get

| | Without Labwright | With Labwright |
|---|---|---|
| "shear stress" | guessed from memory | `6·μQ/(w·h²)` — recomputed from your geometry |
| "n per group" | made up | power analysis from your effect size & σ |
| DMSO carry-over | "negligible" | `working/stock`, flagged if > 0.5% v/v |
| internal consistency | unverifiable | every derived field re-checked by the verifier |
| unit of that shear | whoever read the paper | dyn/cm²-as-Pa misreads detected and flagged as unit errors (0.2 dyn/cm² ≠ 0.2 Pa) |
| shear that can't exist | passes | outside the physiological band → warning; outside physical limits → error |
| a cytotoxic dose | passes | rejected with a reason against the institution's safety boundary |
| "where did this number come from?" | "trust me" | formula + every input (name, value, unit) + code version, in the SOP and the design JSON |

## Verification is layered, and safety is configurable

Arithmetic is only the first layer of "the number is right". Labwright checks
four of them, in order, and never passes a violation silently:

1. **Arithmetic** — the verifier re-runs every governing equation
   ([`labwright/verify/checker.py`](labwright/verify/checker.py)).
2. **Units & dimensions** — every field has a canonical unit
   ([`labwright/verify/units.py`](labwright/verify/units.py)); the alias table
   catches the misreads that actually bite (dyn/cm² vs Pa = 10×, mL/min vs µL/min,
   ...). The benchmark's **unit-misread rate** counts these as unit errors, not
   generic arithmetic errors.
3. **Physiological range** — each quantity sits in a sanity band
   ([`labwright/verify/sanity.py`](labwright/verify/sanity.py)): wall shear
   0.001–10 Pa (hard 1e-4–50), seeding density 10³–10⁷ cells/cm², DMSO
   <0.5% v/v (hard <14%). Soft-band violations warn; hard-band violations error.
4. **Safety & compliance** ([`labwright/verify/safety.py`](labwright/verify/safety.py)) —
   hazardous-compound dose caps (e.g. doxorubicin >0.5 mM rejected with a
   reason), a mandatory matched vehicle control, BSL hints for BSL-2 cell
   material, animal-ethics reminders — and every threshold lives in a
   `SafetyConfig` boundary a lab sets per institution (JSON, or in code):
   ```python
   from labwright.verify.safety import SafetyConfig, set_safety_config
   set_safety_config(SafetyConfig(max_dmso_vv=0.01, institution="C-301"))
   ```

**Computation provenance** ([`labwright/sop/provenance.py`](labwright/sop/provenance.py))
makes "computed by calc, verified by verify" something a reviewer can re-derive
line by line: every bolded SOP number carries its formula, every input (name,
value, unit), the output unit, the Labwright code version, and the verifier's
verdict — appended to the SOP, embedded in the design JSON, and exportable to an
ELN/LIMS (`export_eln(plan, issues, fmt="json"|"csv")`). The web demo shows it
in a clickable traceability panel.

**The agent is constrained to the honest path**
([`labwright/agent/agent.py`](labwright/agent/agent.py)): if the goal is pure
calculation it must call the calculator directly instead of writing the number;
it must decompose the goal into a plan before acting; and when a verification
fails it may **only fix the raw inputs it proposed — never hand-write a derived
number** to silence a check. Every tool's description carries a worked example
and its common mistakes.

**The gate is attack-tested** — the `hallucination_rate == 0.000` cells are
not "0 because we said so"; every alternative path a hallucinated number could
take into a design is closed and proven closed by an adversarial test suite
([`tests/test_gate_security.py`](tests/test_gate_security.py)). The gate's
claim is that a derived number enters a design only when a calculator produced
it and the verifier re-proved it — a rejected plan on the spheroid set (flash
0.011) and a no-submit silence (`pro` 0.067) are the gate *working*, not
failing:

| attack | defense | test |
|---|---|---|
| answer the goal in prose ("shear is 0.25 Pa") | prose answers refused; only `submit_design` is accepted | `test_prose_only_answer_is_refused` |
| smuggle `shear_pa` / `derived{…}` / `culture.seed_per_well` into `submit_design` | `submit_design` rejects every derived field name with a validation error — never silently drops it | `test_submit_rejects_*`, `test_agent_recovers_when_derived_field_rejected` |
| hand-edit / inject a derived field into a finished plan | the verifier re-runs the calculators and flags the mismatch | `test_tampered_*` |
| assert a number in the design's own prose (`rationale`, `caveats`) that contradicts the calculators | a prose-number gate ([`labwright/verify/prose.py`](labwright/verify/prose.py)) extracts every number-with-unit, converts it to the field's unit (so "0.5 dyn/cm²" is judged as 0.05 Pa), and warns when it matches no value the design actually carries | `test_prose_*` |

`*` marks a family of tests in `tests/` — representative names only
(`test_submit_rejects_*` covers `test_submit_rejects_derived_block`,
`test_submit_rejects_top_level_derived_field`, `test_submit_rejects_derived_field_in_culture_block`,
`test_submit_rejects_derived_field_in_spheroid_block`; `test_tampered_*` covers
`test_tampered_derived_field_caught_by_verifier`,
`test_tampered_spheroid_field_caught_by_verifier`; `test_prose_*` covers the
prose gate's positive/negative cases).

Prose assertions are warnings, never errors, so an honest design is never
blocked; threshold phrases ("above 400 µm", "up to 24 h") are domain knowledge,
not design claims, and are not judged.

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
pip install -e .[agent]        # PyPI release pending (name reserved)
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

![Labwright pipeline: goal → LLM proposes raw inputs → calculators compute → verifier re-proves → SOP + design JSON](paper/fig_pipeline.png)

The goal goes in; a design whose every number was computed by
`labwright.calc` and re-proved by `labwright.verify` comes out. The agent
writes the narrative; the arithmetic is exiled to unit-tested code.

- **`calc/`** — pure, unit-tested engineering math (microfluidics, cell, plate
  cell culture, dosing, stats). The moat: an LLM cannot compute these reliably,
  but a calculator can. A second domain — well-format geometry, seeding density,
  hemocytometer counts, viability, passaging — ships as `calc/culture.py` with
  its own gold set (`eval/gold_cell_culture.json`). A third domain — 3D culture:
  spheroid/organoid geometry, cells-per-size, ULA and hanging-drop working
  volumes, necrotic-core limits — ships as `calc/spheroid.py` with its own gold
  set (`eval/gold_spheroid.json`). A fourth domain — single-compartment
  pharmacokinetics on-chip (extraction ratio, clearance, half-life,
  steady-state accumulation) — ships as `calc/pk.py` with its own gold set
  (`eval/gold_pk.json`).
- **`agent/`** — a ReAct loop over the tool registry. It may call any
  calculator and must finish by calling `submit_design`. Prose answers are
  refused: *"numbers you type are not trusted."*
- **`verify/`** — re-runs every governing equation on the agent's own inputs and
  rejects designs that don't match. This is what makes the "no hallucinated
  numbers" claim checkable, not just asserted.
- **`extract/`** — a fine-tuned goal→raw-inputs model (Qwen2.5-1.5B LoRA,
  `extract/pipeline.py`): the natural-language goal seeds the raw inputs the
  calculators then check, so a design can be generated without an agent
  round-trip. Eval (`extract/eval.py`): JSON parse **1.0**, extract→verify
  consistency **0.9976**, field recovery **0.72** on 400 rows + 12 blind goals
  (the pre-expansion blind set) — against **0.40** consistency for the untuned
  `deepseek-v4-flash`/`pro` baselines on the same rows (results are in
  `results/extractor/eval_report.json`).
- **`schema/` + `published.py`** — the verified design plan types
  (`DesignPlan`, `CulturePlan`, …); `published.py` runs the *same* calculators
  backwards over a published protocol's own inputs. A new domain is a
  `calc/` module + a `tools.py` registration, not a fork.

## Related work & differentiation

We are not the first to put LLMs on wet-lab design — and we say so plainly.
[The comparison table above](#the-gap-todays-wet-lab-llms-havent-closed) puts
Labwright next to every closely-related system we could run (Thoth, BPL-COGEN,
ChemCrow, LLM self-check, MMFT). Three of them define the space; Labwright's
claim is narrower and sharper: **no number enters a design unless a
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
   ([`eval/published_protocols/`](eval/published_protocols/)). Scaled to the
   literature, `eval/run_scirecipe_audit.py` ran the same check over **21,094**
   real SciRecipe protocol summaries (14,589 numeric → 5,700 audited). **Read the
   denominators exactly**: of the **5,700 audited protocols, only 104** carried a
   derived number that could be re-derived from the protocol's own inputs — a
   checkable rate of **104/5,700 = 1.8 %**. Among those 104, **30** were
   internally consistent and **74** were contradicted by the papers' own numbers —
   a checkable consistency of **30/104 = 28.8 %**. The other 5,596 rows stated
   no derived number that could be re-derived; they are `unverifiable`, never
   counted as "ok". **This
   28.8 % is the consistency rate among the checkable rows only, not "28.8 % of
   the literature is inconsistent"** — it says: of the 1.8 % of protocols that
   say enough to check, 28.8 % agree with their own inputs. An early run
   inflated the figure to 0.898 by counting no-derived-number rows as "ok";
   those are now `unverifiable` (a regression test pins the fix). The funnel is
   the reproducibility-gap measurement behind the audit figure in
   `paper/fig_scirecipe.py`.
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
instantly callable, verifiable and demonstrable.

Adding a whole *design domain* (a new optional part of a design, like the
3D-spheroid plan) is one declaration too: a `calc/` module, a schema model, a
derive function, and a `Block` in `labwright/blocks.py` — that one entry owns
the domain's raw/derived/consistency keys, its field map, sanity bands and
canonical units, and the design gate, verifier, units layer and benchmark all
import from it. A domain that forgets a band or unit fails loudly at import.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Benchmark

Can an LLM write a wet-lab design without hallucinating the numbers? We measure
it. `eval/` runs five systems — **bare LLM** (the model writes every number from
memory), two naive fixes (**soft-gate**, **self-verify**), **Labwright** (the
model proposes, calculators compute, the verifier re-proves), and a
**fine-tuned raw-input extractor** (a fixed local Qwen2.5-1.5B LoRA) — on five
gold sets:

1. **24 "reading" goals** (`eval/gold_experiments.json`) — every goal states
   the answer (geometry, flow, or the physiological target number). This tests
   whether the pipeline extracts the stated numbers and drives the calculators
   to them. It deliberately does *not* test domain knowledge.
2. **15 "recall" goals** (`eval/gold_blind.json`) — the goal states no number
   ("recapitulate physiological venular wall shear"); the model must supply the
   canonical target itself. Eight are `cold` (answer nowhere: kidney PTEC,
   arterial, HepG2 seeding, PHH seeding, pulmonary artery, gut, retinal
   arteriole, 24-well medium volume); five are `prompt-backed` (liver, venular,
   lung, BBB, lymphatic — the answer sits inside a range in the system prompt —
   the model must still pick the right value, and range-membership is the
   criterion, so venular and lymphatic count as prompted too); two are
   **scenario-only** (the magnitude is stated, so they test a failure mode, not
   cold recall):
   - **unit-ambiguity** — the goal gives the target in dyn/cm² and asks for Pa;
     a dyn-as-Pa misread is exactly 10× off.
   - **multi-target** — two targets (shear 1.0 Pa *and* residence ≥6 s) that are
     jointly satisfiable at Q ≈ 40 µL/min in a 400×100 µm × 100 mm channel; the
     model must hit both at once.
   Every entry pins a citable source; no number is invented.
3. **15 "3D-spheroid" goals** (`eval/gold_spheroid.json`) — a third domain:
   3D culture (spheroid/organoid). Four are reading (solid-sphere geometry,
   cells-per-size), three exercise a standard vessel working volume
   (96-ULA 100 µL, 384-ULA 50 µL, hanging drop), four are failure-mode
   scenarios (**unit-ambiguity** mm-vs-µm, **growth** projection, **multi-target**
   total-cell + total-medium, **cross-domain** spheroid + DMSO dosing), and four
   are blind (two `prompt-backed`: 1000 cells/spheroid and 96-ULA volume live in
   the system-prompt anchors; two `cold`: 384-ULA and hanging-drop volumes are
   nowhere in the prompt). 3D culture is a deliberately knowledge-weak area for
   LLMs — spheroid conventions are fragmented across ULA/hanging-drop vendors —
   so this set stresses recall and cross-domain reasoning rather than a single
   geometry. Every entry pins a citable source or a self-consistent derivation;
   no number is invented.
4. **14 plate-culture goals** (`eval/gold_cell_culture.json`) — a fourth domain:
   2D/plate culture (wells, seeding density, counting, viability, confluence).
   Ten are reading (plate geometry / density stated) and four are blind-`cold`
   (the model must recall the pinned PHH sandwich-plating density or a
   plate-table working volume). This set exists to prove the domain transfer is
   not an artifact of the microfluidics calculators: the plate-culture
   calculators are a separate module, and the blind `cold` cells require
   recall, not derivation.
5. **14 perfused-system PK goals** (`eval/gold_pk.json`) — a fifth domain:
   single-compartment pharmacokinetics on-chip (extraction ratio, clearance,
   half-life, steady-state accumulation, mass cleared). Twelve are
   reading/scenario (every input stated, or the formula's raw numbers given,
   including two **unit traps** — mM-vs-µM and minutes-vs-hours), and two are
   blind `prompt-backed` (propranolol high-extraction / antipyrine
   low-extraction — the classification is stated in the system prompt, the
   target number is not). PK equations are pinned to Rowland & Tozer and
   Gibaldi & Perrier; one literature citation is to Baudoin et al.
   (doi:10.1002/jps.23796).

Five systems are compared, on two frontier models (plus a model-independent
fixed local extractor). The three LLM-memory systems (bare-LLM, soft-gate,
self-verify) write numbers from memory and are scored by *identical* rules —
only the prompt/stage structure differs. Labwright adds the calculators and
the verifier. The fifth, **finetuned-ext**, is a local Qwen2.5-1.5B-Instruct
LoRA fine-tuned on synthetic flow/culture instances whose raw-input targets
are reused from the reading gold set — so the reading and culture columns are
*in-distribution* for it (a plug-in replacement for the API models' extraction
step), while the spheroid and PK columns are out-of-distribution. Its bars are
identical under flash and pro by construction.

**New failure-mode metrics.** Each entry is also classified *why* it failed
(`ok` / `silence` / `calculation_error` / `wrong_target`), whether a
wrong number was a probable **unit misread** (dyn/cm²-vs-Pa etc., via the unit
layer), whether the headline target was **selected** within ±5 %, and the
blind-set cells are split by hint strength (cold vs prompt-backed). The `eval.report`
renderer prints all of it; the classification and misread logic are unit-tested
(`tests/test_metrics.py`).

![Benchmark: self-consistent rate, usable rate and hallucination rate on the 24-reading, 15-blind, 15-3D-spheroid, 14-culture and 14-PK sets (flash & pro; finetuned-ext identical under both). The memory systems (stone / ochre / sage) reach a usable design only on the handful of single-step goals the goal hands over; Labwright (deep blue) holds the gate, misses the blind-set physiology, and stays near the reading-set ceiling on the spheroid, culture and PK sets; the in-distribution fine-tuned extractor (lilac) closes much of the reading-set gap but collapses out-of-distribution.](paper/fig_benchmark.png)

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

\*The non-zero Labwright hallucination cell (0.125) is **silence, not
fabrication**: the three reading-set goals it missed were pure-calculation
goals (Reynolds check, pressure-drop target, power analysis) where the agent
produced **no design** (`plan: false`); a no-submit run scores 1.0 by
convention, so 3/24 → 0.125. It never wrote a number the calculators didn't
check.*

*Definitions: **self-consistent** = every submitted number was re-derived from
its own raw inputs (zero verifier errors); **usable** = self-consistent
**and** every physiological target within ±5 %; **hallucination** = the
fraction of a plan's `derived` fields the verifier rejects at error level,
averaged over goals (a run that submits no design scores 1.0). The
hallucination denominator is derived fields per plan, not goals — a single
goal with one rejected field moves the set-level number by 1/(goals × fields),
so a goal that scores 0.000 does not pull the set to 0.000, and a raw-input
absurdity that the verifier flags (e.g. an unphysical doubling time) makes a
plan invalid without moving hallucination at all — the two signals must be
read together. A plan can be fully self-consistent and still miss the target —
that is exactly what the 15-blind rows show (100 % self-consistent, 40–47 %
usable).*

| set | model | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| 24-reading | `flash` | bare-LLM | 0 % | 0 % | 1.000 |
| 24-reading | `flash` | soft-gate | 12 % | 12 % | 0.875 |
| 24-reading | `flash` | self-verify | 0 % | 0 % | 0.792 |
| 24-reading | `flash` | **Labwright** | **88 %** | **88 %** | **0.125** |
| 24-reading | `flash` | finetuned-ext (in-dist) | 92 % | 79 % | 0.083 |
| 24-reading | `pro` | bare-LLM | 12 % | 12 % | 0.875 |
| 24-reading | `pro` | soft-gate | 8 % | 8 % | 0.917 |
| 24-reading | `pro` | self-verify | 0 % | 0 % | 0.750 |
| 24-reading | `pro` | **Labwright** | **100 %** | **100 %** | **0.000** |
| 24-reading | `pro` | finetuned-ext (in-dist) | 92 % | 79 % | 0.083 |
| 15-blind | `flash` | bare-LLM | 7 % | 0 % | 0.933 |
| 15-blind | `flash` | soft-gate | 13 % | 0 % | 0.867 |
| 15-blind | `flash` | self-verify | 0 % | 0 % | 0.611 |
| 15-blind | `flash` | **Labwright** | **100 %** | **40 %** | **0.000** |
| 15-blind | `flash` | finetuned-ext | 80 % | 7 % | 0.200 |
| 15-blind | `pro` | bare-LLM | 7 % | 0 % | 0.933 |
| 15-blind | `pro` | soft-gate | 13 % | 0 % | 0.867 |
| 15-blind | `pro` | self-verify | 0 % | 0 % | 0.733 |
| 15-blind | `pro` | **Labwright** | **100 %** | **47 %** | **0.000** |
| 15-blind | `pro` | finetuned-ext | 80 % | 7 % | 0.200 |
| 15-3D-spheroid | `flash` | bare-LLM | 20 % | 20 % | 0.800 |
| 15-3D-spheroid | `flash` | soft-gate | 13 % | 13 % | 0.867 |
| 15-3D-spheroid | `flash` | self-verify | 20 % | 20 % | 0.569 |
| 15-3D-spheroid | `flash` | **Labwright** | **93 %** | **87 %** | **0.011** |
| 15-3D-spheroid | `flash` | finetuned-ext (OOD) | 33 % | 0 % | 0.611 |
| 15-3D-spheroid | `pro` | bare-LLM | 27 % | 27 % | 0.733 |
| 15-3D-spheroid | `pro` | soft-gate | 27 % | 27 % | 0.733 |
| 15-3D-spheroid | `pro` | self-verify | 40 % | 20 % | 0.400 |
| 15-3D-spheroid | `pro` | **Labwright** | **93 %** | **87 %** | **0.067** |
| 15-3D-spheroid | `pro` | finetuned-ext (OOD) | 33 % | 0 % | 0.611 |
| 14-plate-culture | `flash` | bare-LLM | 0 % | 0 % | 0.893 |
| 14-plate-culture | `flash` | soft-gate | 0 % | 0 % | 0.893 |
| 14-plate-culture | `flash` | self-verify | 0 % | 0 % | 0.929 |
| 14-plate-culture | `flash` | **Labwright** | **93 %** | **86 %** | **0.071** |
| 14-plate-culture | `flash` | finetuned-ext (in-dist) | 100 % | 64 % | 0.000 |
| 14-plate-culture | `pro` | bare-LLM | 7 % | 7 % | 0.750 |
| 14-plate-culture | `pro` | soft-gate | 7 % | 7 % | 0.786 |
| 14-plate-culture | `pro` | self-verify | 0 % | 0 % | 0.821 |
| 14-plate-culture | `pro` | **Labwright** | **86 %** | **64 %** | **0.043** |
| 14-plate-culture | `pro` | finetuned-ext (in-dist) | 100 % | 64 % | 0.000 |
| 14-perfused-PK | `flash` | bare-LLM | 50 % | 36 % | 0.500 |
| 14-perfused-PK | `flash` | soft-gate | 50 % | 50 % | 0.500 |
| 14-perfused-PK | `flash` | self-verify | 79 % | 29 % | 0.214 |
| 14-perfused-PK | `flash` | **Labwright** | **100 %** | **79 %** | **0.000** |
| 14-perfused-PK | `flash` | finetuned-ext (OOD) | 29 % | 0 % | 0.714 |
| 14-perfused-PK | `pro` | bare-LLM | 43 % | 36 % | 0.536 |
| 14-perfused-PK | `pro` | soft-gate | 50 % | 36 % | 0.500 |
| 14-perfused-PK | `pro` | self-verify | 79 % | 29 % | 0.214 |
| 14-perfused-PK | `pro` | **Labwright** | **100 %** | **86 %** | **0.000** |
| 14-perfused-PK | `pro` | finetuned-ext (OOD) | 29 % | 0 % | 0.714 |

*All memory-system rows come from a single re-run at temperature 0.2 after a
prompt regression that dropped the goal text was found and fixed (see the
transparency note in [`eval/README.md`](eval/README.md)); Labwright rows are
the committed run, preserved verbatim — Labwright's agent always receives the
goal, so the bug never touched it. The 15-3D-spheroid memory-system rows were
additionally rerun under a fairness fix to the scorer: string vessel formats
(`spheroid_format` / `plate_format`) were previously never extracted from
memory-system output, so every spheroid convention goal scored 1.0
unverifiable regardless of the answer; the fix recovers them, recomputes each
derived number from the raws it needs, and excludes
reported-but-not-recomputable numbers. The formerly committed 0 % / 1.000
spheroid cells were this artifact. After the fix the only usable memory-system
entries are the three single-arithmetic-step goals on the 24-reading set (12 %
for `pro` bare / `flash` soft-gate) plus a handful of single-step spheroid
geometry/lookup goals (bare 20 % / 27 %, `flash`/`pro`). A point or two between
memory systems is sampling noise; the qualitative ordering (Labwright ≫ memory
systems) is not. Why the published systems in related work are not benchmarked
here is on the ground in
[`eval/README.md`](eval/README.md#benchmarking-scope-why-these-systems-and-not-the-named-ones).*

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
  drops.** `flash` 88 % → 40 %, `pro` 100 % → 47 %. The gate held: every plan
  was internally verified, hallucination 0.000. But the designs aimed at the
  wrong physiology. On the 15 goals:
  - `flash` recovers **6/15**: arterial 1.5 Pa, HepG2 seeding, 24-well medium
    volume, lung 0.03 Pa, and both scenario goals — the dyn/cm²-as-Pa unit
    test and the shear + residence joint target.
  - `pro` recovers **7/15**: arterial, HepG2 seeding, 24-well medium, venular
    0.3 Pa, lung 0.03 Pa, BBB 1.0 Pa, and the unit-ambiguity goal. (`pro`'s
    multi-target run hits the shear but misses the residence time 0.5×, so it
    is **not** counted as usable.)
  Both usable rates are single-run point estimates with wide error bars: the
  95 % Wilson CI around 6/15 = 40 % is **20–64 %**, around 7/15 = 47 % it is
  **25–70 %** — n=15 is too thin to separate the two models, or either from
  the cold-only 38 % below.
  **Cold-only honesty check:** five of the 15 goals are `prompt-backed` (the
  answer sits inside the system prompt's physiological-anchor range — liver,
  lung, BBB, venular, lymphatic), so on the eight genuinely cold goals `flash`
  and `pro` each recover only **3** (arterial, HepG2, 24-well medium):
  cold-only usable ≈ **38 % / 38 %**, each with a 95 % Wilson CI of **14–69 %**
  — n=8 is still too thin to separate the models, and cold recall is nowhere
  near the reading set. Of the recoveries that look like domain knowledge,
  only those three are actually cold; the others (lung, BBB, venular) sit
  inside the prompted range. Remove the two scenario-only goals (they state
  the magnitude, so they test a failure mode, not recall) and the
  *domain*-target recovery is **4/13 = 31 %** for `flash` and **6/13 = 46 %**
  for `pro` — scenario goals should not be lumped into cold recall.
  **Prompt-backed does not mean recovered:** the anchors are deliberately wide
  ranges (e.g. liver 0.05–0.15 Pa) and a usable design must land within ±5 %
  of the exact conventional value, so a model that picks the wrong end of the
  range fails even with the hint — both models propose liver at 0.10 Pa
  (inside the range but 100 % off the 0.05 Pa convention), and neither
  recovers lymphatic. Both miss the kidney (`flash` 0.50 Pa — 25× off the
  0.02 Pa target; `pro` 0.05 Pa — 2.5×) and the primary-hepatocyte seeding
  density (0.33×). **The gate stops fabricated numbers; it cannot supply
  domain knowledge the model does not have.** That boundary is the honest
  headline, and it is exactly what a wet-lab user must not forget: verify the
  target, not just the arithmetic.
- **The 3D-spheroid set shows the calculator itself carrying domain
  conventions.** Both models land on the same table row — self-consistent
  **93 %** / usable **87 %** (13/15) — with hallucination **0.011** (`flash`)
  / **0.067** (`pro`), near the reading-set ceiling despite 3D culture being a
  knowledge-weak area for LLMs. The two cold entries (384-ULA 50 µL,
  hanging-drop 20 µL) are recovered at **100 %** by Labwright on both models
  but 0 % by the bare model, because those volumes live once in
  `SPHEROID_FORMATS` (the tool registry), not in model memory — the "calculator
  is the knowledge base" claim, made literal. The two failures per model are
  the honest residue, and each is *why* the set-level hallucination is
  non-zero:
  - both models mis-target the cross-domain doxorubicin goal (24 and 54
    spheroids against the gold 96) — internally consistent, `hall 0.000` on
    *that goal* (a target miss, not a fabricated number; the gate rejected the
    design), so it contributes 0 to the set-level hallucination;
  - `flash` additionally fails the blind hepatocyte-formation goal: it
    recovered the 1000 cells/spheroid target, but one of the plan's six
    derived spheroid fields was rejected by the verifier's physiological-range
    layer, so that goal scores **1/6 ≈ 0.167** — flash's whole set-level
    **0.011** is just this one goal (0.167 ÷ 15);
  - `pro` returns silence on the one-line sphere-volume goal — no design at
    all, which scores **1.0** and is pro's whole set-level **0.067** (1.0 ÷ 15).
  Single-run point estimates; the model-pair differences are noise at n=15.
- **The plate-culture set is where every memory system collapses to ~0 %.**
  The three naive systems land at 0 % usable on both models (self-verify
  `flash` hallucination **0.929** — it overwrote correct numbers with confident
  wrong ones on almost every goal), while Labwright holds **86 %** (`flash`) /
  **64 %** (`pro`). This is the *strictest* cross-check in the benchmark: each
  culture answer is re-derived from plate_format + seeding density + wells, and
  a single extra field the goal did not ask for makes the whole entry
  unverifiable. The 4 blind-`cold` recall cells (PHH sandwich density,
  plate-table volumes) are exactly where bare fails — the numbers live in the
  `CULTURE_*` tables, not model memory. Labwright's own non-zero cells on this
  set are the gate catching its own errors, not a failed gate: `flash` **0.071**
  is one silence (the strict hemocytometer goal `plate-hemocytometer-seed-96well`
  produced no design — hall 1.0), and `pro` **0.043** is two goals
  (`plate-96well-total-medium`, the blind-`cold` `blind-96well-area-and-medium`)
  where the verifier rejected one or two derived fields (calculation error) —
  rejected fields, never fabricated numbers.
- **The perfused-PK set is the arithmetic step-up.** Labwright is
  self-consistent **100 %** / usable **79 %** (`flash`) and **86 %** (`pro`)
  with hallucination **0.000**. PK is a *good* news story for the naive
  systems: because most goals hand over the formula's raw numbers, soft-gate
  reaches **50 %** usable and self-verify **29 %** — the same single-step
  arithmetic where those systems occasionally succeed. Labwright's remaining
  gap is the two blind `prompt-backed` propranolol/antipyrine targets (E = 0.8
  and 0.1 are inside the prompted classification range but the exact number is
  not), plus one unit-trap entry where the unit layer caught the mM→µM
  conversion before it entered the plan. The two genuine **unit traps** (mM-vs-µM
  and min-vs-h) are recovered cleanly by Labwright on both models.
- **The fine-tuned extractor separates in-distribution recall from honest
  generalization.** On the reading set — in-distribution, targets reused in its
  synthetic training — it is usable **79 %** / self-consistent **92 %**,
  essentially closing the gap to the API models' extraction step. On the
  plate-culture set (also in-distribution) it is **100 %** self-consistent /
  **0.000** hallucination, beating even Labwright's consistency, though usable
  drops to **64 %** because it reads fewer of the blind-`cold` recall targets.
  On the spheroid set — out-of-distribution, a domain it never trained on —
  usable collapses to **0 %** and hallucination rises to **0.611**: the
  extractor proposes raw inputs the calculators cannot honour, and the gate
  rejects the design. The perfused-PK set repeats the collapse (usable **0 %**,
  hallucination **0.714**) — also a domain the extractor never saw. That OOD
  collapse is the honest boundary of a fine-tuned extractor: strong on what it
  saw, blind beyond it. (The extractor's bars are identical under flash and pro
  by construction.)

## Cross-provider check: does the gate transfer to Kimi Code?

The table above is one backend (DeepSeek). To check the architecture is
backend-agnostic, the same five sets × four systems were re-run against two
Kimi Code models (**`kimi-for-coding`** and **`k3`**, OpenAI-compatible coding
endpoint) under the identical harness. The headline: **the Labwright benefit
transfers to any backend that can reliably run the tool loop — and collapses
for one that cannot.**

- **k3 ≈ DeepSeek.** On the 24-reading set, Labwright turns k3 from 8 % bare →
  **92 % usable** (flash 88 %, pro 100 %), self-consistent 96 %, hallucination
  0.042. k3's two reading misses (`lung-alveolar-shear` — never called
  `submit_design`; `selfconsistent-channel-volume` — wrong target) are *not* the
  three goals flash misses (`power-80-effect-half`, `reynolds-laminar-check`,
  `selfconsistent-pressure-drop-40mm`); k3 in fact hits all three. The misses
  are per-backend tool-loop idiosyncrasies, not systematic blind spots in
  particular goal types.
- **The transfer is set-wide, not reading-only.** Across the other four sets k3
  lands **47 %** usable blind, **73 %** spheroid, **93 %** culture and **86 %**
  PK — the same transfer the DeepSeek backends show — while kimi-for-coding
  stays at **0 %** usable on blind, culture and PK. kimi-for-coding's one
  partial success is the 15-spheroid set (33 % usable, up from 20 % bare): a
  design space simple enough that its tool-loop defect is not exercised on every
  goal. The pattern is uniform: a backend that can run the loop gets the
  Labwright benefit; one that cannot stays flat or worsens.
- **kimi-for-coding fails the tool loop** on 4 of the 5 sets (reading, blind,
  culture, PK all **0 %** usable). On the 24-reading set it called
  `submit_design` on **1/24** goals, and that design missed the target; the
  other 23 never reached `submit_design` at all. On a traced goal it fixated on
  calling `wall_shear_stress` with `viscosity_pas=0`, replayed the same
  validation error (`input must be > 0`) across all 12 iterations, and never
  corrected itself. Notably it is *better without* the agent loop (soft-gate
  reaches 8 % usable on reading, Labwright 0 %): for a backend that cannot
  self-correct a tool argument, the extra machinery is net negative. That is an
  honest boundary condition on the architecture, not a cherry-picked failure.

| set | model | bare | soft-gate | self-verify | Labwright |
|---|---|---|---|---|---|
| 24-reading | `kimi-for-coding` | 4 % | 8 % | 0 % | **0 %** |
| 24-reading | `k3` | 8 % | 8 % | 0 % | **92 %** |
| 15-blind | `kimi-for-coding` | 0 % | 0 % | 0 % | **0 %** |
| 15-blind | `k3` | 0 % | 0 % | 0 % | **47 %** |
| 15-3D-spheroid | `kimi-for-coding` | 20 % | 13 % | 7 % | **33 %** |
| 15-3D-spheroid | `k3` | 13 % | 7 % | 13 % | **73 %** |
| 14-plate-culture | `kimi-for-coding` | 0 % | 0 % | 0 % | **0 %** |
| 14-plate-culture | `k3` | 7 % | 7 % | 0 % | **93 %** |
| 14-perfused-PK | `kimi-for-coding` | 21 % | 36 % | 29 % | **0 %** |
| 14-perfused-PK | `k3` | 29 % | 36 % | 29 % | **86 %** |

![Cross-provider usable rate: the four backends (flash, pro, k3, kimi-for-coding) on the five sets, Labwright (left) vs bare-LLM (right). flash and pro are near the ceiling with the gate; k3 transfers to a new backend in the same family; kimi-for-coding — the backend that cannot run the tool loop — collapses to 0 % everywhere except its one spheroid success. The two panels share one y-axis.](paper/fig_model_compare.png)

*Usable designs (%). Full per-system self-consistent / hallucination columns are
in the committed result files (`results/eval_{set}_{k3,kimicode}.json`). The
five sets × two backends are the complete sweep. Config note: the Kimi runs used
temperature **0.6** with thinking disabled; the DeepSeek runs used **0.2** — the
Kimi coding endpoint's plain completion path validates temperature to 1.0, and
the Labwright request shape (thinking-disabled `extra_body`) accepts 0.6
(`LABWRIGHT_TEMPERATURE` overrides the 0.2 default). A higher temperature cannot
explain k3's high usable rates (it would if anything hurt a consistency-based
metric), and kimi-for-coding's failure is argument fixation, not temperature
sensitivity. The fine-tuned extractor is a fixed local model and is not
re-benchmarked per backend.*

## Reproducibility: prompts, models & provenance

The benchmark is an ablation of *prompts and stage structure* on fixed models, so
both are pinned and committed. Everything below is reproducible from the repo
alone — no unrecorded prompt, model or scoring choice.

**Models.** All benchmark rows use the DeepSeek v4 API
(`https://api.deepseek.com`, OpenAI-compatible): **`deepseek-v4-flash`**
(cheap, thinking disabled) and **`deepseek-v4-pro`**. Temperature **0.2**,
thinking **disabled** (`LLMClient(disable_thinking=True)` default) — the
arithmetic lives in the calculators, not the model. Labwright's agent runs the
same client at temperature 0.2 with a 12-iteration tool budget
(`--max-iterations 12`). `LABWRIGHT_MODEL` / `LABWRIGHT_BASE_URL` override the
model; any OpenAI-compatible model works, but the committed numbers are exactly
these two. These are API models, so no weight pin is possible; the API snapshots
are the models as served on the run dates in the result JSONs (`generated_at`).

A cross-provider sweep over the same five sets is run against the **Kimi Code**
endpoint (`https://api.kimi.com/coding/v1`), models `kimi-for-coding` and `k3`,
so the same protocol can be read across provider families (temperature **0.6** —
the endpoint's plain completion path validates temperature to 1.0, and the
Labwright request shape (thinking disabled) accepts 0.6; `LABWRIGHT_TEMPERATURE`
overrides the 0.2 DeepSeek default). Rows land in
`results/eval_*_kimicode.json` / `results/eval_*_k3.json` and are summarized in
the cross-provider table above.

**The three LLM-memory prompts** are the controllable variables of the ablation,
so they are pinned verbatim (with the exact per-goal key lists) in
[`eval/README.md`](eval/README.md#prompts--models-verbatim) — `bare_prompt_for`,
`soft_gate_prompt_for` and `self_verify_prompt_for` in `eval/benchmark.py`.

**The Labwright system prompt** (`labwright/agent/agent.py`, `SYSTEM_PROMPT`)
is the treatment under test, not an unrecorded variable: it forbids inventing
computed numbers, requires every derived value to come from the calculator
tools, mandates `submit_design` with raw inputs only, and — critically for the
blind set — *leaks physiological anchors* ("Hepatic sinusoidal shear ≈
0.05-0.15 Pa; lung alveolar-capillary ≈ 0.03 Pa; microvascular endothelium ≈
0.1-1 Pa"), plus the PK classification anchors (propranolol is a
high-extraction/flow-limited probe, antipyrine a low-extraction/capacity-limited
probe). The blind goals whose target falls inside one of those ranges are
labelled `prompt-backed`; the others are `cold`.

**Fine-tuned extractor scores** (`results/extractor/eval_report.json`,
n = 400 eval rows + 12 blind goals — the *pre-expansion* blind set, before it
grew to 15 — Qwen2.5-1.5B-Instruct LoRA, adapter at
`results/extractor/lora`):

| system | JSON parse | schema-ok | extract→verify consistency | field recovery (≤5 %) | target recovery |
|---|---|---|---|---|---|
| **fine-tuned 1.5B** | 1.0 | 0.9976 | **0.9976** | 0.72 | 0.0 |
| `deepseek-v4-flash` (untuned) | 1.0 | 0.4005 | 0.4005 | 0.3875 | 0.0 |
| `deepseek-v4-pro` (untuned) | 1.0 | 0.4005 | 0.4005 | 0.3875 | 0.2 |

Target recovery is 0 even for the fine-tuned model: the extractor recovers the
*raw inputs* a goal implies, not the physiological target number (that is the
agent's job). `mean_field_rel_error` is 0.0059 for the fine-tuned model.
`target_recovery` is **not** a rate over the 400 eval rows (nor the 412 with
the blind goals): it is scored only on the blind goals that carry a
physiological shear target **and** whose extracted raw built a design — a
single-digit subset (at most the 10 shear-bearing goals of the 12-goal blind
set; the untuned extractor builds fewer). `pro`'s 0.2 is therefore ~1 hit
within ±20 % out of a few such goals — small-n noise, not a 20 % capability.

**Statistical caveat.** The headline cells in the table above are **single
runs** over 24/15 goals. A 5-seed re-run of the 24-reading set
(`results/eval_seed_benchmark.json`, 24 goals × 5 seeds = 120 trials per
system/model) gives Wilson 95 % CIs (`eval/ci.py`):

| model | system | usable rate (k/n) | 95 % CI |
|---|---|---|---|
| `flash` | bare | 8/120 = 0.067 | [0.034, 0.126] |
| `flash` | **Labwright** | 111/120 = 0.925 | [0.864, 0.960] |
| `pro` | bare | 13/120 = 0.108 | [0.064, 0.177] |
| `pro` | **Labwright** | 115/120 = 0.958 | [0.906, 0.982] |

The qualitative ordering (Labwright ≫ bare; flash vs pro within ~5 %) is
stable across seeds; the blind-set cells are single-run point estimates and
should be read as such.

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

## License & citation

Apache-2.0. Built and maintained by [qgeng1465](https://github.com/qgeng1465).

```bibtex
@software{labwright,
  author = {Geng, Q.},
  title = {Labwright: the AI bench copilot that gets your numbers right},
  year = {2026},
  url = {https://github.com/qgeng1465/labwright},
  license = {Apache-2.0}
}
```

**Disclaimer:** Labwright is an experimental-design aid, not medical device
software. Always verify generated protocols against your own lab's
standard operating procedures and safety regulations.
