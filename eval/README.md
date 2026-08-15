# Labwright benchmark — paper evidence

> **Version note (Aug 2026).** The blind gold set is currently **15** goals
> (expanded 6 → 12 → 15). Anywhere "12 blind goals" appears below it means the
> **pre-expansion** set — the extractor eval (`extract/eval.py`,
> `results/extractor/eval_report.json`) and the thinking-ablation grid were
> run on the 12-goal set, before the scenario entries (unit-ambiguity,
> multi-target) and the 24-well medium-volume goal grew it to 15. Those rows
> are historical and **not** directly comparable to the 15-goal cells; every
> table below states its own set.

## Question

Do LLM-written wet-lab designs contain hallucinated numbers, and does
constraining the LLM to *propose* while *calculators compute* fix it?

## Protocol

On every gold-standard experiment, run the systems being compared:

| system | behavior |
|---|---|
| bare-LLM | the model is given the full design JSON schema and asked to write the complete design *including derived numbers*, from memory |
| soft-gate | bare-LLM plus a "check yourself" instruction: re-derive your own derived numbers from your own reported geometry/flow before finalising. No calculators, no verifier. |
| self-verify | two LLM passes: the model proposes numbers from memory, then is handed back its *own reported raw inputs* and asked to recompute the derived numbers itself. The naive "use the LLM as the verifier" alternative to Labwright's deterministic verifier. |
| Labwright | the model proposes raw inputs; derived numbers come from `labwright.calc` and pass `labwright.verify` |
| finetuned-ext | a fixed local Qwen2.5-1.5B LoRA run through the same design path (goal → raw → derive → verify), scored identically; model-independent, so identical under flash and pro |

The three LLM-memory systems (bare, soft-gate, self-verify) are scored by
**exactly the same** rules — extraction, ±5 % consistency tolerance,
verifiability, unverifiable=1.0. Only the prompt/stage structure differs, so any
measured difference between them is caused by the *approach*, not by scoring.

Six gold sets:

1. **`gold_experiments.json` — 24 "reading" goals.** Every goal states the
   answer (the geometry/flow/density/effect-size, or the physiological target
   number). These test whether the pipeline can *extract the stated numbers and
   drive the calculators to them*. They do **not** test domain knowledge.
2. **`gold_blind.json` — 15 "recall" goals.** The goal states no number at all
   ("recapitulate physiological venular wall shear"); the model must supply the
   canonical target from its own knowledge. **Eight** are `cold` (answer in
   neither goal nor the system prompt: kidney PTEC, arterial, HepG2 density,
   primary-hepatocyte density, pulmonary artery, gut, retinal arteriole,
   24-well medium volume); **five** are `prompt-backed` (liver, lung, BBB,
   venular, lymphatic — the target sits inside a range listed in the Labwright
   system prompt, so the model must still *select* the right value; venular
   0.3 Pa and lymphatic 0.2 Pa fall inside the prompt's "microvascular
   endothelium ≈ 0.1-1 Pa", by the same range-contains-answer criterion as
   liver/lung/BBB); **two** are scenario-only (the magnitude is stated, so they
   exercise a failure mode, not cold recall):
   - **unit-ambiguity** (`blind-kidney-ptec-unit-ambiguity`) — the goal states
     "0.2 dyn/cm²" and asks for Pa; a dyn-as-Pa misread is exactly 10× off
     (0.2 Pa instead of 0.02 Pa).
   - **multi-target** (`blind-bbb-shear-residence-multitarget`) — two targets
     jointly satisfiable at Q ≈ 40 µL/min in a 400×100 µm × 100 mm channel
     (shear 1.0 Pa *and* residence 6.0 s); the model must hit both.
   Every entry pins a citable source in its `source` field; no number is
   invented, and `tests/test_metrics.py` re-derives each scenario entry to prove
   the gold itself is satisfiable (an unwinnable gold would inflate failure
   rates for the wrong reason).
3. **`gold_spheroid.json` — 15 "3D-spheroid" goals.** A third domain —
   3D culture (spheroid/organoid) — where spheroid conventions are fragmented
   across ULA and hanging-drop vendors, a deliberately knowledge-weak area for
   LLMs. Four are reading (solid-sphere geometry and cells-per-size, stated in
   the goal); three exercise a standard vessel working volume (96-ULA 100 µL,
   384-ULA 50 µL, hanging-drop 20 µL — partial-info, the convention must be
   recalled); four are failure-mode scenarios:
   - **unit-ambiguity** (`spheroid-um-mm-unit-ambiguity`) — the goal states a
     diameter in mm and asks for µm; a mm-as-µm misread is exactly 1000× off.
   - **growth** (`spheroid-growth-72h`) — exponential growth from seed count.
   - **multi-target** (`spheroid-96well-total`) — total cells *and* total medium
     for a full 96-ULA plate, jointly.
   - **cross-domain** (`spheroid-doxorubicin-dosing`) — spheroid size/medium
     *and* a DMSO vehicle fraction from a drug stock.
   Four are blind: two `prompt-backed` (1000 cells/spheroid and the 96-ULA
   100 µL volume are stated in the Labwright system-prompt anchors) and two
   `cold` (the 384-ULA 50 µL and hanging-drop 20 µL volumes are in neither the
   goal text nor the system prompt). Every entry pins a citable source or an
   explicit self-consistent derivation; `tests/test_gold_spheroid.py`
   re-derives each entry from its goal raws with `calc/spheroid.py`.
4. **`gold_cell_culture.json` — 14 plate-culture goals.** A fourth domain — 2D
   plate culture (wells, seeding density, counting, viability, confluence).
   Ten are reading (plate geometry / density stated in the goal) and four are
   blind-`cold` (the model must recall the pinned PHH sandwich-plating density
   or a plate-table working volume; the answers live in the `CULTURE_*` tables,
   not in the prompt). This is the *strictest* cross-check in the benchmark:
   every answer is re-derived from plate_format + seeding density + wells, and
   one extra field the goal did not ask for makes the whole entry unverifiable.
   `tests/test_gold_culture.py` re-derives every entry through `calc/culture.py`.
5. **`gold_pk.json` — 14 perfused-system PK goals.** A fifth domain —
   single-compartment pharmacokinetics on-chip (extraction ratio, clearance,
   half-life, steady-state accumulation, mass cleared). Twelve are
   reading/scenario (every input stated, or the formula's raw numbers given,
   including two **unit traps** — mM-vs-µM `pk-mM-unit-trap` and min-vs-h
   `pk-half-life-min-trap`) and two are blind `prompt-backed` (propranolol
   high-extraction / antipyrine low-extraction — the classification is stated
   in the Labwright system prompt's PK anchor, the exact target number is not).
   Equations are pinned to Rowland & Tozer and Gibaldi & Perrier; the propranolol
   intrinsic-clearance design target cites Baudoin et al. (doi:10.1002/jps.23796).
   `tests/test_gold_pk.py` re-derives every entry through `calc/pk.py`.

6. **`gold_new_domains.json` — 14 post-v1 organ-on-chip goals.** Seven more
   domains beyond microfluidics/culture/spheroid/PK — barrier integrity
   (TEER / Papp / clearance), dissolved-pO2 (Krogh penetration depth,
   necrotic core), gravity-driven pumpless perfusion (rocking WSS, OSI),
   lung ALI + breathing stretch (breaths/min, strain rate, ALI film),
   pulsatile / cardiac waveform (Womersley number, OSI, PI), multi-organ
   allometric scaling (organ flow fraction, mass-proportional cells) and
   chemotaxis gradients (steepness, relaxation time). All 14 are
   complete-info (every raw stated) so this set isolates whether the new
   calculators and Blocks integrate end to end; every expected value is
   re-derived by the real calculators in
   `eval/make_gold_new_domains.py`, and each entry pins a citable source.

### Prompts & models (verbatim)

The ablation changes *only* prompt/stage structure on fixed models, so both are
pinned. All rows use the DeepSeek v4 API (`https://api.deepseek.com`,
OpenAI-compatible): **`deepseek-v4-flash`** and **`deepseek-v4-pro`**, at
temperature **0.2**, thinking **disabled** (`LLMClient(disable_thinking=True)`
default); Labwright's agent runs the same client at 0.2 with a 12-iteration
tool budget. These are API models — no weight pin exists; `generated_at` in each
result JSON records the run date. `LABWRIGHT_MODEL` / `LABWRIGHT_BASE_URL`
override, but the committed numbers are exactly these two models.

A cross-provider sweep over the same five sets runs against the **Kimi Code**
endpoint (`https://api.kimi.com/coding/v1`), models `kimi-for-coding` and `k3`
(`results/eval_*_kimicode.json` / `results/eval_*_k3.json`), at temperature
**0.6** — the endpoint's plain completion path validates temperature to 1.0,
and the Labwright request shape (thinking disabled) accepts 0.6;
`LABWRIGHT_TEMPERATURE` overrides the 0.2 DeepSeek default. The finetuned-ext rows come from
`eval/run_finetuned_benchmark.py` (a fixed local Qwen2.5-1.5B LoRA, adapter at
`results/extractor/lora`) and are grafted into both model files of their set by
`eval/merge_finetuned.py` — model-independent by construction.

The three LLM-memory prompts (`eval/benchmark.py`; each is joined with the
per-goal key list `_prompt_keys_for(gold)` and the goal text):

**bare** (`bare_prompt_for`):
```
You are a wet-lab design expert. For the goal below, compute the design numbers
yourself and return a single flat JSON object with ONLY these keys (use exactly
these names; do the arithmetic; omit nothing): <keys>.
Return ONLY the JSON object (no prose, no markdown fences).

Goal: <goal>
```

**soft-gate** (`soft_gate_prompt_for`) — bare + a check step, domain-appropriate:
```
... <bare preamble> ...
BEFORE you finalize: re-derive every derived flow number (<keys>) from your own
width_um/height_um/length_mm/flow_rate_uLmin/viscosity_pas/density_kgm3 using
the standard rectangular-channel formulas, and correct any value that does not
match.
Return ONLY the JSON object (no prose, no markdown fences).

Goal: <goal>
```
(culture goals substitute the plate-dimension / hemocytometer / viability
phrasing from `_CULTURE_DERIVED_KEYS`.)

**self-verify pass 2** (`self_verify_prompt_for`) — the model is handed its own
first-pass raw inputs and asked to recompute:
```
A design proposed these raw inputs: <key=value,…>.
Using the standard rectangular-channel microfluidic formulas, recompute EXACTLY
these derived values yourself (<keys>) from those inputs, and return a single
flat JSON object with ONLY those keys (use exactly these names; do the
arithmetic):
Return ONLY the JSON object (no prose, no markdown fences).
```

**Labwright system prompt** — the treatment under test, quoted in full in
`labwright/agent/agent.py` (`SYSTEM_PROMPT`). The parts that matter for the
blind set are the physiological anchors it leaks, which is why five blind goals
are labelled `prompt-backed`:

> Common physiological anchors (verify against literature before relying on
> them):
> - Hepatic sinusoidal shear ≈ 0.05-0.15 Pa (0.5-1.5 dyn/cm²); lung
>   alveolar-capillary ≈ 0.03 Pa; microvascular endothelium ≈ 0.1-1 Pa.

### Fairness, honestly stated

This is **not** a head-to-head of equal-resource systems; it is an *ablation*:
the bare condition is an LLM writing numbers from memory (the status quo), and
Labwright is the same LLM plus tools, a 12-iteration loop, the verifier, and a
physiological-anchor system prompt. Those are the treatment, and they favour
Labwright on iteration budget and anchor hints. The one asymmetry that favours
the bare model is scoring: it is retried up to 3×, allowed a ±5 % consistency
tolerance, and not required to prove its numbers. We do not claim the systems
are matched; we claim the comparison isolates what the calculators and the
verifier add to the bare model.

## Metrics (from `benchmark.evaluate`)

- **hallucination rate** — the fraction of a plan's *derived* fields the
  verifier rejects at error level, averaged over goals. A system that produces
  no checkable output scores 1.0. Because the denominator is derived fields per
  plan, not goals, one goal with a single rejected field moves the set-level
  number by 1/(goals × fields); and a raw-input absurdity the verifier flags
  (e.g. an unphysical doubling time) makes a plan *invalid* without moving
  hallucination at all — so hallucination and the valid/usable flag must be
  read together.
  - Labwright: derived numbers come from the calculators and the verifier
    recomputes them from the *same* calculators, so for a plan that submits
    cleanly this is **0 by construction**. That is the point: it is an
    architectural guarantee, not a measured win — the metric exists to *make*
    the guarantee checkable. It is non-zero only when the verifier actually
    rejects something: a `plan: false` run (no design) scores 1.0, and the
    3D-spheroid set shows the other case — `flash`'s hepatocyte-formation plan
    had 1 of 6 derived spheroid fields rejected by the physiological-range
    layer (0.167 → set-level 0.011). The guarantee is also **attack-tested**:
    `tests/test_gate_security.py` proves every other path a hallucinated number
    could take is closed — a derived field smuggled into `submit_design` is
    rejected (not silently dropped), a tampered plan field is caught by the
    verifier, and a number asserted in the design's own prose that contradicts
    the calculators is flagged by the prose-number gate.
  - bare-LLM: a number that does not follow from the model's own geometry/flow
    is a hallucination. A bare answer that reports geometry+flow but **no
    derived flow numbers** is *unverifiable* and scored 1.0 — the same
    convention as a Labwright run that never submits ("numbers you type are not
    trusted"). The committed results were recomputed with this rule in
    `recompute_honest.py`; earlier commits scored unverifiable silence as 0.0
    and over-stated the bare self-consistent rate.
- **self-consistent rate** — fraction of gold entries with hallucination rate 0.
- **usable rate** — fraction of gold entries that are self-consistent **and**
  recover every gold target within ±5 %. A design that is internally consistent
  but misses the physiological target is not usable.

New in the 15-entry blind run, every record also carries:

- **failure reason** (`classify_failure`) — one of `ok` / `silence` (nothing
  checkable produced) / `calculation_error` (numbers inconsistent or
  unverifiable) / `wrong_target` (internally consistent but misses the gold).
  This separates "model refused / ran out of budget" from "model fabricated"
  from "model aimed at the wrong physiology".
- **unit-misread rate** (`unit_misreads`) — a claimed value that is a clean
  multiple of a known alias ratio (dyn/cm² vs Pa = 10×, mL/min vs µL/min =
  1000×, ...) with the *right* magnitude is classified as a unit error, not an
  arithmetic one, using `labwright/verify/units.py` against the field's
  canonical unit. An entry flagged here is an extractor/converter failure, not a
  physics miss.
- **target-selection accuracy** — fraction of entries whose *headline* gold
  target (first expected key) is recovered within ±5 %.
- **blind split** — usable rate and hallucination rate reported separately for
  `cold` vs `prompt-backed` goals, so prompt-leaked answers never inflate the
  cold-recall claim.

### What the numbers do — and do not — mean

- **"0.000 hallucination"** means *no number entered a design unless a
  calculator produced it and the verifier re-proved it*. It does **not** mean
  "every design is physiologically correct". The gate cannot tell the model
  which target to aim at. It also does not rest on trust: since the 15-spheroid
  run, a prose-number gate cross-checks numbers the model writes in `rationale`
  / `caveats` against the plan's own values, so an asserted shear that the
  calculators do not reproduce is flagged, not silently carried in a sentence.
- **Recovery ≈ 0 on the 24-reading set** is *by construction*: every goal
  states the answer, and the self-consistent anchors are computed from the same
  equations Labwright uses. The real signal there is number-extraction and
  tool-calling.
- **The blind set is where target selection is actually tested.** There the
  usable rate collapses: `flash` 88 % → 40 %, `pro` 100 % → 47 % (see below).
  The gate held (hallucination 0.000 on every submitted plan) — Labwright
  produced clean, verified designs that aimed at the *wrong physiology*. That
  is the honest boundary of the guarantee.

## Status

- [x] Curate the 24-reading gold set. Every entry carries a provenance rule — a
      pinned source or an explicit `self-consistent` label. All anchors
      hand-checked against the governing equations.
- [x] Curate the blind gold set (`gold_blind.json`, expanded 6 → 12 → **15** in
      Aug 2026), each entry labelled `cold`/`prompt-backed`/scenario with a
      pinned source. Labels re-audited in Aug 2026 against the *actual*
      system-prompt ranges: venular and lymphatic sit inside "microvascular
      endothelium ≈ 0.1-1 Pa", so 5 are `prompt-backed` and 8 `cold`, plus 2
      scenario entries (unit-ambiguity, multi-target).
- [x] Run bare-LLM + Labwright on `deepseek-v4-flash` and `deepseek-v4-pro` on
      the 24-reading set → `results/eval_flash.json`, `results/eval_pro.json`.
      The runner checkpoints after every entry.
- [x] Run the competitor systems (soft-gate, self-verify) on both sets and both
      models → `results/eval_competitors_{flash,pro}.json`,
      `results/eval_blind_competitors_{flash,pro}.json`.
- [x] Re-run all four systems on the expanded **15**-blind set, with the
      failure-reason / unit-misread / target-selection / blind-split metrics →
      `results/eval_blind_{flash,pro}.json`.
- [x] Unit tests for the new metrics and scenario golds
      (`tests/test_metrics.py`), the unit/alias layer (`test_units.py`), the
      sanity bands (`test_sanity.py`), the safety boundary (`test_safety.py`)
      and provenance (`test_provenance.py`).
- [x] Thinking ablation grid (thinking ON for both models on both sets) →
      `results/eval_blind_{flash,pro}_thinking.json`,
      `results/eval_{flash,pro}_thinking.json`.
- [x] Reverse-verify published protocols + labelled synthetic controls:
      `eval/run_verify_batch.py` → `results/eval_verify_batch.json`.
- [x] Preprint draft, Colab notebook, HF Space scaffolding.
- [ ] Submit the preprint to bioRxiv; publish the HF Space.

## Results (honest)

All systems, all five sets, two models (`flash`, `pro`) plus the
model-independent `finetuned-ext`. Self-consistent = zero verifier errors;
usable = self-consistent *and* recovers every gold target within ±5 %. The
memory systems (bare, soft-gate, self-verify) are scored by identical
extraction/tolerance/unverifiable=1.0 rules — only the prompt/stage structure
differs.

| model | set | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `flash` | 24-reading | bare-LLM | 0 % | 0 % | 1.000 |
| `flash` | 24-reading | soft-gate | 12 % | 12 % | 0.875 |
| `flash` | 24-reading | self-verify | 0 % | 0 % | 0.792 |
| `flash` | 24-reading | **Labwright** | **88 %** | **88 %** | **0.125** |
| `flash` | 24-reading | finetuned-ext (in-dist) | 100 % | 92 % | 0.000 |
| `pro` | 24-reading | bare-LLM | 12 % | 12 % | 0.875 |
| `pro` | 24-reading | soft-gate | 8 % | 8 % | 0.917 |
| `pro` | 24-reading | self-verify | 0 % | 0 % | 0.750 |
| `pro` | 24-reading | **Labwright** | **100 %** | **100 %** | **0.000** |
| `pro` | 24-reading | finetuned-ext (in-dist) | 100 % | 92 % | 0.000 |
| `flash` | 15-blind | bare-LLM | 7 % | 0 % | 0.933 |
| `flash` | 15-blind | soft-gate | 13 % | 0 % | 0.867 |
| `flash` | 15-blind | self-verify | 0 % | 0 % | 0.611 |
| `flash` | 15-blind | **Labwright** | **100 %** | **40 %** | **0.000** |
| `flash` | 15-blind | finetuned-ext | 93 % | 27 % | 0.067 |
| `pro` | 15-blind | bare-LLM | 7 % | 0 % | 0.933 |
| `pro` | 15-blind | soft-gate | 13 % | 0 % | 0.867 |
| `pro` | 15-blind | self-verify | 0 % | 0 % | 0.733 |
| `pro` | 15-blind | **Labwright** | **100 %** | **47 %** | **0.000** |
| `pro` | 15-blind | finetuned-ext | 93 % | 27 % | 0.067 |
| `flash` | 15-3D-spheroid | bare-LLM | 20 % | 20 % | 0.800 |
| `flash` | 15-3D-spheroid | soft-gate | 13 % | 13 % | 0.867 |
| `flash` | 15-3D-spheroid | self-verify | 20 % | 20 % | 0.569 |
| `flash` | 15-3D-spheroid | **Labwright** | **93 %** | **87 %** | **0.011** |
| `flash` | 15-3D-spheroid | finetuned-ext (in-dist) | 73 % | 67 % | 0.267 |
| `pro` | 15-3D-spheroid | bare-LLM | 27 % | 27 % | 0.733 |
| `pro` | 15-3D-spheroid | soft-gate | 27 % | 27 % | 0.733 |
| `pro` | 15-3D-spheroid | self-verify | 40 % | 20 % | 0.400 |
| `pro` | 15-3D-spheroid | **Labwright** | **93 %** | **87 %** | **0.067** |
| `pro` | 15-3D-spheroid | finetuned-ext (in-dist) | 73 % | 67 % | 0.267 |
| `flash` | 14-plate-culture | bare-LLM | 0 % | 0 % | 0.893 |
| `flash` | 14-plate-culture | soft-gate | 0 % | 0 % | 0.893 |
| `flash` | 14-plate-culture | self-verify | 0 % | 0 % | 0.929 |
| `flash` | 14-plate-culture | **Labwright** | **93 %** | **86 %** | **0.071** |
| `flash` | 14-plate-culture | finetuned-ext (in-dist) | 93 % | 64 % | 0.071 |
| `pro` | 14-plate-culture | bare-LLM | 7 % | 7 % | 0.750 |
| `pro` | 14-plate-culture | soft-gate | 7 % | 7 % | 0.786 |
| `pro` | 14-plate-culture | self-verify | 0 % | 0 % | 0.821 |
| `pro` | 14-plate-culture | **Labwright** | **86 %** | **64 %** | **0.043** |
| `pro` | 14-plate-culture | finetuned-ext (in-dist) | 93 % | 64 % | 0.071 |
| `flash` | 14-perfused-PK | bare-LLM | 50 % | 36 % | 0.500 |
| `flash` | 14-perfused-PK | soft-gate | 50 % | 50 % | 0.500 |
| `flash` | 14-perfused-PK | self-verify | 79 % | 29 % | 0.214 |
| `flash` | 14-perfused-PK | **Labwright** | **100 %** | **79 %** | **0.000** |
| `flash` | 14-perfused-PK | finetuned-ext (in-dist) | 50 % | 50 % | 0.500 |
| `pro` | 14-perfused-PK | bare-LLM | 43 % | 36 % | 0.536 |
| `pro` | 14-perfused-PK | soft-gate | 50 % | 36 % | 0.500 |
| `pro` | 14-perfused-PK | self-verify | 79 % | 29 % | 0.214 |
| `pro` | 14-perfused-PK | **Labwright** | **100 %** | **86 %** | **0.000** |
| `pro` | 14-perfused-PK | finetuned-ext (in-dist) | 50 % | 50 % | 0.500 |

The memory systems never produce a usable *design* on the flow sets (24-reading,
15-blind), and the two naive "fixes" do not help there. On the 24-reading set the
only usable memory-system entries are three single-arithmetic-step goals — a
residence time with geometry given, a channel volume, a mean velocity — where
the goal supplies every input and no design choice remains (`pro` bare and
`flash` soft-gate both reach 12 % there, `pro` soft-gate 8 %); every goal that
requires choosing geometry and flow to hit a target scores 0 % usable for all
three memory systems, both models. The 3D-spheroid set breaks that floor after
the fairness fix below: once string vessel formats are actually extracted, the
pure-geometry and single-lookup goals become *checkable*, and bare reaches
**20 % / 27 %** usable (`flash` / `pro`) — recovering sphere volume/diameter
from the goal's own raws and the 384-ULA 50 µL convention. That is the entire
usable ceiling: the 96-ULA working volume is remembered wrong (200 µL vs the
correct 100 µL), hanging-drop answers arrive in plate-culture vocabulary, and
the doxorubicin dosing target is missed. Soft-gate (a "re-check yourself"
prompt) adds nothing — its usable ≤ bare on both models (13 % / 27 %). Self-
verify's second pass now *helps* on the spheroid set's simpler arithmetic
(hallucination drops 0.800 → 0.569 `flash`, 0.733 → 0.400 `pro`) but still
rarely reaches a usable design (20 % both models): a recompute pass that is
itself wrong (e.g. 1000 cells of 15 µm "→" 1000 µm spheroids) is now visible as
a contradiction rather than carried as a fact. Only Labwright's deterministic
calculators + verifier reach usable > 30 % on any set.

The **plate-culture** set is the strictest cross-check and the memory systems'
floor: every culture answer is re-derived from plate_format + seeding density +
wells, and one extra field the goal did not ask for voids the whole entry, so
all three memory systems land at **0 % usable** on both models (self-verify
`flash` hallucination **0.929** — its recompute pass overwrote correct numbers
with confident wrong ones). Labwright holds **86 %** (`flash`) / **64 %** (`pro`);
its own non-zero hallucination here is the gate catching its errors, not a
failed gate — `flash` 0.071 is one silence (the strict hemocytometer goal
`plate-hemocytometer-seed-96well` produced no design), `pro` 0.043 is two goals
(`plate-96well-total-medium`, the blind-`cold` `blind-96well-area-and-medium`)
where the verifier rejected one or two derived fields (calculation error).
The **perfused-PK** set is the arithmetic step-up where the naive systems do
best: most goals hand over the formula's raw numbers, so soft-gate reaches
**50 % usable** and self-verify **29 %** (single-step arithmetic is exactly the
regime where those systems occasionally succeed), while Labwright is
self-consistent **100 %** / usable **79 %** (`flash`) / **86 %** (`pro`) with
hallucination **0.000**, its residue being the two blind propranolol/antipyrine
targets and one mM→µM unit-trap that the unit layer caught.

The **finetuned-ext** rows are the 11-domain multi-block extractor (lora_v5,
fine-tuned on ~49.8k synthetic rows covering flow, culture, spheroid, pk and
the seven post-v1 domains; single-block lora_v3 on 49,500 rows and multi-block
lora_v4 are the earlier generations). Reading is the in-distribution ceiling:
**100 % self-consistent**, **92 % usable**, **0.000** hallucination (23/24 of
those goals are verbatim gold pairs — memorization, not transfer; the single
regression vs v4 is the 400×100-shear goal, recovery residual 0 → 3.0).
Spheroid — previously out-of-distribution and at **0 % usable** — is now a
trained domain: **73 % self-consistent**, **67 % usable** (hallucination
**0.267**), with **9/14** of the *novel* goals recovering. pk is all-novel
(none of its goals is a verbatim pair): **50 % usable**, hallucination
**0.500**, 7/14 novel goals recover (v4 reached 8/14). Culture stays in-dist at
**64 % usable** (hallucination **0.071**). The blind set drops to **27 %
usable** — the goal does not state the target, so a clean design can miss the
physiology — with hallucination held at **0.067** (14/15 self-consistent).
Against lora_v4, v5 adds hand-written-register variants for the seven post-v1
domains, and that is the one clear win: the 14 new-domain goals go from **0 %
to 29 % usable** (**4/14**, hallucination **0.446** = six silence rows + one
partial row). The remaining honest boundary is still those domains: 10/14 of
the hand-written goals do not transfer from the synthetic phrasing — that
boundary is improved but not solved.

The blind-set drop is the honest headline: when the goal does not hand over the
target, Labwright's verified designs hit the wrong physiology. On the expanded
15 goals:
- `flash` recovers **6/15**: arterial 1.5 Pa, HepG2 seeding, 24-well medium
  volume, lung 0.03 Pa, and both scenario goals — the dyn/cm²-as-Pa unit test
  and the shear + residence joint target.
- `pro` recovers **7/15**: arterial, HepG2 seeding, 24-well medium, venular
  0.3 Pa, lung 0.03 Pa, BBB 1.0 Pa, and the unit-ambiguity goal. (`pro`'s
  multi-target run hits the shear but misses the residence time 0.5×, so it is
  **not** counted as usable.)
Both usable rates are single-run point estimates with wide error bars: the
95 % Wilson CI around 6/15 = 40 % is **20–64 %**, around 7/15 = 47 % it is
**25–70 %** — n=15 is too thin to separate the two models, or either from the
cold-only 38 % below.
**Cold-only sub-rates:** five of the 15 goals are `prompt-backed` (liver, lung,
BBB, venular, lymphatic), so the headline 40 %/47 % usable overstates recall on
the genuinely cold goals — on the eight cold entries `flash` and `pro` each
recover only 3 (arterial, HepG2 seeding, 24-well medium), i.e. cold-only usable
≈ **38 % / 38 %**, each with a 95 % Wilson CI of 14–69 %; n=8 is still too thin
to separate the two models, and cold recall is nowhere near the reading set. Of
the recoveries that look like domain knowledge, only those three are actually
cold; the others (lung, BBB, venular) sit inside prompted ranges. Remove the
two scenario-only goals (they state the magnitude, so they test a failure mode,
not recall) and the *domain*-target recovery is **4/13 = 31 %** for `flash` and
**6/13 = 46 %** for `pro` — scenario goals should not be lumped into cold
recall.
Both models correctly select the prompt-backed entries they are primed for
(`pro` recovers venular 0.3 Pa, lung and BBB; `flash` recovers lung) yet both
miss liver (0.05 Pa): they propose the mid-range 0.10 Pa — inside the prompt's
0.05–0.15 Pa range but 100 % off the low-shear convention, and usable demands
the exact target within ±5 %, so range-membership alone does not recover it —
and neither recovers lymphatic (0.2 Pa). The remaining cold entries are mostly
wrong (recovery = relative error of the proposed shear vs the target): kidney
PTEC 0.02 Pa is proposed at 0.50 Pa (`flash`, recovery 24) / 0.05 Pa (`pro`,
recovery 1.5), gut epithelium 0.002 Pa at 0.005 Pa (`flash`, recovery 1.5) /
~0.013 Pa (`pro`, recovery 5.7), retinal arteriole 0.72× off (`flash`) / 0.63×
off (`pro`), pulmonary artery 0.9× off (`flash`) / 0.25× off (`pro`), and the
primary-hepatocyte seeding density 0.33× off on both. The gate never failed —
every plan was internally verified — it just could not supply domain knowledge
the model did not have.

The new per-entry metrics on the same run: the failure-reason breakdown for
Labwright is `ok` 6/15 (`flash`) / 7/15 (`pro`) and `wrong_target` 9/15 / 8/15 —
`silence` and `calculation_error` do not occur on the blind set (a plan either
submits verified or fails to submit). The spheroid set is the exception that
proves the taxonomy: `flash` 1/15 `calculation_error` (hepatocyte-formation —
one derived field rejected by the verifier) and `pro` 1/15 `silence`
(sphere-volume — no design). The memory
systems fail almost entirely with `calculation_error` (12/15 `flash` / 11/15
`pro` for bare). **Target-selection accuracy** (headline target within ±5 %) is 40 %
(`flash`) / 53 % (`pro`) for Labwright vs 33 % for the bare model — a bare model
names the right target about a third of the time but never produces an
internally consistent design behind it. The **unit-misread layer** fires rarely
and never inside a Labwright plan (0.000 for both models): the probable
dyn/cm²-as-Pa misreads are caught on the memory side (flash self-verify on
liver; pro bare/soft-gate on kidney PTEC, both proposing 0.20 Pa for the
0.02 Pa target; pro self-verify on the unit-ambiguity scenario), and the two
scenario goals exist precisely to prove the layer would catch it inside a
verified design too — `flash` recovers both, `pro` the unit-ambiguity one.

**The 15-3D-spheroid set is the third domain and the strongest separation.** It
mixes stated-goal geometry (spheroid volume/diameter from the goal's own numbers)
with the fragmented conventions of 3D culture (ULA and hanging-drop working
volumes, hepatocyte spheroid seeding density) that the model must recall. On the
stated-goal arithmetic Labwright is close to the reading set: `flash` 13/15 and
`pro` 13/15 usable (both **87 %**, Wilson CI 62–96 %), and both models recover
every gold target within tolerance on 13 of the 15 goals. The conventions still
generalize — both models' Labwright recovers the cold 384-ULA 50 µL and
hanging-drop 20 µL at **100 %** (2/2 each) precisely because those values live
in the calculator's tool registry, not in model memory. The bare model's memory
of the same conventions is patchy: after the fairness fix it scores **0 %**
(`flash`) / **50 %** (`pro`) on the two cold convention goals (the one recovery,
pro on hanging-drop 20 µL, is a correct recall, not a lookup), and on the
reading set the format string is no longer the blocker — a parseable
"384-well ULA plate" now scores clean at 50 µL on both models — but a wrong
memory still fails (96-ULA reported at 200 µL vs the correct 100 µL, and
hanging-drop answers arrive in plate-culture vocabulary). This is the paper's
"the calculator is the knowledge base" claim made literal: a convention encoded
once in `SPHEROID_FORMATS` is available to both models on every run, cold or
not, while the bare model must re-recall each value from memory. The remaining failures
are the honest residue, and each is also why the set-level hallucination is
non-zero. Both models mis-target the doxorubicin dosing goal (`flash` 24
spheroids against the gold 96, `pro` 54 — internally consistent, `hall 0.000`
on *that goal*, so the gate held and the design was rejected: a target miss,
not a fabricated number, contributing 0 to the set-level hallucination);
`flash` additionally fails the blind hepatocyte-formation goal — it recovered
the correct 1000 cells/spheroid, but the verifier's physiological-range layer
rejected one of the plan's six derived spheroid fields (`hall 0.167`, the whole
of flash's set-level 0.011); `pro` returns *silence* on the pure-geometry
`spheroid-volume-from-diameter` goal — a one-line sphere-volume derivation, yet
no design was submitted at all (scores hall 1.0, the whole of pro's set-level
0.067). Target-selection accuracy is 93 % (`flash` Labwright) and 87 % (`pro`
Labwright, equal across systems) — the model names the right target, but only
the calculator path turns that into a verified, usable design. As with the
blind set, these are single-run point estimates; the differences between the
two models' two failures each are noise at n=15.

Three corrections make these numbers what they are, all reported honestly. First,
earlier committed figures counted unverifiable answers (geometry and flow with
no derived numbers to check) as consistent; `recompute_honest.py` applies the
same unverifiable=1.0 rule the Labwright path already used, dropping the
reading-set self-consistent figures to 0 %/12 % (`flash`/`pro`). Second, a
prompt regression briefly dropped the *goal text* from the bare-family prompts
(a `+ goal` suffix was lost in a refactor), so memory-system runs from that
period asked the model "Goal: " with nothing after it; the model emitted the
same template chip for every goal. The regression is pinned by three targeted
tests (`../tests/test_benchmark_prompts.py`), and **every memory-system number
in this table is from a single post-fix re-run** at temperature 0.2. The
Labwright numbers are the committed run, preserved verbatim — Labwright's agent
always received the goal through a separate code path and was never affected.

Third, the 3D-spheroid competitor rows above were recomputed under a **fairness
fix** to the scorer (the three systems rerun, `results/eval_spheroid_*.json`).
The scorer previously extracted only *float* keys from memory-system output, so
string vessel formats (`spheroid_format` / `plate_format`: `"96-ula"`,
`"384-well ULA plate"`, …) were never recovered — every convention goal for
bare / soft-gate / self-verify was scored unverifiable → 1.0 *regardless of the
answer*, and spheroid geometry was checkable only when a vessel format was
present, so geometry goals like sphere-volume-from-diameter were penalized too.
The fix (a) recovers string keys, (b) recomputes each derived number from
exactly the raws it needs (geometry from cell count + diameter, vessel numbers
from a parseable format, growth from doubling × duration), and (c) *excludes*
reported-but-not-recomputable numbers instead of counting them wrong. Only the
three competitor systems were rerun — the Labwright records are carried from the
pre-fix run because Labwright scores typed `DesignPlan`s through a separate
path the fix does not touch. The previously committed 0 % / 1.000 cells were
this artifact; the Labwright cells never changed.

### Cross-provider: Kimi Code

The tables above are one backend (DeepSeek). To check the architecture
transfers, the same five sets × four systems were re-run against **`k3`** and
**`kimi-for-coding`** (Kimi Code, OpenAI-compatible coding endpoint) under the
identical harness — no re-typed numbers, all derived from the committed
per-entry records (`results/eval_{set}_{k3,kimicode}.json`). The `finetuned-ext`
row is a fixed local model and is identical under every backend, so it is not
repeated here. The five sets × two backends below are the complete sweep.

| model | set | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `k3` | 24-reading | bare-LLM | 8 % | 8 % | 0.917 |
| `k3` | 24-reading | soft-gate | 8 % | 8 % | 0.917 |
| `k3` | 24-reading | self-verify | 0 % | 0 % | 0.882 |
| `k3` | 24-reading | **Labwright** | **96 %** | **92 %** | **0.042** |
| `kimi-for-coding` | 24-reading | bare-LLM | 4 % | 4 % | 0.958 |
| `kimi-for-coding` | 24-reading | soft-gate | 8 % | 8 % | 0.917 |
| `kimi-for-coding` | 24-reading | self-verify | 0 % | 0 % | 0.944 |
| `kimi-for-coding` | 24-reading | **Labwright** | **4 %** | **0 %** | **0.958** |
| `kimi-for-coding` | 15-blind | bare-LLM | 0 % | 0 % | 1.000 |
| `kimi-for-coding` | 15-blind | soft-gate | 7 % | 0 % | 0.933 |
| `kimi-for-coding` | 15-blind | self-verify | 0 % | 0 % | 0.967 |
| `kimi-for-coding` | 15-blind | **Labwright** | **7 %** | **0 %** | **0.933** |
| `k3` | 15-blind | bare-LLM | 0 % | 0 % | 0.967 |
| `k3` | 15-blind | soft-gate | 7 % | 0 % | 0.900 |
| `k3` | 15-blind | self-verify | 0 % | 0 % | 0.878 |
| `k3` | 15-blind | **Labwright** | **100 %** | **47 %** | **0.000** |
| `kimi-for-coding` | 15-3D-spheroid | bare-LLM | 20 % | 20 % | 0.800 |
| `kimi-for-coding` | 15-3D-spheroid | soft-gate | 13 % | 13 % | 0.867 |
| `kimi-for-coding` | 15-3D-spheroid | self-verify | 7 % | 7 % | 0.600 |
| `kimi-for-coding` | 15-3D-spheroid | **Labwright** | **40 %** | **33 %** | **0.600** |
| `k3` | 15-3D-spheroid | bare-LLM | 13 % | 13 % | 0.867 |
| `k3` | 15-3D-spheroid | soft-gate | 7 % | 7 % | 0.933 |
| `k3` | 15-3D-spheroid | self-verify | 13 % | 13 % | 0.613 |
| `k3` | 15-3D-spheroid | **Labwright** | **93 %** | **73 %** | **0.067** |
| `kimi-for-coding` | 14-plate-culture | bare-LLM | 0 % | 0 % | 0.929 |
| `kimi-for-coding` | 14-plate-culture | soft-gate | 0 % | 0 % | 0.893 |
| `kimi-for-coding` | 14-plate-culture | self-verify | 0 % | 0 % | 0.881 |
| `kimi-for-coding` | 14-plate-culture | **Labwright** | **64 %** | **0 %** | **0.357** |
| `k3` | 14-plate-culture | bare-LLM | 7 % | 7 % | 0.750 |
| `k3` | 14-plate-culture | soft-gate | 7 % | 7 % | 0.750 |
| `k3` | 14-plate-culture | self-verify | 0 % | 0 % | 0.786 |
| `k3` | 14-plate-culture | **Labwright** | **93 %** | **93 %** | **0.071** |
| `kimi-for-coding` | 14-perfused-PK | bare-LLM | 36 % | 21 % | 0.643 |
| `kimi-for-coding` | 14-perfused-PK | soft-gate | 50 % | 36 % | 0.500 |
| `kimi-for-coding` | 14-perfused-PK | self-verify | 79 % | 29 % | 0.214 |
| `kimi-for-coding` | 14-perfused-PK | **Labwright** | **7 %** | **0 %** | **0.929** |
| `k3` | 14-perfused-PK | bare-LLM | 50 % | 29 % | 0.500 |
| `k3` | 14-perfused-PK | soft-gate | 50 % | 36 % | 0.500 |
| `k3` | 14-perfused-PK | self-verify | 79 % | 29 % | 0.214 |
| `k3` | 14-perfused-PK | **Labwright** | **100 %** | **86 %** | **0.000** |

**The gate transfers to any backend that can run the tool loop.** `k3` ≈
`flash`: Labwright takes it from 8 % bare to **92 % usable** on the reading set
(flash 88 %, pro 100 %). Its two reading misses (`lung-alveolar-shear` — never
called `submit_design`; `selfconsistent-channel-volume` — wrong target) are not
the three goals `flash` misses (`power-80-effect-half`, `reynolds-laminar-check`,
`selfconsistent-pressure-drop-40mm`) — k3 in fact hits all three, so no goal
type is a systematic blind spot; the misses are per-backend tool-loop
idiosyncrasies. The transfer is set-wide, not reading-only: on the other sets
k3 lands 47 % usable blind, 73 % spheroid, **93 % plate-culture** and 86 % PK,
with **0.000 hallucination** on blind and PK — the same pattern as the DeepSeek
backends. Its one non-zero-hallucination row on culture (0.071, one plan) is the
same plan the DeepSeek backends flag there; the culture set's `*_hallucination`
cells are explained in the Results section.

**…and collapses for a backend that cannot.** `kimi-for-coding` called
`submit_design` on 1/24 reading goals (and that design missed the target); the
other 23 never reached it. A traced goal fixated on `wall_shear_stress` with
`viscosity_pas=0`, replayed the same validation error (`input must be > 0`)
across all 12 iterations, and never corrected itself. It is even *better
without* the agent loop (soft-gate 8 % usable > Labwright 0 % on reading; the
same 0 % usable on the blind and culture sets, and 0 % on PK) — for a backend
that cannot self-correct a tool argument, Labwright's machinery is net negative.
That is an honest boundary condition on the architecture. Its one partial
success is the 15-spheroid set (33 % usable, up from 20 % bare): a design space
simple enough that the tool-loop defect is not exercised on every goal.

**Config caveat.** The Kimi runs used temperature **0.6** with thinking
disabled; the DeepSeek runs used **0.2** — the Kimi coding endpoint's plain
completion path validates temperature to 1.0, and the Labwright request shape
(thinking-disabled `extra_body`) accepts 0.6. A higher temperature cannot
explain `k3`'s high usable rate (it would if anything hurt a consistency-based
metric), and `kimi-for-coding`'s failure is argument fixation, not temperature
sensitivity.

### Statistical precision: single runs vs seed intervals

The headline cells above are **single runs** over 24/15/14 goals. Every set has
now been re-run over multiple seeds (Wilson 95 % CI via `eval/ci.py`), so the
point estimates have honest bounds: the 24-reading set over **5** seeds
(`results/eval_seed_benchmark.json`, 24 goals × 5 seeds = 120 trials per
system/model), and the blind / spheroid / culture / PK sets over **3** seeds
(`results/eval_seed_{blind,spheroid,culture,pk}.json`).

24-reading, 5 seeds:

| model | system | usable rate (k/n) | 95 % CI |
|---|---|---|---|
| `flash` | bare | 8/120 = 0.067 | [0.034, 0.126] |
| `flash` | soft-gate | 15/120 = 0.125 | [0.077, 0.196] |
| `flash` | self-verify | 0/120 = 0.000 | [0.000, 0.031] |
| `flash` | **Labwright** | 111/120 = 0.925 | [0.864, 0.960] |
| `pro` | bare | 13/120 = 0.108 | [0.064, 0.177] |
| `pro` | soft-gate | 19/120 = 0.158 | [0.104, 0.234] |
| `pro` | self-verify | 0/120 = 0.000 | [0.000, 0.031] |
| `pro` | **Labwright** | 115/120 = 0.958 | [0.906, 0.982] |

The four other sets, 3 seeds each (pooled usable rate, Wilson 95 % CI):

| set | model | bare | soft-gate | self-verify | **Labwright** |
|---|---|---|---|---|---|
| 15-blind | `flash` | 0 % [0.000, 0.079] | 0 % [0.000, 0.079] | 0 % [0.000, 0.079] | **44 % [0.309, 0.588]** |
| 15-blind | `pro` | 0 % [0.000, 0.079] | 2 % [0.004, 0.116] | 0 % [0.000, 0.079] | **49 % [0.350, 0.630]** |
| 15-3D-spheroid | `flash` | 20 % [0.109, 0.338] | 13 % [0.063, 0.262] | 13 % [0.063, 0.262] | **93 % [0.821, 0.977]** |
| 15-3D-spheroid | `pro` | 29 % [0.177, 0.434] | 27 % [0.160, 0.410] | 22 % [0.125, 0.363] | **96 % [0.852, 0.988]** |
| 14-plate-culture | `flash` | 0 % [0.000, 0.084] | 0 % [0.000, 0.084] | 0 % [0.000, 0.084] | **90 % [0.779, 0.962]** |
| 14-plate-culture | `pro` | 7 % [0.025, 0.190] | 7 % [0.025, 0.190] | 0 % [0.000, 0.084] | **79 % [0.641, 0.883]** |
| 14-perfused-PK | `flash` | 31 % [0.191, 0.460] | 45 % [0.312, 0.601] | 33 % [0.210, 0.484] | **81 % [0.667, 0.900]** |
| 14-perfused-PK | `pro` | 29 % [0.172, 0.436] | 38 % [0.250, 0.532] | 33 % [0.210, 0.484] | **76 % [0.615, 0.865]** |

The qualitative ordering (Labwright ≫ memory systems; flash vs pro within ~5 %)
is stable across seeds. The Labwright interval and the memory-system interval
never overlap on any set, so the headline gap is not a sampling artifact; on the
hardest set (blind) the Labwright point estimate itself has a wide interval
(44–49 %, n = 45 trials), which is honest about how much headroom remains. The
thinking-ablation cells below are single-run and should be read as such — a
single point is a pilot, not a precision claim.

### Ablation: thinking on vs off

Thinking is normally disabled (`LABWRIGHT_DISABLE_THINKING=0` re-enables it).
The 2 × 2 grid below isolates whether the blind-set misses are a
reasoning-budget gap. Labwright self-consistency / usable / hallucination:

| model | set | thinking | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `flash` | 24-reading | off | 88 % | 88 % | 0.125 |
| `flash` | 24-reading | on | 100 % | 100 % | 0.000 |
| `flash` | 12-blind* | off | 100 % | 25 % | 0.000 |
| `flash` | 12-blind* | on | 100 % | 17 % | 0.000 |
| `pro` | 24-reading | off | 100 % | 100 % | 0.000 |
| `pro` | 24-reading | on | 100 % | 100 % | 0.000 |
| `pro` | 12-blind* | off | 100 % | 33 % | 0.000 |
| `pro` | 12-blind* | on | 100 % | 42 % | 0.000 |

\* The thinking-ablation grid was run on the **12-goal** blind set, before the
scenario expansion grew it to 15; the thinking rows were not repeated on the
expanded set, so they are historical and are **not** directly comparable to the
15-entry cells above.

The blind misses persist with thinking on (17 % / 42 % vs 25 % / 33 % off —
within a goal of each other): thinking neither recovers targets the model does
not know (`flash` dropped a prompt-backed hit, BBB; `pro` picked up one cold
goal, arterial) nor breaks the gate's 100 % self-consistency. The misses are a
domain-knowledge gap, not an effort one. The one row where thinking *helps* is
the `flash` 24-reading set (88 % → 100 % usable): the three silent
non-completions of the thinking-off run each submitted a verified design under
thinking — effort recovers goals the answer is handed over, not physiology the
model does not know.

### Ablation: the same calculators, with the verifier switched off

The benchmark's "0.000 hallucination" invites a circularity objection: *the
calculators derive the numbers and the verifier recomputes them from the same
calculators, so of course they match.* The objection is answered empirically
with a fifth system, **`tool_no_gate`**: the exact same model runs the exact
same ReAct loop with the exact same calculator tools, but the verification layer
is switched off — `submit_design` never runs the verifier, never reports
`review_required`, and the system prompt drops every promise the verifier backs
("never invent a computed number", "derived fields are computed and verified for
you", the fix-and-resubmit loop). The calculators are untouched: the only
removed component is verification. The no-gate plans are then scored post-hoc
with the *identical* rules as `labwright`, so a plan the verifier would have
rejected still reads as a failed submission.

If the verifier were circular — a no-op echo of the calculators — removing it
would change nothing. The measured difference is small on every set, and it
does *not* point in one direction:

| set | model | labwright usable | tool_no_gate usable | entries labwright-only valid | entries no-gate-only valid |
|---|---|---|---|---|---|
| 24-reading | `flash` | 23/24 = 96 % | 22/24 = 92 % | 1 | 0 |
| 24-reading | `pro` | 23/24 = 96 % | 24/24 = 100 % | 0 | 1 |
| 15-blind | `flash` | 6/15 = 40 % | 7/15 = 47 % | 1 | 2 |
| 15-3D-spheroid | `flash` | 13/15 = 87 % | 12/15 = 80 % | 1 | 0 |
| 14-plate-culture | `flash` | 9/14 = 64 % | 10/14 = 71 % | 2 | 3 |
| 14-perfused-PK | `flash` | 11/14 = 79 % | 12/14 = 86 % | 1 | 2 |
| **all six runs** | | **85/106 = 80 %** | **87/106 = 82 %** | **6** | **8** |

Read the table carefully: across 106 submissions the no-gate agent is *slightly
ahead* (87 vs 85 usable). Every one of the 14 divergent entries — six where
`labwright` is valid and no-gate is not, eight the other way — failed for the
same reason on both sides: `wrong_target` (the model anchored on a different
output quantity than the goal asked for; `spheroid-um-mm-unit-ambiguity`,
`selfconsistent-channel-volume`, `blind-seed-hepg2-log`, …). Not one divergent
entry is a hallucination or a unit error. The honest conclusion is **not** "the
verifier rescues usable rate" — removing it changed usable rate by −1 to +1
goals per set (sets of 14–24), i.e. within noise, and it is **not** a
hallucination
shield the way the headline 0.000 might suggest. The calculators carry the load.
The verifier's measurable added value sits elsewhere:

- **When the verifier fires, it is correct — but it fires rarely.** Across the
  same six runs the no-gate agent produced exactly **one** hallucinated derived
  number that the gate-backed agent did not (`spheroid-volume-from-diameter`,
  hallucination 1.0 — the no-gate agent silently substituted a number instead of
  computing it; the gate-backed agent submitted no such plan). The verifier is
  also what makes that 0.000 measurable at all — the no-gate agent's own
  `submit_design` reports `ok` on every plan (unit test pinned in
  `tests/test_no_gate_ablation.py`).
- **The verifier's runtime role is consistency, not domain knowledge.** It
  recomputes claimed numbers from the plan's own inputs, checks units, and
  applies sanity bands; it cannot supply a physiological value the model never
  had. That is why the 14 divergent entries are all `wrong_target`: the drift is
  the model picking the wrong quantity, which no self-consistency check can
  repair. This is the boundary the paper draws — verification guarantees
  *internal* correctness, the goal set's `source` (real literature or an explicit
  "design target") guarantees the *external* target.
- **The one-directional discipline case is real but narrow.** The verifier's
  presence in the prompt is a promise with teeth: on `selfconsistent-channel-volume`
  (reading) the gate-backed agent computes the channel volume, the no-gate agent
  drifts off-target. Single-goal, and it is exactly the "keep output inside what
  you can verify" behaviour the paper claims — but the net effect on the
  six-run aggregate is not measurable above noise.

*Honesty notes:* the no-gate agent's prompt necessarily differs from Labwright's
— you cannot remove the verifier's enforcement while leaving its promises in the
prompt. The ablation therefore removes the verification *layer* (prompt
discipline + gate + scoring requirement), keeping calculators, loop and model
fixed; the residual gap is what that layer adds. `tool_no_gate` is run through
the same `evaluate()` driver and scored by the same `_score_design`; results in
`results/eval_nogate_*.json`. The reading/pro rows are single runs; the
blind/spheroid/culture/pk numbers are single-run flash (Wilson-95 % pooled
3-seed CIs for `labwright` on those sets: blind 44 % [0.31, 0.59], spheroid
93 % [0.82, 0.98], culture 90 % [0.78, 0.96], pk 81 % [0.67, 0.90]).

### Agent attempt: an iterating fix-and-resubmit agent (`labwright_iter`)

The paper's "agent" contribution is not the single-shot system but an *iterating*
one: when the verifier returns `review_required`, the agent reads the
verification report, fixes **only** the flagged raw inputs, and resubmits, up to
`max_submission_attempts = 3`. The first-submit `labwright` treats the first
`review_required` as terminal and is scored on the dirty plan; `labwright_iter`
keeps the same honest verdict but gets to repair it. Both run the same
calculators, the same prompt, the same model (paired, same run).

| set | model | labwright usable | labwright_iter usable | verifier fired | recovered to ok | exhausted budget |
|---|---|---|---|---|---|---|
| 15-blind | `flash` | 6/15 = 40 % | 6/15 = 40 % | 15 | 15 | 0 |
| 15-3D-spheroid | `flash` | 12/15 = 80 % | 12/15 = 80 % | 13 | 13 | 0 |
| 14-plate-culture | `flash` | 14/14 = 100 % | 14/14 = 100 % | 10 | 10 | 0 |
| 14-perfused-PK | `flash` | 11/14 = 79 % | 11/14 = 79 % | 3 | 3 | 0 |
| **all four sets** | | **43/58 = 74 %** | **43/58 = 74 %** | **41** | **41** | **0** |

The blind row is the sharpest. The verifier fired on **all 15** entries — every
plan needed at least one correction — and the iter loop recovered **all 15** to
verifier-clean (`self-consistent` 100 %, hallucination 0.000), with a mean 1.87
fix rounds and not one entry exhausting its budget. Yet usable rate is identical
to first-submit (6/15 = 40 %). The reason is the failure taxonomy: the 9
unusable entries fail with `wrong_target` — the model selected a different
physiological value than the goal asked for (e.g. `blind-bbb-shear` wants the
blood–brain-barrier shear; the model reports a generic microfluidic value). The
verifier's `review_required` catches *internal* inconsistency — a claimed number
that does not follow from the plan's own inputs — and the iter loop repairs
exactly that; it cannot supply a *physiological* value the model never had. The
one entry rescued (`blind-arterial-shear`) was a fixable raw-input slip; the one
entry lost (`blind-phh-seed`) drifted off-target *during* a fix round. Iteration
is a correctness loop, not a domain-knowledge loop.

The spheroid row shows the same pattern at higher absolute rate: the iter loop
rescued 2 entries the first-submit agent lost (`spheroid-384ula-medium`,
`spheroid-um-mm-unit-ambiguity`) and lost 2 it would have kept
(`spheroid-volume-from-diameter`, where a fix round degraded a valid plan into a
hallucinated one; `blind-spheroid-384ula-medium`). On the culture and PK sets
there is zero divergence either way and every verifier firing is repaired. The
aggregate is the point: across 58 submissions the verifier fired 41 times and
the iter loop repaired **all 41** to verifier-clean, exhausting its budget on
none, yet usable rate is **exactly equal** (43/58 for both) because the
entries that stay unusable fail on `wrong_target` — a physiological value the
model never had — which no self-consistency round can supply. (The per-run
`labwright` rate differs across the ablation and iter runs on culture — 9/14
vs 14/14 — which is expected sampling variance; each comparison is a *paired*
run of both systems.)

This is the honest division of labour the paper claims: the calculators own the
arithmetic, the verifier owns internal consistency, and the goal set's `source`
annotation (real literature or an explicit "design target") owns the external
target. `labwright_iter` demonstrates that when an error *is* verifiable, the
agent repairs it autonomously — 41/41 across the four sets, 0 exhausted — while
errors that are not verifiable (missing domain knowledge) remain visible in the
failure taxonomy instead of being silently hallucinated. The iterating agent is
registered as a distinct system (`labwright_iter`); results in
`results/eval_iter_*.json`, analysed by `eval/analyze_iter.py`.

### Benchmarking scope: why these systems, and not the named ones

A reviewer will ask why the four systems above (bare, soft-gate, self-verify,
Labwright) are benchmarked and not the published systems named in the manuscript
(§6.1). Each named system was **investigated on the ground**, not assumed:

- **MMFT OoC Designer** (IEEE TCAD 2024) — cloned, installed, ran. Its input
  JSON already *contains the target shear* it is asked to realize (the
  committed run sizes a three-organ network to a 1.5 Pa target and exports the
  channel network, a 2D layout and an STL). It is a deterministic *geometry
  synthesizer*: given the answer it
  produces geometry. On the blind set its input cannot even be constructed; on
  the reading set it is handed the answer key. It has no LLM, no natural-language
  interface, and no cell-biology/dosing/statistics layer, so it is not directly
  comparable on this benchmark.
- **BPL-COGEN** (bioRxiv 2026) — the released pipeline couples a compiler with a
  **30B**-parameter model (~60 GB of GPU memory in BF16); the single available
  GPU here is 32 GB, so the comparison is *physically infeasible*, not declined.
  It compiles protocols into a formal language (type-safety), not a
  goal-to-design generator.
- **Thoth** (ICLR 2026) — an 8B LLM with public weights (`manglu3935/Thoth`,
  cc-by-4.0) trained to generate protocol text with a structured reward. Its
  native output is protocol *prose* (structured `<think>`/`<key>`/`<orc>`
  sections), not design JSON. We tried running it through the identical harness
  as a prompt-only comparison; forcing prose through a design-JSON schema makes
  the result a *harness-adaptation artifact* — it emits nothing checkable for
  format reasons, not capability ones — so we do **not** report a head-to-head
  row for it. Functionally it is the same boundary as the other named systems:
  its verification is a learned, model-internal reward, so it cannot prove a
  number follows from its own inputs.

## Run

```bash
python -m eval.run_benchmark                                  # 24-reading, flash
python -m eval.run_benchmark --gold eval/gold_blind.json --out results/eval_blind_flash.json
python -m eval.run_benchmark --limit 3 --out /tmp/eval.json
python -m eval.run_benchmark --model deepseek-v4-pro --out results/eval_pro.json
python -m eval.recompute_honest results/eval_*.json          # after any run
python -m eval.report results/eval_flash.json                # render the comparison
```

Requires `LABWRIGHT_API_KEY`/`DEEPSEEK_API_KEY`. Results are written to
`results/` and committed so the benchmark is reproducible from the repo alone.
