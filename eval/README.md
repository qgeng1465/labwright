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

The three LLM-memory systems (bare, soft-gate, self-verify) are scored by
**exactly the same** rules — extraction, ±5 % consistency tolerance,
verifiability, unverifiable=1.0. Only the prompt/stage structure differs, so any
measured difference between them is caused by the *approach*, not by scoring.

Three gold sets:

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

### Prompts & models (verbatim)

The ablation changes *only* prompt/stage structure on fixed models, so both are
pinned. All rows use the DeepSeek v4 API (`https://api.deepseek.com`,
OpenAI-compatible): **`deepseek-v4-flash`** and **`deepseek-v4-pro`**, at
temperature **0.2**, thinking **disabled** (`LLMClient(disable_thinking=True)`
default); Labwright's agent runs the same client at 0.2 with a 12-iteration
tool budget. These are API models — no weight pin exists; `generated_at` in each
result JSON records the run date. `LABWRIGHT_MODEL` / `LABWRIGHT_BASE_URL`
override, but the committed numbers are exactly these two models.

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

- **hallucination rate** — fraction of a system's derived numbers the verifier
  rejects. A system that produces no checkable output scores 1.0.
  - Labwright: derived numbers come from the calculators and the verifier
    recomputes them from the *same* calculators, so for any plan that submits,
    this is **0 by construction**. That is the point: it is an architectural
    guarantee, not a measured win — the metric exists to *make* the guarantee
    checkable. The only way it is non-zero is a `plan: false` run (the agent
    produced no design), scored 1.0.
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
  which target to aim at.
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

All systems, all three sets, two models (`flash`, `pro`). Self-consistent = zero
verifier errors; usable = self-consistent *and* recovers every gold target
within ±5 %. The memory systems (bare, soft-gate, self-verify) are scored by
identical extraction/tolerance/unverifiable=1.0 rules — only the prompt/stage
structure differs.

| model | set | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `flash` | 24-reading | bare-LLM | 0 % | 0 % | 1.000 |
| `flash` | 24-reading | soft-gate | 12 % | 12 % | 0.875 |
| `flash` | 24-reading | self-verify | 0 % | 0 % | 0.792 |
| `flash` | 24-reading | **Labwright** | **88 %** | **88 %** | **0.125** |
| `pro` | 24-reading | bare-LLM | 12 % | 12 % | 0.875 |
| `pro` | 24-reading | soft-gate | 8 % | 8 % | 0.917 |
| `pro` | 24-reading | self-verify | 0 % | 0 % | 0.750 |
| `pro` | 24-reading | **Labwright** | **100 %** | **100 %** | **0.000** |
| `flash` | 15-blind | bare-LLM | 7 % | 0 % | 0.933 |
| `flash` | 15-blind | soft-gate | 13 % | 0 % | 0.867 |
| `flash` | 15-blind | self-verify | 0 % | 0 % | 0.611 |
| `flash` | 15-blind | **Labwright** | **100 %** | **40 %** | **0.000** |
| `pro` | 15-blind | bare-LLM | 7 % | 0 % | 0.933 |
| `pro` | 15-blind | soft-gate | 13 % | 0 % | 0.867 |
| `pro` | 15-blind | self-verify | 0 % | 0 % | 0.733 |
| `pro` | 15-blind | **Labwright** | **100 %** | **47 %** | **0.000** |
| `flash` | 15-3D-spheroid | bare-LLM | 0 % | 0 % | 1.000 |
| `flash` | 15-3D-spheroid | soft-gate | 0 % | 0 % | 1.000 |
| `flash` | 15-3D-spheroid | self-verify | 0 % | 0 % | 1.000 |
| `flash` | 15-3D-spheroid | **Labwright** | **93 %** | **87 %** | **0.011** |
| `pro` | 15-3D-spheroid | bare-LLM | 0 % | 0 % | 1.000 |
| `pro` | 15-3D-spheroid | soft-gate | 0 % | 0 % | 1.000 |
| `pro` | 15-3D-spheroid | self-verify | 0 % | 0 % | 1.000 |
| `pro` | 15-3D-spheroid | **Labwright** | **93 %** | **87 %** | **0.067** |

The memory systems never produce a usable *design* on either set, and the two
naive "fixes" do not help. The only usable memory-system entries anywhere are
the three single-arithmetic-step goals on the 24-reading set — a residence time
with geometry given, a channel volume, a mean velocity — where the goal supplies
every input and no design choice remains (`pro` bare and `flash` soft-gate both
reach 12 % there, `pro` soft-gate 8 %). On every goal that requires choosing
geometry and flow to hit a target, all memory systems score 0 % usable, both
sets, both models. Soft-gate (a "re-check yourself" prompt) occasionally
completes one of those three single-step goals but never rescues a design;
self-verify (using the LLM as its own verifier) collapses to **0 %**
everywhere: handed its own raw inputs, the model recomputes them wrong, so the
second pass actively corrupts the proposal. Only Labwright's deterministic
calculators + verifier reach usable > 0 % on design goals.

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
`silence` and `calculation_error` are architecturally excluded from the
Labwright path (a plan either submits verified or fails to submit). The memory
systems fail almost entirely with `calculation_error` (14/15 for bare on both
models). **Target-selection accuracy** (headline target within ±5 %) is 40 %
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
generalize — both recover the cold 384-ULA 50 µL and hanging-drop 20 µL at
**100 %** (2/2 each) precisely because those values live in the calculator's
tool registry, not in model memory: the bare model scores 0 % on the same two
entries (its answers are `calculation_error`, 0 % verifiable — the format-driven
medium volume never appears). This is the paper's "the calculator is the
knowledge base" claim made literal: a convention encoded once in `SPHEROID_FORMATS`
is available to both models on every run, cold or not. The remaining failures
are the honest residue: `flash` mis-targets the doxorubicin dosing goal (proposes
24 spheroids against the gold 96 — internally consistent, `hall 0.000`, so the
gate held and the design was rejected) and fails the blind hepatocyte-formation
goal by inconsistency (`hall 0.167`: correct 1000 cells/spheroid but a derived
diameter that does not follow from its own cell size); `pro` proposes 54
spheroids on the same dosing goal and returns *silence* on the pure-geometry
`spheroid-volume-from-diameter` goal — a one-line sphere-volume derivation, yet
no design was submitted at all. Target-selection accuracy is 93 % (`flash`
Labwright) and 87 % (`pro` Labwright, equal across systems) — the model names
the right target, but only the calculator path turns that into a verified,
usable design. As with the blind set, these are single-run point estimates; the
differences between the two models' two failures each are noise at n=15.

Two corrections make these numbers what they are, both reported honestly. First,
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

### Statistical precision: single runs vs 5-seed intervals

The headline cells above are **single runs** over 24/15 goals. A 5-seed re-run
of the 24-reading set (`results/eval_seed_benchmark.json`: 24 goals × 5 seeds =
120 trials per system/model, Wilson 95 % CI via `eval/ci.py`) bounds the
point estimates:

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

The qualitative ordering (Labwright ≫ memory systems; flash vs pro within ~5 %)
is stable across seeds; the blind-set cells and the thinking-ablation cells are
single-run point estimates and should be read as such — a single point is a
pilot, not a precision claim.

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
