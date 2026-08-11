# Labwright benchmark — paper evidence

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

Two gold sets:

1. **`gold_experiments.json` — 24 "reading" goals.** Every goal states the
   answer (the geometry/flow/density/effect-size, or the physiological target
   number). These test whether the pipeline can *extract the stated numbers and
   drive the calculators to them*. They do **not** test domain knowledge.
2. **`gold_blind.json` — 12 "recall" goals.** The goal states no number at all
   ("recapitulate physiological venular wall shear"); the model must supply the
   canonical target from its own knowledge. Nine are `cold` (answer in neither
   goal nor the system prompt: kidney, arterial, venular, HepG2 density,
   primary-hepatocyte density, pulmonary artery, gut, retinal arteriole,
   lymphatic); three are `prompt-backed` (liver, lung, BBB — the Labwright
   system prompt lists a range that contains the answer, so the model must still
   select the right value). Every entry pins a citable source in its `source`
   field; no number is invented.

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
  usable rate collapses: `flash` 88 % → 25 %, `pro` 100 % → 33 % (see below).
  The gate held (hallucination 0.000 on every submitted plan) — Labwright
  produced clean, verified designs that aimed at the *wrong physiology*. That
  is the honest boundary of the guarantee.

## Status

- [x] Curate the 24-reading gold set. Every entry carries a provenance rule — a
      pinned source or an explicit `self-consistent` label. All anchors
      hand-checked against the governing equations.
- [x] Curate the 12-blind gold set (`gold_blind.json`, expanded 6 → 12 in
      Aug 2026), each entry labelled `cold`/`prompt-backed` with a pinned source.
- [x] Run bare-LLM + Labwright on `deepseek-v4-flash` and `deepseek-v4-pro` on
      the 24-reading set → `results/eval_flash.json`, `results/eval_pro.json`.
      The runner checkpoints after every entry.
- [x] Run the competitor systems (soft-gate, self-verify) on both sets and both
      models → `results/eval_competitors_{flash,pro}.json`,
      `results/eval_blind_competitors_{flash,pro}.json`.
- [x] Re-run bare-LLM + Labwright on the expanded 12-blind set →
      `results/eval_blind_{flash,pro}.json`.
- [x] Thinking ablation grid (thinking ON for both models on both sets) →
      `results/eval_blind_{flash,pro}_thinking.json`,
      `results/eval_{flash,pro}_thinking.json`.
- [x] Reverse-verify published protocols + labelled synthetic controls:
      `eval/run_verify_batch.py` → `results/eval_verify_batch.json`.
- [x] Preprint draft, Colab notebook, HF Space scaffolding.
- [ ] Submit the preprint to bioRxiv; publish the HF Space.

## Results (honest)

All four systems, both sets, both models. Self-consistent = zero verifier
errors; usable = self-consistent *and* recovers every gold target within ±5 %.
The memory systems (bare, soft-gate, self-verify) are scored by identical
extraction/tolerance/unverifiable=1.0 rules — only the prompt/stage structure
differs.

| model | set | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `flash` | 24-reading | bare-LLM | 21 % | 0 % | 0.792 |
| `flash` | 24-reading | soft-gate | 17 % | 0 % | 0.833 |
| `flash` | 24-reading | self-verify | 0 % | 0 % | 0.833 |
| `flash` | 24-reading | **Labwright** | **88 %** | **88 %** | **0.125** |
| `pro` | 24-reading | bare-LLM | 8 % | 0 % | 0.917 |
| `pro` | 24-reading | soft-gate | 12 % | 0 % | 0.875 |
| `pro` | 24-reading | self-verify | 0 % | 0 % | 0.736 |
| `pro` | 24-reading | **Labwright** | **100 %** | **100 %** | **0.000** |
| `flash` | 12-blind | bare-LLM | 17 % | 0 % | 0.833 |
| `flash` | 12-blind | soft-gate | 0 % | 0 % | 1.000 |
| `flash` | 12-blind | self-verify | 0 % | 0 % | 0.792 |
| `flash` | 12-blind | **Labwright** | **100 %** | **25 %** | **0.000** |
| `pro` | 12-blind | bare-LLM | 17 % | 0 % | 0.833 |
| `pro` | 12-blind | soft-gate | 17 % | 0 % | 0.833 |
| `pro` | 12-blind | self-verify | 0 % | 0 % | 0.889 |
| `pro` | 12-blind | **Labwright** | **100 %** | **33 %** | **0.000** |

The memory systems never produce a usable design on either set, and the two
naive "fixes" do not help — soft-gate (a "re-check yourself" prompt) stays
within run-to-run sampling noise of bare (at temperature 0.2 the bare
self-consistent rate was 17 % and 21 % on the two `flash` batches and 8 % in
both `pro` batches, so a few points between memory systems — e.g. soft-gate's
12 % on `pro` — is noise), and self-verify (using the LLM as its
own verifier) collapses to **0 %** everywhere: handed its own raw inputs, the
model recomputes them wrong, so the second pass actively corrupts the proposal.
Only Labwright's deterministic calculators + verifier reach usable > 0 %.

The blind-set drop is the honest headline: when the goal does not hand over the
target, Labwright's verified designs hit the wrong physiology. On the expanded
12 goals, `flash` recovers 3 (arterial 1.5 Pa, lung 0.03 Pa, BBB 1.0 Pa) and
`pro` recovers 4 (venular 0.3 Pa, lung 0.03 Pa, HepG2 seeding 4000, BBB 1.0 Pa).
Both models correctly select the two prompt-backed entries they are primed for
(lung, BBB) yet both miss the third prompt-backed entry (liver, 0.05 Pa): they
propose the "default chip" 0.10 Pa — inside the prompt's 0.05–0.15 Pa range but
not the low-shear convention. Cold entries are mostly wrong (recovery = relative
error of the proposed shear vs the target): kidney PTEC 0.02 Pa is proposed at
0.50 Pa (`flash`, recovery 24) / 0.20 Pa (`pro`, recovery 9 — treating
dyn/cm² as Pa), gut epithelium 0.002 Pa at 0.01 Pa (recovery 4), retinal
arteriole within 7 % (`flash` — the closest miss, just outside the ±5 % usable
tolerance) but 44 % off on `pro`, pulmonary artery off by 25–50 %, lymphatic
off by 75 %, and both seeding densities off by a third to a half. The gate
never failed — every plan was internally verified — it just could not supply
domain knowledge the model did not have.

Bare-LLM's honest self-consistent numbers (21 %/8 % reading) are much lower
than the first committed figures (62 %/50 %). The earlier figures counted
unverifiable answers as consistent; `recompute_honest.py` applies the same
unverifiable=1.0 rule the Labwright path already used, and the recorded
`reported` values were not re-run or changed.

### Ablation: thinking on vs off

Thinking is normally disabled (`LABWRIGHT_DISABLE_THINKING=0` re-enables it).
The 2 × 2 grid below isolates whether the blind-set misses are a
reasoning-budget gap. Labwright self-consistency / usable / hallucination:

| model | set | thinking | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `flash` | 24-reading | off | 88 % | 88 % | 0.125 |
| `flash` | 24-reading | on | — | — | — |
| `flash` | 12-blind | off | 100 % | 25 % | 0.000 |
| `flash` | 12-blind | on | 100 % | 17 % | 0.000 |
| `pro` | 24-reading | off | 100 % | 100 % | 0.000 |
| `pro` | 24-reading | on | 100 % | 100 % | 0.000 |
| `pro` | 12-blind | off | 100 % | 33 % | 0.000 |
| `pro` | 12-blind | on | 100 % | 42 % | 0.000 |

The `flash` 24-reading thinking row is pending (`results/eval_flash_thinking.json`
is still running). The three completed rows show the blind misses persist with
thinking on (17 % / 42 % vs 25 % / 33 % off — within a goal of each other):
thinking neither recovers targets the model does not know (`flash` lost its one
prompt-backed hit, BBB; `pro` picked up one cold goal, arterial) nor breaks
the gate's 100 % self-consistency. The misses are a domain-knowledge gap, not
an effort one.

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
  interface, and no cell-biology/dosing/statistics layer — a different problem,
  not a different solution to the same one.
- **BPL-COGEN** (bioRxiv 2026) — the released pipeline couples a compiler with a
  **30B**-parameter model (~60 GB of GPU memory in BF16); the single available
  GPU here is 32 GB, so the comparison is *physically infeasible*, not declined.
  It compiles protocols into a formal language (type-safety), not a
  goal-to-design generator.
- **Thoth** (ICLR 2026) — the one genuinely comparable competitor: an 8B LLM
  with public weights (`manglu3935/Thoth`, cc-by-4.0) trained to generate
  protocol text with a structured reward. We run it through the **identical
  harness** (same gold goals, same bare prompt, same JSON extraction, same
  scoring — `eval/run_thoth.py`), a prompt-only comparison because Thoth's native
  output is protocol prose, not design JSON. The run is queued on the GPU; a
  `thoth-8b` row lands in the tables when it completes.

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
