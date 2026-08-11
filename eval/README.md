# Labwright benchmark — paper evidence

## Question

Do LLM-written wet-lab designs contain hallucinated numbers, and does
constraining the LLM to *propose* while *calculators compute* fix it?

## Protocol

On every gold-standard experiment, run two systems:

| system | behavior |
|---|---|
| bare-LLM | the model is given the full design JSON schema and asked to write the complete design *including derived numbers*, from memory |
| Labwright | the model proposes raw inputs; derived numbers come from `labwright.calc` and pass `labwright.verify` |

Two gold sets:

1. **`gold_experiments.json` — 24 "reading" goals.** Every goal states the
   answer (the geometry/flow/density/effect-size, or the physiological target
   number). These test whether the pipeline can *extract the stated numbers and
   drive the calculators to them*. They do **not** test domain knowledge.
2. **`gold_blind.json` — 6 "recall" goals.** The goal states no number at all
   ("recapitulate physiological venular wall shear"); the model must supply the
   canonical target from its own knowledge. Four are `cold` (answer in neither
   goal nor the system prompt: kidney, arterial, venular, HepG2 density); two
   are `prompt-backed` (liver, lung — the Labwright system prompt lists a range
   that contains the answer, so the model must still select the right value).

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
  usable rate collapses: `flash` 88 % → 33 %, `pro` 100 % → 17 % (see below).
  The gate held (hallucination 0.000 on every submitted plan) — Labwright
  produced clean, verified designs that aimed at the *wrong physiology*. That
  is the honest boundary of the guarantee.

## Status

- [x] Curate the 24-reading gold set. Every entry carries a provenance rule — a
      pinned source or an explicit `self-consistent` label. All anchors
      hand-checked against the governing equations.
- [x] Curate the 6-blind gold set (`gold_blind.json`), labelled
      `cold`/`prompt-backed`.
- [x] Run both systems on `deepseek-v4-flash` and `deepseek-v4-pro` on the
      24-reading set → `results/eval_flash.json`, `results/eval_pro.json`.
      `python -m eval.report results/eval_flash.json`. The runner checkpoints
      after every entry.
- [x] Recompute bare metrics with the unverifiable=1.0 rule:
      `python -m eval.recompute_honest results/eval_flash.json results/eval_pro.json`.
- [x] Run both systems on the 6-blind set → `results/eval_blind_flash.json`,
      `results/eval_blind_pro.json`.
- [x] Reverse-verify published protocols + labelled synthetic controls:
      `eval/run_verify_batch.py` → `results/eval_verify_batch.json`.
- [x] Ablation: thinking on/off on the blind set
      (`results/eval_blind_flash_thinking.json`; see "Ablation" below).
- [x] Preprint draft, Colab notebook, HF Space scaffolding.
- [ ] Submit the preprint to bioRxiv; publish the HF Space.

## Results (honest)

| model | set | system | self-consistent | usable | hallucination |
|---|---|---|---|---|---|
| `flash` | 24-reading | bare-LLM | 21 % | 0 % | 0.792 |
| `flash` | 24-reading | **Labwright** | **88 %** | **88 %** | **0.125** |
| `pro` | 24-reading | bare-LLM | 8 % | 0 % | 0.917 |
| `pro` | 24-reading | **Labwright** | **100 %** | **100 %** | **0.000** |
| `flash` | 6-blind | bare-LLM | 33 % | 0 % | 0.667 |
| `flash` | 6-blind | **Labwright** | **100 %** | **33 %** | **0.000** |
| `pro` | 6-blind | bare-LLM | 33 % | 0 % | 0.667 |
| `pro` | 6-blind | **Labwright** | **100 %** | **17 %** | **0.000** |

The blind-set drop is the honest headline: when the goal does not hand over the
target, Labwright's verified designs hit the wrong physiology. `flash` proposed
kidney PTEC shear at 0.50 Pa (target 0.02 — 25× off) and `pro` at 0.20 Pa
(target 0.02 — 10× off, treating dyn/cm² as Pa); both proposed hepatic 0.10 Pa
(2× the 0.05 low-shear convention) and venular 0.40 Pa (in the 0.1–0.6 Pa range
but off the 0.3 anchor). The gate never failed — every plan was internally
verified — it just could not supply domain knowledge the model did not have.

Bare-LLM's honest self-consistent numbers (21 %/8 %) are much lower than the
first committed figures (62 %/50 %). The earlier figures counted unverifiable
answers as consistent; `recompute_honest.py` applies the same
unverifiable=1.0 rule the Labwright path already used, and the recorded
`reported` values were not re-run or changed.

### Ablation: thinking on vs off

`deepseek-v4-flash` normally runs with thinking disabled (the arithmetic lives
in the calculators, not the model). Re-running the 6-blind set with thinking
enabled (`LABWRIGHT_DISABLE_THINKING=0`) changes nothing: **33 % usable either
way** — the same two entries correct (arterial 1.5 Pa, lung 0.03 Pa), the same
four wrong (liver 0.10 Pa, kidney 0.10 Pa, venular 0.40 Pa, seed 8000). With
thinking on the model converged harder on the "default chip" (1000×100 µm @
10 µL/min → 0.1 Pa), fixing kidney from 0.5 → 0.10 Pa but still 5× off. The
blind-set failures are a *domain-knowledge* gap — the model does not know the
canonical targets — not a reasoning-budget one, and no amount of thinking
recovers them. That is the honest headline for a tool whose entire point is
the hard gate: **the gate holds regardless of model effort; it cannot supply
knowledge the model does not have.**

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
