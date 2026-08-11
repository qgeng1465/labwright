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

Metrics (from `benchmark.evaluate`):

- **parameter recovery** — mean relative error of the design vs the gold
  standard on `shear_pa`, `reynolds`, `pressure_drop_pa`, `residence_time_s`,
  `channel_volume_ul`, `mean_velocity_mms`, `flow_rate_uLmin`, `seed_count`,
  `dmso_fraction_vv`, `n_per_group`.
- **hallucination rate** — fraction of derived fields that the verifier rejects
  on a design that *parses*. A system that produces no usable JSON at all scores
  1.0 (an unusable output is fully hallucinated).
- **self-consistent / usable rate** — see *Metrics, carefully* below; these are
  the headline "can it be used?" numbers.

Expected result to publish: bare-LLM hallucination rate substantially above
zero (reported figures for unconstrained biomedical LLMs are routinely
30–60% on numerical protocol details); Labwright rate ≈ 0 by construction.
Recovery accuracy should also improve because the model is told the actual
tool outputs mid-loop. Note the *parroting* subtlety: a bare LLM can "name" a
target shear from the goal text while its own geometry/flow imply something
else — recovery then looks perfect while hallucination is high. That is
precisely why the two metrics are reported together.

## Provenance rules (hard requirements)

- **Every gold entry must carry a verifiable source** (DOI or paper reference).
- `STATUS: needs_doi` entries are placeholders: their physiology anchor is
  common knowledge, but the *exact* source must be pinned by hand **before**
  any number enters the paper.
- Self-consistent entries (derived purely from the governing equations) are
  allowed — label them as such.

## Metrics, carefully

- **hallucination rate** — fraction of a design's derived numbers that the
  verifier rejects. A bare-LLM number that doesn't follow from its own
  geometry/flow is a hallucination; Labwright's derived numbers come from the
  calculators, so this is ≈ 0 by construction.
- **self-consistent rate** — fraction of gold entries with hallucination rate 0.
- **usable rate** — fraction of gold entries that are self-consistent **and**
  recover every gold target parameter within ±5 %. A design that is internally
  consistent but misses the physiological target (e.g. builds a clean 0.1 Pa
  chip when the goal demanded 0.05 Pa) is not usable.

The bare-LLM asymmetry is deliberate and favours the bare model: it is retried
up to 3× on empty responses, given reasoning disabled (else the v4 models spend
the whole output budget thinking and emit nothing), asked for a *minimal*
per-goal key set, and allowed a generous ±5 % consistency tolerance, while
Labwright must match the calculators to 1e-6.

## Status

- [x] Curate the gold set: **24 organ-on-chip design goals** spanning
      microfluidics, cell seeding, dosing/DMSO and statistics. Every entry
      carries a provenance rule — a pinned source (e.g. kidney PTEC 0.2 dyn/cm²,
      Jang 2013, `10.1039/c3ib40049b`; arterial/venular shear from
      Papaioannou & Stefanadis, PMID 15807389) or an explicit
      `self-consistent` label. No fabricated literature numbers anywhere.
- [x] Run the two-system comparison on `deepseek-v4-flash` and `deepseek-v4-pro`
      → `results/eval_flash.json`, `results/eval_pro.json`; `python -m eval.report results/eval_flash.json`.
      The runner checkpoints after every entry, so a mid-run failure never
      loses the API spend.
- [x] Reverse-verify a batch of published protocols + labelled synthetic
      controls: `eval/run_verify_batch.py` → `results/eval_verify_batch.json`.
- [x] Preprint draft in `paper/manuscript.md` (numbers from the committed
      results), Colab notebook, HF Space scaffolding.
- [ ] Ablations: model size, RAG context on/off, tool-calling on/off.
- [ ] Submit the preprint to bioRxiv; publish the HF Space.

## Run

```bash
python -m eval.run_benchmark                     # all gold entries
python -m eval.run_benchmark --limit 3 --out /tmp/eval.json
python -m eval.run_benchmark --model deepseek-chat
python -m eval.report results/eval_flash.json    # render the comparison table
```

Requires `LABWRIGHT_API_KEY`/`DEEPSEEK_API_KEY`. Results are written to
`results/` and committed so the benchmark is reproducible from the repo alone.
