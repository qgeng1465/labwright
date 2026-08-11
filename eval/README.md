# Labwright benchmark — paper evidence

## Question

Do LLM-written wet-lab designs contain hallucinated numbers, and does
constraining the LLM to *propose* while *calculators compute* fix it?

## Protocol (reserved experiment window)

On every gold-standard experiment, run two systems:

| system | behavior |
|---|---|
| bare-LLM | the model writes the full design JSON, including derived numbers, from memory |
| Labwright | the model proposes raw inputs; derived numbers come from `labwright.calc` and pass `labwright.verify` |

Metrics (from `benchmark.evaluate`):

- **parameter recovery** — mean relative error of the design vs the gold
  standard on `shear_pa`, `flow_rate_uLmin`, `seed_count`, `n_per_group`.
- **hallucination rate** — fraction of derived fields that the verifier rejects.

Expected result to publish: bare-LLM hallucination rate substantially above
zero (reported figures for unconstrained biomedical LLMs are routinely
30–60% on numerical protocol details); Labwright rate ≈ 0 by construction.
Recovery accuracy should also improve because the model is told the actual
tool outputs mid-loop.

## Provenance rules (hard requirements)

- **Every gold entry must carry a verifiable source** (DOI or paper reference).
- `STATUS: needs_doi` entries are placeholders: their physiology anchor is
  common knowledge, but the *exact* source must be pinned by hand (or via a
  literature agent) **before** any number enters the paper.
- Self-consistent entries (derived purely from the governing equations) are
  allowed — label them as such.

## Status

- [ ] Pin DOIs for the physiology-anchored gold entries (target: 20–30 real
      experiments curated from organ-on-chip papers).
- [ ] Run the two-system comparison with the DeepSeek key.
- [ ] Ablations: model size, RAG context on/off, tool-calling on/off.
- [ ] Write up as a preprint (bioRxiv) with the repo + this benchmark.

## Run

```bash
python -m eval.benchmark
```

Requires `LABWRIGHT_API_KEY`/`DEEPSEEK_API_KEY`. This is a reserved experiment
window — it is launched only after the code and demo have traction, per the
project's real-significance rule.
