# Benchmark gold data — provenance & invariants

The benchmark rests on two hand-curated gold sets. Every number in them has a
stated origin; no value is invented. This file says how to read that, and how
to add a goal without breaking the invariants.

## The two sets

| file | n | what it tests | does the goal state the answer? |
|---|---|---|---|
| `gold_experiments.json` | 24 | **reading**: number-extraction + tool-calling | yes — every goal states the geometry/flow/density/effect-size or the physiological target |
| `gold_blind.json` | 12 | **recall**: target selection from domain knowledge | no — the goal states physiology ("recapitulate physiological venular wall shear"), the model must supply the canonical number |

Each entry is `{id, goal, expected, source}` (blind adds `blind_strength`).

## Provenance rules (both sets)

Every `expected` value must fall into exactly one of three buckets:

1. **Pinned to a citable source** — a DOI or PMID is named, e.g. Jang et al.
   (*Lab Chip* 2013, c3ib40049b), Papaioannou & Stefanadis (PMID 15807389),
   Koutsiaris (*Int J Nanomedicine*), HepG2 seeding
   (Sci Rep 10.1038/s41598-021-81733-3), primary-hepatocyte sandwich plating
   (Bioengineering 10.3390/bioengineering10020131), Sumida
   (10.1177/0960327111399325).
2. **Explicit `design-target` / `self-consistent` label** — the entry does *not*
   claim a literature number; it is a construction target that the system
   should be able to derive from the goal's own stated inputs (a goal that
   hands over geometry + flow and asks for the resulting shear is
   self-consistent by construction). These are labelled as such in `source`;
   they must never be phrased as if they came from a paper.
3. **`prompt-backed` blind entries** (liver, lung, BBB) — the canonical target
   *is* listed as a range in the Labwright system prompt, but the model must
   still select the right value within it.

No bucket may be empty and no number may sit in an unnamed fourth bucket. If a
value cannot be sourced, it is removed or relabelled, not silently kept.

## Scoring invariants (enforced in `benchmark.py`, mirrored in `eval/README.md`)

- **±5 % consistency tolerance** on every derived field.
- **Unverifiable = 1.0 hallucination**: a system that reports raw inputs but no
  checkable derived numbers is scored as if it produced nothing — the same
  convention Labwright uses for a run that never submits.
- **Reading-set recovery is constructive**: the self-consistent anchors are
  computed from the same equations Labwright uses, so recovery ≈ 0 there is by
  construction. The real signal is extraction and tool-calling.
- **Blind-set usability is where domain knowledge is tested**: usable =
  self-consistent **and** recovers every gold target within ±5 %. The honest
  cold-only sub-rates (excluding the three `prompt-backed` goals) are reported
  alongside the headline 25 %/33 %.

## Adding a goal

1. Write the `goal` and `expected`, and pin `source` to one of the three
   buckets above — no new bucket, no empty source.
2. For a reading goal, add the raw inputs to the harness so the anchors derive
   from the calculators (see `benchmark.py` `_RAW_KEYS` / `_DERIVED_FIELDS`).
3. Re-run the benchmark and commit the new `results/eval_*.json`.
4. Add a regression test that re-derives the entry's `expected` values from the
   goal's stated numbers, so transcription drift is caught at test time.

## Reproducibility

All `results/eval_*.json` are committed, and every paper figure
(`paper/fig_*.py`) renders from those committed outputs. A figure is
reproducible iff its script reads only `results/` and `eval/` — the figures
never read the (local-only) manuscript.
