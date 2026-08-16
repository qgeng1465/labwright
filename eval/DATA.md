# Benchmark gold data — provenance & invariants

The benchmark rests on five hand-curated gold sets. Every number in them has a
stated origin; no value is invented. This file says how to read that, and how
to add a goal without breaking the invariants.

## The five sets

| file | n | what it tests | does the goal state the answer? |
|---|---|---|---|
| `gold_experiments.json` | 24 | **reading**: number-extraction + tool-calling | yes — every goal states the geometry/flow/density/effect-size or the physiological target |
| `gold_blind.json` | 15 | **recall**: target selection from domain knowledge | no — the goal states physiology ("recapitulate physiological venular wall shear"), the model must supply the canonical number |
| `gold_cell_culture.json` | 14 | **reading + recall** in the plate-culture domain (wells, seeding, counting, viability, confluence) | 10 reading (plate geometry / density stated) and 4 blind-`cold` (model must recall the pinned PHH sandwich density or plate-table volume) |
| `gold_spheroid.json` | 15 | **reading + recall + scenarios** in the 3D-culture domain (spheroid/organoid geometry, ULA & hanging-drop working volumes, necrotic-core limits) | 11 reading/scenario (geometry and targets stated, or a failure mode named) and 4 blind (2 `prompt-backed`: 1000 cells/spheroid, 96-ULA 100 µL — stated in the system prompt; 2 `cold`: 384-ULA 50 µL, hanging-drop 20 µL — in neither the goal nor the prompt) |
| `gold_pk.json` | 14 | **reading + recall + scenarios** in the perfused-system PK domain (extraction ratio, clearance, half-life, accumulation, mass cleared; unit traps) | 12 reading/scenario (every input stated, or the formula's raw numbers given; 2 unit-ambiguity) and 2 blind `prompt-backed` (propranolol high-extraction / antipyrine low-extraction — the classification is stated in the system prompt) |

Each entry is `{id, goal, expected, source}` (blind adds `blind_strength`).

## Provenance rules (all sets)

Every `expected` value must fall into exactly one of three buckets:

1. **Pinned to a citable source** — a DOI or PMID is named, e.g. Jang et al.
   (*Lab Chip* 2013, c3ib40049b), Papaioannou & Stefanadis (PMID 15807389),
   Koutsiaris (*Int J Nanomedicine*), HepG2 seeding
   (Sci Rep 10.1038/s41598-021-81733-3), primary-hepatocyte sandwich plating
   (Bioengineering 10.3390/bioengineering10020131), Sumida
   (10.1177/0960327111399325); PK-classification and single-compartment equations
   to Rowland & Tozer and Gibaldi & Perrier, and the propranolol intrinsic-clearance
   design target to J Pharm Sci 2014 (doi:10.1002/jps.23796, Baudoin et al.).
   An earlier draft cited a "PhysioMimix LC-12 media-exchange flow of 60 µL/min"
   from "Docci et al., Lab Chip 2022" — that DOI (10.1039/d1lc00784f) does not
   exist on Crossref, the real Docci paper is doi:10.1039/d1lc01161h, and the
   specific flow-rate numbers could not be independently verified in accessible
   full text, so the entry was removed rather than kept on an unverifiable
   literature claim.
2. **Explicit `design-target` / `self-consistent` label** — the entry does *not*
   claim a literature number; it is a construction target that the system
   should be able to derive from the goal's own stated inputs (a goal that
   hands over geometry + flow and asks for the resulting shear is
   self-consistent by construction). These are labelled as such in `source`;
   they must never be phrased as if they came from a paper.
3. **`prompt-backed` blind entries** (liver, venular, lung, BBB, lymphatic; spheroid 1000
   cells/spheroid and 96-ULA 100 µL; PK propranolol high-extraction and
   antipyrine low-extraction) — the canonical target *is* listed in the
   Labwright system prompt (as a range or an anchor value), but the model must
   still select the right value within it.

No bucket may be empty and no number may sit in an unnamed fourth bucket. If a
value cannot be sourced, it is removed or relabelled, not silently kept.

## Scoring invariants (enforced in `benchmark.py`, mirrored in `eval/README.md`)

- **±5 % consistency tolerance** on every derived field.
- **Unverifiable = 1.0 hallucination**: a system that reports raw inputs but no
  checkable derived numbers is scored as if it produced nothing — the same
  convention Labwright uses for a run that never submits. The bare/soft-gate
  checkers are domain-aware: flow answers are cross-checked against geometry +
  flow, plate-culture answers against plate_format + seeding density (+ wells),
  so a culture gold is scored by the same "do its numbers follow from its own
  inputs" test as a flow gold; spheroid answers are cross-checked against
  spheroid_format + cells_per_spheroid + spheroid_count + cell_diameter_um, so a
  spheroid gold is scored by the same self-consistency test; PK answers are
  cross-checked against inlet/outlet concentration × flow (extraction ratio and
  clearance), plus the extra fields only when their own inputs (system volume,
  dose interval, molecular weight) are reported. A zero inlet concentration
  makes the answer unverifiable (E is undefined) — scored 1.0, never a crash.
- **String formats are extracted, and each derived field is recomputed from
  exactly the raws it needs** (fairness fix): `spheroid_format` / `plate_format`
  are found by name anywhere in the reported JSON (a `"384-well ULA plate"`
  string, not just floats), spheroid geometry is checkable from
  cells_per_spheroid + cell_diameter_um alone, vessel-derived numbers only from
  a *parseable* format string, growth from doubling × duration — and a number
  that is reported but not recomputable from the raws is *excluded*, never
  counted wrong. This removed the artifact where every spheroid convention goal
  scored 1.0 for the memory systems because the vessel string was never
  extracted.
- **Reading-set recovery is constructive**: the self-consistent anchors are
  computed from the same equations Labwright uses, so recovery ≈ 0 there is by
  construction. The real signal is extraction and tool-calling.
- **Blind-set usability is where domain knowledge is tested**: usable =
  self-consistent **and** recovers every gold target within ±5 %. The honest
  cold-only sub-rates (excluding the five `prompt-backed` and two scenario
  goals) are reported alongside the headline 40 %/47 %.

## Adding a goal

1. Write the `goal` and `expected`, and pin `source` to one of the three
   buckets above — no new bucket, no empty source.
2. For a reading goal, add the raw inputs to the harness so the anchors derive
   from the calculators. Each design domain's keys are declared once in
   `labwright/blocks.py` (a domain's `Block`: `raw_keys`, `derived_keys`,
   `consistency_keys`, plus its `field_map` / `sanity_bands` / `canonical_units`);
   `benchmark.py` imports them from there, so adding a goal's inputs means
   editing that domain's `Block`, not a per-system list in the harness.
3. Re-run the benchmark and commit the new `results/eval_*.json`.
4. Add a regression test that re-derives the entry's `expected` values from the
   goal's stated numbers, so transcription drift is caught at test time. For the
   culture set this is `tests/test_gold_culture.py::test_gold_is_self_consistent`,
   for the spheroid set `tests/test_gold_spheroid.py::test_gold_is_self_consistent`,
   for the PK set `tests/test_gold_pk.py::test_gold_is_self_consistent`
   — each recomputes every gold `expected` through the domain calculators
   (`labwright.calc.culture` / `labwright.calc.spheroid` / `labwright.calc.pk`)
   from the raw inputs stated in the goal prose.

## Reproducibility

All `results/eval_*.json` are committed, and every paper figure
(`paper/fig_*.py`) renders from those committed outputs. A figure is
reproducible iff its script reads only `results/` and `eval/` — the figures
never read the (local-only) manuscript.
