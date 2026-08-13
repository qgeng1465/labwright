---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
  - other
tags:
  - scientific-audit
  - protocol
  - reproducibility
  - computational-verification
size_categories:
  - 1K<n<10K
pretty_name: SciRecipe Audit Dataset
---

# SciRecipe Audit Dataset

A large-scale, machine-checkable audit of **21,094 real biological protocol
summaries** from the [SciRecipe corpus](https://huggingface.co/datasets/manglu3935/SciRecipe)
(manglu3935/SciRecipe), re-deriving every reported number from the protocol's
own stated inputs and classifying each protocol as internally consistent,
internally inconsistent, or not checkable from its abstract. This is the dataset
behind the reproducibility-gap figure in the Labwright paper; it is released
standalone so it can be used, extended, or contested independently of Labwright.

## What it measures

SciRecipe's `orc` column is an ordered natural-language protocol summary
(abstract-level). For every summary that states at least one number+unit and
routes to a checkable domain (plate culture or microfluidics), the audit:

1. **harvests the derived numbers the text asserts** (shear stress, Reynolds
   number, seed per well, medium volume per well, confluence …);
2. **extracts raw inputs** (geometry, flow rate, seeding density, chip
   dimensions) with a fine-tuned extractor;
3. **recomputes** each derived number from those inputs with the same
   unit-tested calculators Labwright uses;
4. **classifies** the row `ok` (every stated number follows from the stated
   inputs), `review_required` (some stated number is contradicted by the
   protocol's own inputs), or `unverifiable` (the summary does not say enough —
   no domain, no extractable raw, or no stated derived number).

### The funnel (read the denominators exactly)

| stage | count |
|---|---|
| SciRecipe rows | 21,094 |
| state ≥1 number+unit | 14,589 |
| route to a checkable domain (culture / flow) | 5,700 |
| audited (with extractor) | 5,700 |
| state a derived number that can be re-derived | **104** |
| — internally consistent (`ok`) | **30** |
| — internally inconsistent (`review_required`) | **74** |

The headline rate is **30/104 = 28.8 %** internally consistent **among the rows
that say enough to check** — never "28.8 % of the literature". 5,596 of the
5,700 audited rows state no derived number recoverable from their own inputs and
are `unverifiable`, counted as neither consistent nor inconsistent. An early
version of this audit counted no-derived-number rows as "ok", inflating the rate
to 0.898; a regression test pins the honest version below.

## Literature provenance

SciRecipe stores each protocol's *objective* (`exp_goal`); it does **not** store
a DOI. A subset of `exp_goal`s embed the source protocol's quoted title (e.g.
`In the "Spot Assays for Viability Analysis of Cyanobacteria" protocol, …`). The
audit harvests those titles and resolves them to real papers via Crossref
(`works?query.bibliographic`), keeping a hit **only** when the resolved title
matches the quoted title to high string similarity (≥ 0.90) — a loose query is
recorded as `medium`/`none` and *not* counted as a citation.

Measured resolution coverage (this release):

| stage | count | rate |
|---|---|---|
| audited rows | 5,700 | — |
| rows with ≥1 quoted protocol title in `exp_goal` | 2,825 | 49.6 % of audited |
| distinct quoted titles | 2,426 | — |
| titles resolved to a **high-confidence** DOI (match ≥ 0.90) | 2,185 | 77.3 % of titled rows |
| — resolved to a `medium` match (0.60–0.90, flagged, not cited) | 360 | — |
| — no/below-threshold match | 280 | — |

So 2,185 / 5,700 = **38.3 %** of audited rows carry a DOI verified against the
quoted protocol title. The other 62 % do **not** — the majority because their
`exp_goal` embeds no quoted title (a real property of the source text, reported
not hidden), a minority because Crossref returned no sufficiently similar paper.
Every `literature` record carries its own `quality` and match `score` so a reader
can re-apply a stricter gate.

Within the 2,185 high-confidence-DOI rows, the audit's checkable subpopulation
is `ok` 9 / `review_required` 32 (honest-consistent 9/41 = 22.0 %), consistent
with the 28.8 % headline across all rows — the reported numbers do not change
when restricted to rows whose literature provenance is verified.

## Files

| file | contents |
|---|---|
| `scirecipe_audit_enriched.json` | full audit records: verdict, reason, claimed/computed numbers, `source_idx` → SciRecipe row, `exp_goal`, quoted titles, and `literature` (Crossref DOI + match quality) when resolved |
| `scirecipe_provenance.csv` | one row per (source row, quoted title): DOI, container, year, match score, quality |
| `../eval/run_scirecipe_audit.py` | the audit funnel + verifier (rerunnable) |
| `../eval/enrich_scirecipe.py` | the Crossref provenance layer (rerunnable, incremental cache) |
| `../eval/retry_crossref.py` | slow serial retry for rate-limited Crossref titles, then re-emits the dataset |

## Reproducibility

```bash
# 1. audit (GPU extractor optional; --funnel-only needs none)
python -m eval.run_scirecipe_audit --funnel-only            # deterministic funnel
python -m eval.run_scirecipe_audit --adapter results/extractor/lora   # with extractor

# 2. attach real-literature provenance
python -m eval.enrich_scirecipe

# 3. (optional) slow, polite retry of rate-limited Crossref titles
python -m eval.retry_crossref
```

The verifier and calculators are the same unit-tested modules Labwright uses at
design time; the audit is therefore *not* a bespoke scorer — it is Labwright's
`verify_published_protocol` applied to the literature.

## Licensing

The source SciRecipe corpus is released by its authors under the MIT license.
This audit is derived analysis of that corpus; it is released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to
SciRecipe (manglu3935/SciRecipe). DOI metadata is Crossref data (available under
the Crossref Terms of Use). *This is the author's judgment; for a specific reuse,
check the source corpus's license and the Crossref terms.*

## Contact / citation

Labwright project (github.com/qgeng1465/labwright). When using this dataset,
cite the Labwright paper (in submission) and the SciRecipe corpus paper
(arXiv:2510.15600).
