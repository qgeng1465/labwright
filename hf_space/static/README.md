---
title: Labwright
emoji: 🧪
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: apache-2.0
---

# 🧪 Labwright

Verified wet-lab design assistant. Type a wet-lab experiment goal and get a
**verified** design — chip geometry, flow & shear, seeding, DMSO carry-over,
spheroid volumes — where the language model proposes raw inputs and
deterministic calculators derive and verify every number.

This static page is a showcase. The interactive Gradio app lives in the
repository's `hf_space/` package and can be deployed as a Gradio Space
(the design tab calls an OpenAI-compatible API via a Space secret; the
reverse-verify tab runs offline on pure calculators).

- **Repo:** https://github.com/qgeng1465/labwright
- **Open audit dataset** (the same verifier over 21,094 real SciRecipe
  protocols, with Crossref DOI provenance):
  https://huggingface.co/datasets/qgeng1465/scirecipe-audit
- **Key idea:** the calculator, not the model, is the knowledge base — every
  submitted number is re-derived from its own raw inputs.

**Honest boundary:** the gate re-derives every submitted number from the
design's own raw inputs, so it verifies *arithmetic* (a number is either
recomputed by a calculator or it does not enter the design) — it does **not**
vouch for physiology. On the blind set (no target stated) the usable rate
drops to 40–47 %, cold-only 38 %; the fast-path extractor's new-domain score
(4/14, 5/14 with schema repair) is measured on goals whose phrasing its
training register mirrors — training values are sampled and golds withheld,
never verbatim. Full, machine-checkable numbers and the register disclosure:
the repo's [`eval/README.md`](https://github.com/qgeng1465/labwright/blob/main/eval/README.md).
