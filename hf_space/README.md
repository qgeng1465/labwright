---
title: Labwright
emoji: 🧪
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# 🧪 Labwright — the AI bench copilot that gets your numbers right

Type a wet-lab experiment goal and get a **verified** design: chip geometry,
flow & shear stress, seeding, DMSO carry-over and replicate counts.

The language model **proposes** raw inputs; the **calculators** produce and
**verify** every derived number — so nothing is hallucinated. Reverse-verify a
published protocol's numbers in the second tab (no API key needed: it's pure
deterministic physics).

## Setup for Space admins

Add a `DEEPSEEK_API_KEY` Space secret to enable the design tab (any
OpenAI-compatible API works; default model `deepseek-v4-flash`). The
reverse-verification tab works without any key.

- Repo: https://github.com/qgeng1465/labwright
- Open audit dataset (the same verifier over 21,094 real SciRecipe protocols,
  with Crossref DOI provenance):
  https://huggingface.co/datasets/qgeng1465/scirecipe-audit
- Benchmark (bare LLM vs Labwright, five gold sets — reading, blind,
  spheroid, plate-culture, perfused-PK): a bare frontier LLM hallucinates
  ~0.9–1.0 of its derived numbers and rarely produces a usable design;
  Labwright's hallucination rate is **0.000** by construction with **64–100 %**
  usable designs on every set except the blind set, where usable drops to
  40–47 % because the gate cannot supply physiology the model doesn't know. The
  gap is stable across seeds (95 % Wilson intervals never overlap the memory
  systems), and an iterating fix-and-resubmit agent repairs all verifier-fired
  errors (41/41) while being a wash on usable rate.

Labwright is an experimental-design aid, not medical-device software.
