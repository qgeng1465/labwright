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
- Benchmark (bare LLM vs Labwright): **0.000** hallucination rate vs **33–58 %**.

Labwright is an experimental-design aid, not medical-device software.
