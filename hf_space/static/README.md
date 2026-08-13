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
- **Key idea:** the calculator, not the model, is the knowledge base — every
  submitted number is re-derived from its own raw inputs.
