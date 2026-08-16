# Paper figures

Benchmark figures for the Labwright paper. Each figure is rendered by a
committed script from the committed benchmark outputs, so the figures are
fully reproducible:

| figure | script | data |
|---|---|---|
| system framework (Fig 1) | `fig_pipeline.py` | `eval/` + `labwright/` |
| graphical abstract | `fig_abstract.py` | headline numbers synced from `results/eval_{flash,pro}.json` (in-code `BENCH` band) |
| architecture anatomy (a)–(e) | `fig_architecture.py` | 46-tool registry from `labwright/tools.py` + `BENCH` dict synced from `results/eval_*.json` |
| blind-goal shear recovery | `fig_blind_goals.py` | `results/eval_blind_{flash,pro}.json` |
| benchmark small multiples | `fig_benchmark.py` | `results/eval_{flash,pro}.json`, `eval_blind_*`, `eval_spheroid_*`, `eval_culture_*`, `eval_pk_*` |
| cross-backend comparison | `fig_model_compare.py` | `results/eval_{flash,pro,k3,kimicode}.json` and the `eval_blind_*`/`eval_spheroid_*`/`eval_culture_*`/`eval_pk_*` variants |
| reverse-verify judgement matrix | `fig_verify.py` | `results/eval_verify_batch.json` |
| SciRecipe reverse-verification | `fig_scirecipe.py` | `results/eval_scirecipe_audit.json` |

```bash
python paper/fig_pipeline.py
python paper/fig_abstract.py
python paper/fig_architecture.py
python paper/fig_blind_goals.py \
    results/eval_blind_flash.json results/eval_blind_pro.json
python paper/fig_benchmark.py \
    results/eval_flash.json results/eval_flash.json \
    results/eval_pro.json results/eval_pro.json \
    results/eval_blind_flash.json results/eval_blind_flash.json \
    results/eval_blind_pro.json results/eval_blind_pro.json \
    results/eval_spheroid_flash.json results/eval_spheroid_flash.json \
    results/eval_spheroid_pro.json results/eval_spheroid_pro.json \
    results/eval_culture_flash.json results/eval_culture_flash.json \
    results/eval_culture_pro.json results/eval_culture_pro.json \
    results/eval_pk_flash.json results/eval_pk_flash.json \
    results/eval_pk_pro.json results/eval_pk_pro.json
python paper/fig_model_compare.py
python paper/fig_verify.py results/eval_verify_batch.json
python paper/fig_scirecipe.py
```

After rendering, `python paper/_check_render.py` re-checks every figure (7 of
the 8; `fig_verify` is verified by the in-script overlap census) for
text-overlap and canvas overflow and must exit `TOTAL 0`.

> **The manuscript itself is not in this repository.** It is kept local-only
> while the preprint is in submission; this repo ships the reproducible
> figures, the benchmark data (`results/`), and the honest methodology write-up
> in `eval/README.md`.
