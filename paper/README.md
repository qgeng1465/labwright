# Paper figures

Benchmark figures for the Labwright paper. Each figure is rendered by a
committed script from the committed benchmark outputs, so the figures are
fully reproducible:

| figure | script | data |
|---|---|---|
| pipeline | `fig_pipeline.py` | `eval/` + `labwright/` |
| blind-goal shear recovery | `fig_blind_goals.py` | `results/eval_blind_{flash,pro}.json` |
| benchmark small multiples | `fig_benchmark.py` | `results/eval_{flash,pro}.json` |
| reverse-verify judgement matrix | `fig_verify.py` | `results/eval_verify_batch.json` |

```bash
python paper/fig_pipeline.py
python paper/fig_blind_goals.py
python paper/fig_benchmark.py \
    results/eval_flash.json results/eval_flash.json \
    results/eval_pro.json results/eval_pro.json \
    results/eval_blind_flash.json results/eval_blind_flash.json \
    results/eval_blind_pro.json results/eval_blind_pro.json
python paper/fig_verify.py
```

> **The manuscript itself is not in this repository.** It is kept local-only
> while the preprint is in submission; this repo ships the reproducible
> figures, the benchmark data (`results/`), and the honest methodology write-up
> in `eval/README.md`.
