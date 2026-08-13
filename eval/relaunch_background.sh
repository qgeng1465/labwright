#!/bin/sh
# Relaunch of background benchmark jobs, key sourced from .env (the key file
# at ~/deepseek_key.txt is stale/invalid). Runs two parallel chains:
#   A: spheroid seed-CI sweep (full systems, flash+pro, 3 seeds)
#   B: nogate culture+pk (flash) -> labwright_iter comparison (4 sets, flash)
set -e
cd /data/qiushuogeng/projects/labwright
export DEEPSEEK_API_KEY="$(grep '^DEEPSEEK_API_KEY=' .env | cut -d= -f2-)"

# ---- Chain A: spheroid seed-CI ----
nohup .venv/bin/python -m eval.run_seed_benchmark \
    --gold eval/gold_spheroid.json --seeds 3 \
    --systems bare,soft_gate,self_verify,labwright \
    --models deepseek-v4-flash,deepseek-v4-pro \
    --out results/eval_seed_spheroid.json \
    > results/seed_spheroid.log 2>&1 &
echo "chain A (spheroid seed) pid $!"

# ---- Chain B: nogate then iter, sequential ----
nohup sh -c '
  export DEEPSEEK_API_KEY="$(grep "^DEEPSEEK_API_KEY=" /data/qiushuogeng/projects/labwright/.env | cut -d= -f2-)"
  cd /data/qiushuogeng/projects/labwright
  echo "== nogate culture $(date +%H:%M:%S) =="
  .venv/bin/python -m eval.run_benchmark --gold eval/gold_cell_culture.json \
      --systems labwright,tool_no_gate --out results/eval_nogate_culture_flash.json \
      || echo "nogate culture FAILED"
  echo "== nogate pk $(date +%H:%M:%S) =="
  .venv/bin/python -m eval.run_benchmark --gold eval/gold_pk.json \
      --systems labwright,tool_no_gate --out results/eval_nogate_pk_flash.json \
      || echo "nogate pk FAILED"
  for s in blind:gold_blind spheroid:gold_spheroid culture:gold_cell_culture pk:gold_pk; do
    name="${s%%:*}"; gold="${s##*:}"
    echo "== iter $name $(date +%H:%M:%S) =="
    .venv/bin/python -m eval.run_benchmark --gold "eval/$gold.json" \
        --systems labwright,labwright_iter --max-submission-attempts 3 \
        --out "results/eval_iter_${name}_flash.json" \
        || echo "iter $name FAILED"
  done
  echo "== chain B done $(date +%H:%M:%S) =="
' > results/chain_b.log 2>&1 &
echo "chain B (nogate+iter) pid $!"
