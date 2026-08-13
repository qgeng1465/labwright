#!/bin/sh
# labwright_iter (fix-and-resubmit) comparison across the harder gold sets.
# Paired labwright (first-submit) vs labwright_iter (max 3 submission attempts)
# on the same set, flash model. Runs the four sets sequentially so the API load
# stays at one extra client.
set -e
cd /data/qiushuogeng/projects/labwright
export DEEPSEEK_API_KEY="$(tr -d '\r\n ' < /home/qiushuogeng/deepseek_key.txt)"
export LABWRIGHT_MODEL=deepseek-v4-flash

run_set() {
    gold="$1"; out="$2"
    echo "=== [$gold] $(date +%H:%M:%S) ==="
    .venv/bin/python -m eval.run_benchmark \
        --gold "$gold" \
        --systems labwright,labwright_iter \
        --max-submission-attempts 3 \
        --out "$out"
}

run_set eval/gold_blind.json     results/eval_iter_blind_flash.json
run_set eval/gold_spheroid.json  results/eval_iter_spheroid_flash.json
run_set eval/gold_cell_culture.json results/eval_iter_culture_flash.json
run_set eval/gold_pk.json        results/eval_iter_pk_flash.json
echo "=== iter sweep complete $(date +%H:%M:%S) ==="
