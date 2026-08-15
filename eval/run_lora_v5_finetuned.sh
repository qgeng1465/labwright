#!/bin/sh
# lora_v5 six-set benchmark: plain + repair-retries=2 for every gold set.
# lora_v5 = v4 data + hand-written-register (natural-prose) variants for the
# seven v2 domains, trained on results/extractor_11dom_v3. The headline
# question is whether the new-register training lifts newdomains from v4's
# 0/14 usable (the schema-prompt A/B said the gap is data, not prompt).
# Outputs results/eval_finetuned_*_lora_v5{,_repair}.json, compared against
# the committed lora_v4 files with eval/compare_repair.py.
# GPU-bound: V100 fp16, ~4 s/row base + repair attempts.
# Usage:  sh eval/run_lora_v5_finetuned.sh
set -e
cd /data/qiushuogeng/projects/labwright
ADAPTER=lora_v5
ADAPTER_DIR="results/extractor/${ADAPTER}"
echo "== lora_v5 benchmark adapter=$ADAPTER at $(date +%H:%M:%S) =="

for GOLD in reading:gold_experiments culture:gold_cell_culture \
            spheroid:gold_spheroid pk:gold_pk blind:gold_blind; do
    NAME="${GOLD%%:*}"; FILE="${GOLD##*:}"
    .venv/bin/python -m eval.run_finetuned_benchmark \
        --gold "eval/${FILE}.json" \
        --adapter "${ADAPTER_DIR}" \
        --out "results/eval_finetuned_${NAME}_${ADAPTER}.json" \
        --multi-block
    .venv/bin/python -m eval.run_finetuned_benchmark \
        --gold "eval/${FILE}.json" \
        --adapter "${ADAPTER_DIR}" \
        --out "results/eval_finetuned_${NAME}_${ADAPTER}_repair.json" \
        --multi-block --repair-retries 2
done

# New-domain 14-gold benchmark
.venv/bin/python -m eval.run_finetuned_benchmark \
    --gold eval/gold_new_domains.json \
    --adapter "${ADAPTER_DIR}" \
    --out "results/eval_finetuned_newdomains_${ADAPTER}.json" \
    --multi-block
.venv/bin/python -m eval.run_finetuned_benchmark \
    --gold eval/gold_new_domains.json \
    --adapter "${ADAPTER_DIR}" \
    --out "results/eval_finetuned_newdomains_${ADAPTER}_repair.json" \
    --multi-block --repair-retries 2

echo "== done lora_v5 benchmark at $(date +%H:%M:%S) =="
