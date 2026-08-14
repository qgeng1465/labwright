#!/bin/sh
# Finetuned benchmark for lora_v4: the 11-domain v2 dataset (57k rows with
# cross-domain composites + negative samples) trains with the multi-block
# prompt, so the benchmark must decode with --multi-block too. Run only after
# results/extractor/lora_v4 exists. GPU-bound: V100 fp16, ~3 s/row.
# Usage:  sh eval/run_lora_v4_finetuned.sh
set -e
cd /data/qiushuogeng/projects/labwright
ADAPTER=lora_v4
ADAPTER_DIR="results/extractor/${ADAPTER}"
echo "== finetuned benchmark adapter=$ADAPTER (--multi-block) at $(date +%H:%M:%S) =="

# Legacy five-domain golds
for GOLD in reading:gold_experiments culture:gold_cell_culture \
            spheroid:gold_spheroid pk:gold_pk blind:gold_blind; do
    NAME="${GOLD%%:*}"; FILE="${GOLD##*:}"
    .venv/bin/python -m eval.run_finetuned_benchmark \
        --gold "eval/${FILE}.json" \
        --adapter "${ADAPTER_DIR}" \
        --out "results/eval_finetuned_${NAME}_${ADAPTER}.json" \
        --multi-block
done

# New-domain 14-gold benchmark
.venv/bin/python -m eval.run_finetuned_benchmark \
    --gold eval/gold_new_domains.json \
    --adapter "${ADAPTER_DIR}" \
    --out "results/eval_finetuned_newdomains_${ADAPTER}.json" \
    --multi-block

echo "== done $ADAPTER at $(date +%H:%M:%S) =="
