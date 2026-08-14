#!/bin/sh
# Run the fine-tuned extractor benchmark for lora_v3 (and lora_v2) across the
# five legacy domains + the 14 new-domain golds. Run only after the target
# adapter exists. GPU-bound: V100 fp16, ~3 s/row.
# Usage:  sh eval/run_lora_v3_finetuned.sh [lora_v2|lora_v3]
set -e
cd /data/qiushuogeng/projects/labwright
ADAPTER=${1:-lora_v3}
ADAPTER_DIR="results/extractor/${ADAPTER}"
echo "== finetuned benchmark adapter=$ADAPTER at $(date +%H:%M:%S) =="

# Legacy five-domain golds
for GOLD in reading:gold_experiments culture:gold_cell_culture \
            spheroid:gold_spheroid pk:gold_pk blind:gold_blind; do
    NAME="${GOLD%%:*}"; FILE="${GOLD##*:}"
    .venv/bin/python -m eval.run_finetuned_benchmark \
        --gold "eval/${FILE}.json" \
        --adapter "${ADAPTER_DIR}" \
        --out "results/eval_finetuned_${NAME}_${ADAPTER}.json"
done

# New-domain 14-gold benchmark
.venv/bin/python -m eval.run_finetuned_benchmark \
    --gold eval/gold_new_domains.json \
    --adapter "${ADAPTER_DIR}" \
    --out "results/eval_finetuned_newdomains_${ADAPTER}.json"

echo "== done $ADAPTER at $(date +%H:%M:%S) =="
