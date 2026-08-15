#!/bin/sh
# Schema-repair benchmark for lora_v4: when the extractor's raw block fails the
# schema/build gate, re-prompt the model with the validator error (up to 2 extra
# attempts) and re-run the same gate. A/B against the plain lora_v4 files
# (results/eval_finetuned_*_lora_v4.json): greedy decoding is deterministic, so
# entries the repair does not touch are byte-identical.
# GPU-bound: V100 fp16, ~3 s/row base + repair attempts.
# Usage:  sh eval/run_lora_v4_repair.sh
set -e
cd /data/qiushuogeng/projects/labwright
ADAPTER=lora_v4
ADAPTER_DIR="results/extractor/${ADAPTER}"
echo "== repair benchmark adapter=$ADAPTER retries=2 at $(date +%H:%M:%S) =="

# Legacy five-domain golds
for GOLD in reading:gold_experiments culture:gold_cell_culture \
            spheroid:gold_spheroid pk:gold_pk blind:gold_blind; do
    NAME="${GOLD%%:*}"; FILE="${GOLD##*:}"
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
    --out "results/eval_finetuned_newdomains_${ADAPTER}_repair.json" \
    --multi-block --repair-retries 2

echo "== done repair at $(date +%H:%M:%S) =="
