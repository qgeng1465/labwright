#!/bin/sh
# Schema-prompt A/B for lora_v4 on the two schema-failing sets (newdomains, pk).
#
# Root cause (v4 adopt, 2026-08-15): the model selects the right block type on
# the hand-written new-domain goals but drops REQUIRED fields (e.g.
# resistance_total_ohm, target_po2_mmhg) or emits derived fields it was told
# never to compute — a field-completeness gap on out-of-register prose. The
# explicit key-name/unit SCHEMA_PROMPT_MULTI is normally reserved for untrained
# API baselines; this A/B asks whether handing it to the fine-tuned model at
# inference rescues the gap (prompt knob) or whether the gap needs more of the
# model (lora_v5 data-diversity retrain).
#
# Four outputs per set:
#   * _schemaprompt.json          -- schema prompt, no repair
#   * _schemaprompt_repair.json   -- schema prompt + repair_retries=2
# Compared against the committed baseline + repair files.
# GPU-bound: V100 fp16, ~3 s/row base + repair attempts.
# Usage:  sh eval/run_lora_v4_schemaprompt.sh
set -e
cd /data/qiushuogeng/projects/labwright
ADAPTER=lora_v4
ADAPTER_DIR="results/extractor/${ADAPTER}"
echo "== schema-prompt A/B adapter=$ADAPTER at $(date +%H:%M:%S) =="

for GOLD in pk:gold_pk newdomains:gold_new_domains; do
    NAME="${GOLD%%:*}"; FILE="${GOLD##*:}"
    # schema prompt only
    .venv/bin/python -m eval.run_finetuned_benchmark \
        --gold "eval/${FILE}.json" \
        --adapter "${ADAPTER_DIR}" \
        --out "results/eval_finetuned_${NAME}_${ADAPTER}_schemaprompt.json" \
        --multi-block --schema-prompt
    # schema prompt + repair
    .venv/bin/python -m eval.run_finetuned_benchmark \
        --gold "eval/${FILE}.json" \
        --adapter "${ADAPTER_DIR}" \
        --out "results/eval_finetuned_${NAME}_${ADAPTER}_schemaprompt_repair.json" \
        --multi-block --schema-prompt --repair-retries 2
done

echo "== done schema-prompt A/B at $(date +%H:%M:%S) =="
