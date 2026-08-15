#!/bin/bash
# v6 training watchdog (61k rows, natural-register core): auto-resume from the latest HF checkpoint.
#
# On 2026-08-15 the first lora_v5 run died (same guard for v6) silently at step 550/4206 (whole
# setsid group killed, no traceback; cause not readable without system logs).
# save_strategy="epoch" had produced no checkpoint yet, so the loss was total.
# This watchdog relaunches the same training up to MAX_RESTARTS times, each
# retry resuming from the newest results/extractor/lora_v6/checkpoint-* dir
# (train.py --resume), so a repeat kill costs at most one epoch of work
# instead of the whole run. A clean exit (rc==0, adapter saved) stops the loop.
#
# Run under arbitrate:  python3 <arbitrate.py> run \
#     --name labwright-lora-v6 --gpu-mem 20 --cpu 4 --ram 16 --detach -- \
#     bash eval/run_lora_v5_train_watchdog.sh
# Progress: /tmp/lora_v6_train.log (appended across restarts);
#           /tmp/lora_v6_watchdog.status (machine-checkable).

set -u
cd /data/qiushuogeng/projects/labwright
OUT=results/extractor/lora_v6
LOG=/tmp/lora_v6_train.log
STATUS=/tmp/lora_v6_watchdog.status
MAX_RESTARTS=3

run_once() {
    local attempt=$1
    local resume_arg=""
    local ckpt
    if [ "$attempt" -gt 0 ]; then
        ckpt=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "$ckpt" ]; then
            resume_arg="--resume $ckpt"
            echo "[watchdog] resume from $ckpt" | tee -a "$LOG"
        else
            echo "[watchdog] no checkpoint to resume (full restart)" | tee -a "$LOG"
        fi
    fi
    echo "[watchdog] attempt $attempt starting at $(date '+%H:%M:%S')" | tee -a "$LOG"
    echo "attempt=$attempt start=$(date '+%s')" > "$STATUS"
    # shellcheck disable=SC2086
    # steps-based checkpoints: a silent kill costs at most ~save_steps of work
    # and there is always a resumable checkpoint on disk.
    HF_HUB_OFFLINE=1 .venv/bin/python -m labwright.extract.train \
        --data results/extractor_11dom_v4 \
        --out "$OUT" \
        --max-len 2048 --multi-block \
        --save-steps 250 --save-total-limit 2 \
        --log results/extractor/train_61k_v4.log \
        $resume_arg >> "$LOG" 2>&1
    local rc=$?
    echo "[watchdog] attempt $attempt exited rc=$rc at $(date '+%H:%M:%S')" | tee -a "$LOG"
    echo "attempt=$attempt end=$(date '+%s') rc=$rc" > "$STATUS"
    return $rc
}

attempt=0
while [ "$attempt" -le "$MAX_RESTARTS" ]; do
    run_once "$attempt"
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$OUT/adapter_config.json" ]; then
        echo "[lora-v6-done] adapter saved to $OUT" | tee -a "$LOG"
        echo "done=1 adapter=$OUT" > "$STATUS"
        exit 0
    fi
    if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
        echo "[lora-v6-failed] after $MAX_RESTARTS restarts" | tee -a "$LOG"
        echo "done=0 failed=1 restarts=$attempt" > "$STATUS"
        exit 1
    fi
    attempt=$((attempt + 1))
done
