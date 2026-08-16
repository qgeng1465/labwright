#!/usr/bin/env bash
# One-click reproduction of the LabMath-Bench pipeline (reviewer demand #4).
#
# Reproduces, in order:
#   1. the LabMath-Bench gold sets (deterministic, committed seeds),
#   2. the benchmark runs (bare / code_interpreter / labwright core, plus the
#      extension ablation on the stratified subset, plus the adversarial set),
#   3. the TBA / ablation (confusion-matrix → CER) / fail-safe figures,
#   4. the protocol information-flow DAG + the text-overlap render check,
#   5. the supplementary traceability log,
#   6. the test suite + the claims audit (every README number re-derived).
#
# By default it runs a SMOKE reproduction (--limit 5) so anyone can verify the
# pipeline without spending API tokens. Set FULL=1 to reproduce the committed
# full runs (needs LABWRIGHT_API_KEY; each full benchmark costs real tokens).
#
#   FULL=1 ./scripts/reproduce_all.sh        # full committed reproduction
#   ./scripts/reproduce_all.sh               # 5-entry smoke test
#
# Everything deterministic (gold generation, offline analysis, figures, tests,
# audit) always runs in full regardless of FULL. Only the live-LLM benchmark
# loops are gated.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

FULL="${FULL:-0}"
LIMIT="${LIMIT:-5}"
PY="${PY:-.venv/bin/python}"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

if [ ! -x "$PY" ]; then
    echo "no venv at $PY — create one and install requirements.txt first"
    exit 1
fi
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

step "1/6 — regenerate the LabMath-Bench gold sets (committed seeds)"
"$PY" -m eval.make_labmath_bench --seed 20260817 --out eval/gold_labmath_bench.json
"$PY" -m eval.make_gold_new_domains --out eval/gold_new_domains.json
"$PY" -m eval.tag_existing_levels --out eval/gold_labmath_combined.json
"$PY" -m eval.make_labmath_subset --seed 20260817 --out eval/gold_labmath_subset.json

step "2/6 — benchmark runs"
if [ "$FULL" = "1" ]; then
    "$PY" -m eval.run_benchmark \
        --gold eval/gold_labmath_combined.json --systems bare,code_interpreter,labwright \
        --out results/eval_labmath_flash.json
    "$PY" -m eval.run_benchmark \
        --gold eval/gold_labmath_subset.json --systems soft_gate,self_verify,tool_no_gate,labwright_iter \
        --out results/eval_labmath_ext_flash.json
    "$PY" -m eval.run_benchmark \
        --gold eval/gold_labmath_combined.json --systems bare,code_interpreter,labwright \
        --model deepseek-v4-pro --out results/eval_labmath_pro.json
    "$PY" -m eval.run_benchmark \
        --gold eval/gold_labmath_subset.json --systems soft_gate,self_verify,tool_no_gate,labwright_iter \
        --model deepseek-v4-pro --out results/eval_labmath_ext_pro.json
    "$PY" -m eval.run_adversarial --model deepseek-v4-flash --out results/adversarial_flash.json
    "$PY" -m eval.run_adversarial --model deepseek-v4-pro --out results/adversarial_pro.json
else
    echo "  (FULL=0) smoke benchmark on $LIMIT entries each:"
    "$PY" -m eval.run_benchmark --limit "$LIMIT" \
        --gold eval/gold_labmath_combined.json --systems bare,code_interpreter,labwright \
        --out results/repro_smoke_flash.json
    "$PY" -m eval.run_benchmark --limit "$LIMIT" \
        --gold eval/gold_labmath_subset.json --systems soft_gate,self_verify,tool_no_gate,labwright_iter \
        --out results/repro_smoke_ext.json
    "$PY" -m eval.run_adversarial --model deepseek-v4-flash --limit "$LIMIT" \
        --out results/repro_smoke_adversarial.json
fi

step "3/6 — TBA, ablation (CER) and fail-safe figures"
if [ -f results/eval_labmath_pro.json ]; then
    "$PY" paper/fig_tba.py results/eval_labmath_flash.json results/eval_labmath_pro.json
    "$PY" paper/fig_ablation.py results/eval_labmath_flash.json results/eval_labmath_pro.json
else
    echo "  (pro results absent — rendering flash-only smoke panels)"
    "$PY" paper/fig_tba.py results/eval_labmath_flash.json results/repro_smoke_flash.json
    "$PY" paper/fig_ablation.py results/eval_labmath_flash.json results/repro_smoke_flash.json
fi
if [ -f results/adversarial_pro.json ]; then
    "$PY" paper/fig_failsafe.py results/adversarial_flash.json results/adversarial_pro.json
else
    "$PY" paper/fig_failsafe.py results/adversarial_flash.json results/adversarial_flash.json
fi

step "4/6 — protocol DAG + render QA"
"$PY" paper/fig_protocol_dag.py
"$PY" paper/_check_render.py

step "5/6 — supplementary traceability log"
"$PY" -m eval.make_traceability_log \
    --results results/eval_labmath_flash.json results/eval_labmath_pro.json \
    --out supplementary/traceability

step "6/6 — test suite + claims audit"
"$PY" -m pytest -q
"$PY" -m eval.audit_claims

echo
echo "reproduction complete. Committed full results are untouched; smoke runs land in results/repro_smoke_*.json"
