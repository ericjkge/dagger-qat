#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
method="${1:?Usage: run_baseline.sh fp|awq|gptq [evaluation options]}"
shift
case "$method" in fp|awq|gptq) ;; *) echo 'Expected fp, awq or gptq' >&2; exit 2;; esac
run_id="${DQAT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}}"
args=(--checkpoint "$DQAT_DATA_DIR/checkpoints/openvla-oft-libero-10"
      --method "$method" --output "$DQAT_DATA_DIR/results/$run_id-$method")
if [[ "$method" != fp ]]; then
  args+=(--dataset-dir "$DQAT_DATA_DIR/datasets/libero_10"
         "--$method-cache" "$DQAT_DATA_DIR/calibration/$method-w4g128-32frames-seed7.pt")
fi
exec "$DQAT_PYTHON" -m dqat.eval_libero "${args[@]}" "$@"
