#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
method="${1:?Usage: run_baseline.sh fp|awq [evaluation options]}"
shift
case "$method" in fp|awq) ;; *) echo 'Expected fp or awq' >&2; exit 2;; esac
run_id="${DQAT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}}"
args=(--checkpoint "$DQAT_DATA_DIR/checkpoints/openvla-oft-libero-10"
      --method "$method" --output "$DQAT_DATA_DIR/results/$run_id-$method")
if [[ "$method" == awq ]]; then
  args+=(--dataset-dir "$DQAT_DATA_DIR/datasets/libero_10"
         --awq-cache "$DQAT_DATA_DIR/calibration/awq-w4g128-32frames-seed7.pt")
fi
exec "$DQAT_PYTHON" -m dqat.eval_libero "${args[@]}" "$@"
