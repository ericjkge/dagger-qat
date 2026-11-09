#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
phase="${1:?Usage: run_onpolicy.sh prepare|evaluate student|expert_control 1|2 EXPERIMENT_ROOT}"
arm="${2:?Missing arm}"
iteration="${3:?Missing iteration}"
experiment="${4:?Missing experiment root}"
case "$phase" in prepare|evaluate) ;; *) echo 'Expected prepare or evaluate' >&2; exit 2;; esac
case "$arm" in student|expert_control) ;; *) echo 'Expected student or expert_control' >&2; exit 2;; esac
case "$iteration" in 1|2) ;; *) echo 'Expected iteration 1 or 2' >&2; exit 2;; esac
round_dir="$experiment/$arm/round-$iteration"
checkpoint="$DQAT_DATA_DIR/checkpoints/openvla-oft-libero-10"
dataset="$DQAT_DATA_DIR/datasets/libero_10"
if [[ "$phase" == prepare ]]; then
  args=(--checkpoint "$checkpoint" --dataset-dir "$dataset" --arm "$arm"
        --round "$iteration" --output "$round_dir")
  if [[ "$iteration" == 2 ]]; then
    args+=(--previous-collection "$experiment/$arm/round-1/collection")
  fi
  if [[ "$arm" == student ]]; then
    if [[ "$iteration" == 1 ]]; then
      args+=(--actor-cache "$DQAT_DATA_DIR/calibration/gptq-w4g128-32frames-seed7.pt"
             --actor-run "${DQAT_BASELINE_GPTQ_RUN:?Set the completed baseline GPTQ run.json path}")
    else
      args+=(--actor-cache "$experiment/$arm/round-1/gptq.pt"
             --actor-run "$experiment/$arm/round-1/validation/run.json")
    fi
  fi
  "$DQAT_PYTHON" -m dqat.collect_libero "${args[@]}"
  output="$round_dir/validation"
  extra=(--tasks 0 --episodes 1)
else
  [[ "$iteration" == 2 ]] || { echo 'Primary evaluation is pre-set to round 2' >&2; exit 2; }
  [[ -f "$round_dir/gptq.pt" ]] || { echo 'Validated GPTQ cache is required' >&2; exit 2; }
  output="$round_dir/evaluation"
  extra=()
fi
# Separate process reloads the original BF16 checkpoint before every calibration.
exec "$DQAT_PYTHON" -m dqat.eval_libero --checkpoint "$checkpoint" --method gptq \
  --dataset-dir "$dataset" --gptq-cache "$round_dir/gptq.pt" \
  --calibration-bundle "$round_dir/bundle" --output "$output" "${extra[@]}"
