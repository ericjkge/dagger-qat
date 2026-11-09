#!/usr/bin/env bash
#SBATCH --job-name=dqat-plus
#SBATCH --partition=gpu,gpu_requeue
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=data/logs/plus-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/common.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$DQAT_DATA_DIR/libero-plus-config"
export PYTHONPATH="$DQAT_ROOT/data/libero-plus-deps:$PYTHONPATH"
plus_root="$DQAT_ROOT/data/vendor/LIBERO-plus-4976dc30028e805ff8094b55501d532c48fec182"
experiment="${1:?Usage: cannon_plus.sbatch EXPERIMENT_ROOT smoke|array}"
mode="${2:?Missing mode}"
manifest="${DQAT_PLUS_MANIFEST:-configs/libero-plus-long-pilot-v1.json}"
args=(--checkpoint "$DQAT_DATA_DIR/checkpoints/openvla-oft-libero-10"
      --plus-root "$plus_root" --manifest "$manifest")
if [[ "$mode" == smoke ]]; then
  arm=fp
  args+=(--smoke --shard 0 --num-shards 1 --output "$experiment/smoke")
elif [[ "$mode" == array ]]; then
  index="${SLURM_ARRAY_TASK_ID:?Must be launched as an array}"
  arms=(fp awq offline_gptq student expert_control)
  arm="${arms[$((index / 4))]}"
  shard=$((index % 4))
  args+=(--shard "$shard" --num-shards 4 --output "$experiment/$arm/shard-$shard")
else
  echo 'Expected smoke or array' >&2; exit 2
fi
case "$arm" in
  fp) ;;
  awq) args+=(--cache "$DQAT_DATA_DIR/calibration/awq-w4g128-32frames-seed7.pt"
              --source-run "${DQAT_AWQ_RUN:?Set completed AWQ run.json}") ;;
  offline_gptq) args+=(--cache "$DQAT_DATA_DIR/calibration/gptq-w4g128-32frames-seed7.pt"
                       --source-run "${DQAT_BASELINE_GPTQ_RUN:?Set completed GPTQ run.json}") ;;
  student|expert_control) args+=(--cache "${DQAT_LOOP_EXPERIMENT:?Set completed nominal loop experiment}/$arm/round-2/gptq.pt"
                                --source-run "$DQAT_LOOP_EXPERIMENT/$arm/round-2/evaluation/run.json") ;;
esac
exec "$DQAT_PYTHON" -m dqat.eval_plus --arm "$arm" "${args[@]}"
