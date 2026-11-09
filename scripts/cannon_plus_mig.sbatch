#!/usr/bin/env bash
#SBATCH --job-name=dqat-plus-mig
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=data/logs/plus-mig-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/common.sh
export OMP_NUM_THREADS=4 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4
export DQAT_OSMESA_LIBRARY="$DQAT_ROOT/data/osmesa-runtime/usr/lib64/libOSMesa.so.8.0.0"
export LD_LIBRARY_PATH="$DQAT_ROOT/data/osmesa-runtime/usr/lib64:/usr/lib64/llvm17/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBERO_CONFIG_PATH="$DQAT_DATA_DIR/libero-plus-config"
export PYTHONPATH="$DQAT_ROOT/data/libero-plus-deps:$PYTHONPATH"
experiment="${1:?Usage: cannon_plus_mig.sbatch EXPERIMENT_ROOT}"
arms=(awq offline_gptq student fp expert_control)
arm="${arms[${SLURM_ARRAY_TASK_ID:?Launch as array 0-4%2}]}"
args=(--checkpoint "$DQAT_DATA_DIR/checkpoints/openvla-oft-libero-10"
      --plus-root "$DQAT_ROOT/data/vendor/LIBERO-plus-4976dc30028e805ff8094b55501d532c48fec182"
      --manifest configs/libero-plus-long-pilot-v1.json --arm "$arm"
      --rapid --software-rendering --shard 0 --num-shards 1
      --output "$experiment/$arm/shard-0")
case "$arm" in
  fp) ;;
  awq) args+=(--cache "$DQAT_DATA_DIR/calibration/awq-w4g128-32frames-seed7.pt"
              --source-run "${DQAT_AWQ_RUN:?Set completed AWQ run.json}") ;;
  offline_gptq) args+=(--cache "$DQAT_DATA_DIR/calibration/gptq-w4g128-32frames-seed7.pt"
                       --source-run "${DQAT_BASELINE_GPTQ_RUN:?Set completed GPTQ run.json}") ;;
  student|expert_control) args+=(--cache "${DQAT_LOOP_EXPERIMENT:?Set completed nominal loop experiment}/$arm/round-2/gptq.pt"
                                --source-run "$DQAT_LOOP_EXPERIMENT/$arm/round-2/evaluation/run.json") ;;
esac
exec "$DQAT_PYTHON" -m dqat.eval_plus "${args[@]}"
