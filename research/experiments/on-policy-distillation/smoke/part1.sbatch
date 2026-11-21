#!/bin/bash
# GPU smoke test, part 1 of 2, on one GPU: PEAR's checks, then teacher-episode collection at the smallest size that runs
# its real code path: 2 LIBERO episodes, all kept (collect.min_score=0: a base 4B seldom succeeds). check.py verifies
# its outputs.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5: a short, restartable job
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-part1-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh

# PEAR's checks of the training environment on this GPU and of a LIBERO step (setup.sh, its install already done)
stage check env CUDA_ARCH=8.9 BENCHES=libero MODEL_FILES=configs/models/qwen3_5_4b.sh bash scripts/setup.sh
stage collect env MODEL=$BASE RUN=$RUNS/collect \
  bash scripts/run.sh scripts/collect.sh collect.min_score=0 chat_template_file="$TEMPLATE" "$UNITS"

"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID" collect
