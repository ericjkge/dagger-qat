#!/bin/bash
# GPU smoke test, part 1 of 2, on one GPU: PEAR's checks, then every step but distillation at the smallest size that runs
# its real code path:
#   collect 2 LIBERO episodes, all kept (collect.min_score=0: a base 4B seldom succeeds); SFT 2 updates on them, saved
#   also as a Hugging Face model (FSDP2's offload policy keeps its fp32 parameters, gradients and optimizer state, about
#   64 GB for a 4B, on the CPU); evaluate that model on the 2 episodes. check.py verifies each step's outputs.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5: a short, restartable job
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G  # the SFT stage's fp32 parameters, gradients and Adam state (about 64 GB) live here
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-part1-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh

# PEAR's checks of the training environment on this GPU and of a LIBERO step (setup.sh, its install already done)
stage check env CUDA_ARCH=8.9 BENCHES=libero MODEL_FILES=configs/models/qwen3_5_4b.sh bash scripts/setup.sh
stage collect env MODEL=$BASE RUN=$RUNS/collect \
  bash scripts/run.sh scripts/collect.sh collect.min_score=0 chat_template_file="$TEMPLATE" "$UNITS"
stage sft env MODEL=$BASE EPISODES_DIR=$RUNS/collect/episodes RUN=$RUNS/sft NGPUS=1 \
  bash scripts/sft.sh trainer.total_training_steps=2 data.train_batch_size=2 data.micro_batch_size_per_gpu=1 \
  engine.offload_policy=true trainer.logger='[console]'
stage eval_sft env MODEL_FILE=$STUDENT_FILE MODEL=$SFT_MODEL RUN=$RUNS/eval-sft \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE" "$UNITS"

"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID" collect sft eval-sft
