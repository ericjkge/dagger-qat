#!/bin/bash
# GPU smoke test, part 2 of 2, after part 1, on three GPUs:
#   SFT 2 updates on part 1's episodes, on two GPUs (FSDP2 shards a 4B's fp32 parameters and gradients, and the
#   optimizer state comes onto the GPUs only for the update), saved also as a Hugging Face model; evaluate it on the 2
#   episodes;
#   distill from it, 2 updates after a validation pass, on a trainer, a rollout and a teacher GPU; export the result
#   with PEAR's script and evaluate it on the 2 episodes.
# check.py verifies each step's outputs.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5: a short, restartable job
#SBATCH --gres=gpu:l40s:3
#SBATCH --time=01:45:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-part2-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh

stage sft env MODEL=$BASE EPISODES_DIR=$RUNS/collect/episodes RUN=$RUNS/sft NGPUS=2 \
  bash scripts/sft.sh trainer.total_training_steps=2 data.train_batch_size=2 data.micro_batch_size_per_gpu=1 \
  engine.optimizer_offload=true trainer.logger='[console]'
stage eval_sft env ROLLOUT_GPUS=1 MODEL_FILE=$STUDENT_FILE MODEL=$SFT_MODEL RUN=$RUNS/eval-sft \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE" "$UNITS"
stage distill env TRAINER_GPUS=1 ROLLOUT_GPUS=1 TEACHER_GPUS=1 TEACHER_TP=1 STEPS=2 BATCH=2 \
  MODEL_FILE=$STUDENT_FILE MODEL=$SFT_MODEL TEACHER=$BASE RUN=$RUNS/distill \
  bash scripts/run.sh scripts/distill.sh chat_template_file="$TEMPLATE" "$UNITS"
stage export bash third_party/pear/scripts/export_checkpoint.sh "$RUNS/distill/ckpt/global_step_2" "$RUNS/distilled"
stage eval_distilled env ROLLOUT_GPUS=1 MODEL_FILE=$STUDENT_FILE MODEL=$RUNS/distilled RUN=$RUNS/eval-distilled CKPT_STEP=2 \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE" "$UNITS"

"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID" sft eval-sft distill eval-distilled
