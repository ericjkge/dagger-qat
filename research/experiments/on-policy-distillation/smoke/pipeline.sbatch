#!/bin/bash
# GPU smoke test of the whole pipeline at the smallest size that runs every step's real code path, on one machine of 3
# GPUs, with Qwen3.5-4B in both roles (the teacher's mechanics are the same at 27B):
#   1. collect: 2 LIBERO episodes, every episode kept (collect.min_score=0: a base 4B seldom succeeds)
#   2. SFT: 2 updates on them, saved also as a Hugging Face model
#   3. distill: 2 updates from the SFT model, after a validation pass, on a trainer, a rollout and a teacher GPU
#   4. eval: the distilled model, exported with PEAR's export script, on the 2 episodes
# Responses are capped at 4096 tokens per episode. check.py verifies every step's outputs and records the run.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-pipeline-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh
export EPISODES=$SMOKE/episodes.json VAL_EPISODES=$SMOKE/episodes.json MAX_RESPONSE=4096 LIBERO_SERVERS=2 TP=1
BASE=$WORK_DIR/models/Qwen3.5-4B
MODEL_FILE=$CODE/configs/models/qwen3_5_4b.sh
# the teacher's chat template at its pinned revision, without the teacher's weights
TEMPLATE=$OUT/tokenizers/Qwen3.8-27B-PEAR-LIBERO_only/chat_template.jinja
RUNS=$OUT/runs/smoke-$SLURM_JOB_ID
mkdir -p "$RUNS"

stage() {  # stage <name> <command...>: runs it, recording its seconds
  local start=$SECONDS
  "${@:2}"
  echo "$1 $((SECONDS - start))" >> "$RUNS/seconds"
}

stage collect env ROLLOUT_GPUS=1 MODEL=$BASE RUN=$RUNS/collect \
  bash scripts/run.sh scripts/collect.sh collect.min_score=0 chat_template_file="$TEMPLATE"
stage sft env MODEL=$BASE EPISODES_DIR=$RUNS/collect/episodes RUN=$RUNS/sft NGPUS=1 \
  bash scripts/sft.sh trainer.total_training_steps=2 data.train_batch_size=2 data.micro_batch_size_per_gpu=1 \
  trainer.logger='[console]'
SFT_MODEL=$RUNS/sft/ckpt/global_step_2/huggingface
stage distill env TRAINER_GPUS=1 ROLLOUT_GPUS=1 TEACHER_GPUS=1 TEACHER_TP=1 STEPS=2 BATCH=2 \
  MODEL_FILE=$MODEL_FILE MODEL=$SFT_MODEL TEACHER=$BASE RUN=$RUNS/distill \
  bash scripts/run.sh scripts/distill.sh chat_template_file="$TEMPLATE"
stage export bash third_party/pear/scripts/export_checkpoint.sh "$RUNS/distill/ckpt/global_step_2" "$RUNS/distilled"
stage eval env ROLLOUT_GPUS=1 MODEL_FILE=$MODEL_FILE MODEL=$RUNS/distilled RUN=$RUNS/eval CKPT_STEP=2 \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE"

"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID"
