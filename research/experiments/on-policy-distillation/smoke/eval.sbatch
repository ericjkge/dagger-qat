#!/bin/bash
# Smoke test of steps 4 and 1 on one GPU: the base 4B student evaluated on 4 LIBERO episodes (scripts/eval.sh), then the
# same episodes through the collector (scripts/collect.sh, the 4B in the teacher's place), each through scripts/run.sh's
# one-machine Ray cluster. A short response budget bounds the time; the checks are in check.py.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-eval-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh
export EPISODES=$SMOKE/episodes.json MAX_RESPONSE=16384 LIBERO_SERVERS=4 TP=1
STUDENT=$WORK_DIR/models/Qwen3.5-4B
# the teacher's chat template at its pinned revision, without downloading the teacher's weights
TEMPLATE=$OUT/tokenizers/Qwen3.8-27B-PEAR-LIBERO_only/chat_template.jinja
RUNS=$OUT/runs/smoke-$SLURM_JOB_ID
mkdir -p "$RUNS"

SECONDS=0
MODEL_FILE=configs/models/qwen3_5_4b.sh MODEL=$STUDENT RUN=$RUNS/eval \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE"
echo "eval $SECONDS" >> "$RUNS/seconds"
SECONDS=0
MODEL=$STUDENT RUN=$RUNS/collect bash scripts/run.sh scripts/collect.sh
echo "collect $SECONDS" >> "$RUNS/seconds"
"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID"
