#!/usr/bin/env bash
# Step 4: evaluates a model on a benchmark's episode file, by default LIBERO's 2,000 (40 tasks x 50 initial
# states); EPISODES=<file> evaluates a subset, e.g. ~10 episodes per task of a perturbation.
# Runs on the Ray head of a cluster whose GPU machines all started as rollout ($PEAR/scripts/start_ray_node.sh).
#   WORK_DIR=... GPU=h100 ROLLOUT_NNODES=<n> MODEL_FILE=configs/models/<model>.sh MODEL=<model dir> \
#     RUN=<run dir> [STEP=<checkpoint step>] [BENCH=libero|libero_plus|libero_pro] bash scripts/eval.sh [overrides]
# The teacher's effort is an override: +data.apply_chat_template_kwargs.reasoning_effort=xhigh (or low).
source "$(dirname "$0")/common.sh"

pear_args

pear_exec opd.eval "${RUN_ARGS[@]}" trainer.eval_step="${STEP:-0}" "$@"
