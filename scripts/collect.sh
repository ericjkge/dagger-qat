#!/usr/bin/env bash
# Step 1: runs the teacher over LIBERO's four suites and keeps every successful episode's full multi-round context
# (token ids, which tokens the teacher generated, images) under $RUN/episodes.
#   MODEL=<teacher dir> RUN=<run dir> bash scripts/run.sh scripts/collect.sh [overrides]
source "$(dirname "$0")/common.sh"

MODEL_FILE=$ROOT/configs/models/teacher.sh
pear_args

pear_exec opd.collect "${RUN_ARGS[@]}" trainer.eval_step=0 +data.apply_chat_template_kwargs.reasoning_effort=xhigh "$@"
