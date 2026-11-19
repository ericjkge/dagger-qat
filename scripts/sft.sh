#!/usr/bin/env bash
# Step 2: SFT of a student on the collected episodes, on the GPUs of this machine.
#   WORK_DIR=... MODEL=<student dir> EPISODES_DIR=<collect run>/episodes RUN=<run dir> [NGPUS=8] bash scripts/sft.sh
source "$(dirname "$0")/common.sh"

MODEL=${MODEL:?the student model directory}
EPISODES_DIR=${EPISODES_DIR:?the episodes directory scripts/collect.sh wrote}
RUN=$(readlink -m "${RUN:?the run directory}")
mkdir -p "$RUN"

# shellcheck source=/dev/null
source "$PEAR/scripts/gpu_process_variables.sh" "$VERL_ENV"
export PYTHONPATH=$PYTHON_PATH
torchrun --standalone --nproc_per_node="${NGPUS:-8}" -m opd.sft \
  data.train_files="$EPISODES_DIR" \
  model.path="$MODEL" \
  trainer.default_local_dir="$RUN/ckpt" \
  trainer.experiment_name="$(basename "$RUN")" \
  hydra.run.dir="$RUN/hydra" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
