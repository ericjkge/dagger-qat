#!/usr/bin/env bash
# One-time install on a GPU machine: PEAR's environments under WORK_DIR (verl, LIBERO and its perturbed variants,
# Mesa), then the teacher's and the students' weights at their pinned revisions.
#   WORK_DIR=<a directory every node sees> bash scripts/setup.sh
source "$(dirname "$0")/common.sh"

git -C "$ROOT" submodule update --init
for target in libero_run libero_plus libero_pro; do
  bash "$PEAR/scripts/install.sh" "$target"
done
for model in "$ROOT"/configs/models/*.sh; do
  MODEL_FILE=$model bash "$PEAR/scripts/download_model.sh"
done
