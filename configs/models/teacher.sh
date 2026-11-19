# shellcheck shell=bash disable=SC2034
# Sourced: the teacher, Qwen3.8-27B-PEAR-LIBERO_only (PEAR's LIBERO-only RL checkpoint, step 225). PEAR's own model
# file gives the Qwen3.5-family arguments.
# shellcheck source=../../third_party/pear/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh
source "$PEAR/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh"
# after PEAR's file, which names its own base model: scripts/setup.sh reads these two lines
MODEL_REPO=mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only
MODEL_REVISION=63e96d966ad0af83f4eb63b79b94cadb06289bea
