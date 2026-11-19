# shellcheck shell=bash disable=SC2034
# Sourced: the student Qwen3.5-4B. Same architecture family as the teacher (qwen3_5), so PEAR's model file applies;
# a 4B model needs no tensor parallelism, so one replica per GPU.
TP=${TP:-1}
# shellcheck source=../../third_party/pear/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh
source "$PEAR/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh"
# after PEAR's file, which names its own base model: scripts/setup.sh reads these two lines
MODEL_REPO=Qwen/Qwen3.5-4B
MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
