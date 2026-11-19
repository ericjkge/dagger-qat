# shellcheck shell=bash disable=SC2034
# Sourced: the student Qwen3.5-9B. Same architecture family as the teacher (qwen3_5), so PEAR's model file applies;
# a 9B model needs no tensor parallelism, so one replica per GPU.
TP=${TP:-1}
# shellcheck source=../../third_party/pear/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh
source "$PEAR/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh"
# after PEAR's file, which names its own base model: scripts/setup.sh reads these two lines
MODEL_REPO=Qwen/Qwen3.5-9B
MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
