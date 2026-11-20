# Sourced by the smoke jobs: the code they run (the repository's main line at CODE_SHA, checked out on scratch) and
# the shared work directory (environments, models, runs).
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE_SHA=${CODE_SHA:?the commit of the code to run}
CODE=$OUT/code/$CODE_SHA
export WORK_DIR=$OUT/work
SMOKE=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke
if [ ! -d "$CODE" ]; then
  git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"
  git -C "$CODE" checkout -q "$CODE_SHA"
  git -C "$CODE" submodule update -q --init
fi
cd "$CODE"
echo "code $(git rev-parse HEAD) on $(hostname): $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null || echo no GPU)"

# the smoke test's shared settings: 2 episodes, 4096-token responses, Qwen3.5-4B in both roles (the teacher's mechanics
# are the same at 27B) with the teacher's chat template at its pinned revision (without the teacher's weights)
export EPISODES=$SMOKE/episodes.json VAL_EPISODES=$SMOKE/episodes.json MAX_RESPONSE=4096 LIBERO_SERVERS=2 TP=1
BASE=$WORK_DIR/models/Qwen3.5-4B
STUDENT_FILE=$CODE/configs/models/qwen3_5_4b.sh
TEMPLATE=$OUT/tokenizers/Qwen3.8-27B-PEAR-LIBERO_only/chat_template.jinja
RUNS=$OUT/runs/smoke-${CODE_SHA:0:7}
SFT_MODEL=$RUNS/sft/ckpt/global_step_2/huggingface
mkdir -p "$RUNS"

stage() {  # stage <name> <command...>: runs it, recording its seconds in $RUNS/seconds
  local start=$SECONDS
  "${@:2}"
  echo "$1 $((SECONDS - start))" >> "$RUNS/seconds"
}
