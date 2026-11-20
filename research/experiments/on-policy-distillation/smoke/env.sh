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
