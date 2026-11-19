#!/usr/bin/env bash
# Runs a Ray step of the pipeline (collect.sh, distill.sh, eval.sh) with its overrides, on one of:
#   - the Ray cluster at RAY_ADDRESS, started with scripts/ray_node.sh on every machine, with ROLLOUT_NNODES,
#     TRAINER_NNODES and TEACHER_NNODES set to its machines of each role;
#   - a Slurm allocation: a cluster over its machines, ROLES naming each machine's role in node order;
#   - otherwise this machine alone, every role on it (scripts/ray_node.sh all).
#   bash scripts/run.sh scripts/eval.sh [overrides]
#   sbatch --nodes=3 --gpus-per-node=8 --export=ALL,ROLES="trainer rollout teacher" scripts/run.sh scripts/distill.sh
STEP_SCRIPT=${1:?a step script: scripts/collect.sh, scripts/distill.sh or scripts/eval.sh}
shift
# by the step's path: under sbatch this file runs from a copy
SCRIPTS=$(cd "$(dirname "$STEP_SCRIPT")" && pwd -P)
source "$SCRIPTS/common.sh"

if [ -n "${RAY_ADDRESS:-}" ]; then
  exec bash "$STEP_SCRIPT" "$@"
fi

PORT=${PORT:-6379}
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
  NODES=("$(hostname)")
fi
if [ "${#NODES[@]}" -eq 1 ]; then
  ROLE_LIST=(all)
else
  read -r -a ROLE_LIST <<< "${ROLES:?ROLES: one role per machine of the allocation}"
  [ "${#ROLE_LIST[@]}" -eq "${#NODES[@]}" ] || { echo "ROLES names ${#ROLE_LIST[@]} roles for ${#NODES[@]} machines" >&2; exit 1; }
fi

# start_node <index> [head address]: the machine's Ray node, in the background
start_node() {
  if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    srun --nodes=1 --ntasks=1 --overlap -w "${NODES[$1]}" bash "$SCRIPTS/ray_node.sh" "${ROLE_LIST[$1]}" "${2:-}" &
  else
    bash "$SCRIPTS/ray_node.sh" "${ROLE_LIST[$1]}" "${2:-}" &
  fi
}

HEAD=$(getent hosts "${NODES[0]}" | awk '{print $1; exit}')
start_node 0
trap '"$VERL_ENV/bin/ray" stop --force > /dev/null; wait' EXIT
export RAY_ADDRESS=$HEAD:$PORT
until "$VERL_ENV/bin/ray" status --address="$RAY_ADDRESS" > /dev/null 2>&1; do sleep 5; done
for i in "${!NODES[@]}"; do
  [ "$i" -eq 0 ] || start_node "$i" "$RAY_ADDRESS"
done
until [ "$("$VERL_ENV/bin/python" -c 'import ray; ray.init(); print(sum(n["Alive"] for n in ray.nodes()))' 2> /dev/null)" = "${#NODES[@]}" ]; do
  sleep 5
done

count() {  # the machines of a role, at least 1: on one machine every role has it
  local n=0 role
  for role in "${ROLE_LIST[@]}"; do [ "$role" = "$1" ] && n=$((n + 1)); done
  echo $((n > 0 ? n : 1))
}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(count rollout)}
export TRAINER_NNODES=${TRAINER_NNODES:-$(count trainer)}
export TEACHER_NNODES=${TEACHER_NNODES:-$(count teacher)}
bash "$STEP_SCRIPT" "$@"
