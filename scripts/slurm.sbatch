#!/usr/bin/env bash
# A pipeline step as one Slurm job on HMS O2: starts a Ray cluster over the job's machines, runs the step's script on
# the head with GPU=o2 and the cluster's sizes, then stops Ray.
#   sbatch [--gres=gpu:l40s:4] [--time=...] scripts/slurm.sbatch scripts/eval.sh [overrides]
#   sbatch -p gpu_mpi_quad --nodes=3 --export=ALL,ROLES="trainer rollout teacher" scripts/slurm.sbatch scripts/distill.sh
# ROLES gives each machine's role in node order (trainer | rollout | teacher; default: all rollout). On one machine the
# roles split its 4 GPUs instead: TRAINER_GPUS, ROLLOUT_GPUS and TEACHER_GPUS.
#
# Priority: O2's fairshare bills each job by what it requests (5 per GPU, 1 per CPU, 1 per 16 GB), and O2 flags users
# whose jobs use under half the memory they request. Keep --time, --cpus-per-task and --mem near a run's measured use
# (sacct -j <job> -o Elapsed,MaxRSS,TotalCPU) rather than at these defaults. Short, resumable runs can go to
# gpu_requeue, billed at a fiftieth of gpu_quad but preemptible.
#SBATCH --job-name=opd
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

STEP=${1:?the step script, e.g. scripts/eval.sh}
shift
source "$(dirname "$STEP")/common.sh"
export GPU=o2
# shellcheck source=/dev/null
source "$PEAR/scripts/gpu_process_variables.sh" "$VERL_ENV"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
read -r -a ROLE_LIST <<< "${ROLES:-$(printf 'rollout %.0s' "${NODES[@]}")}"
[ "${#ROLE_LIST[@]}" -eq "${#NODES[@]}" ] || { echo "ROLES names ${#ROLE_LIST[@]} roles for ${#NODES[@]} machines"; exit 1; }
HEAD=${NODES[0]}
PORT=6379
LIBERO_SERVERS=${LIBERO_SERVERS:-16}  # simulator processes per rollout machine, rendering on its CPUs

# the Ray resources a machine offers: PEAR places the actor on "trainer", the replicas on "rollout", the simulators on
# "libero"; the teacher's pool takes GPUs no label claims
resources() {
  if [ "${#NODES[@]}" -eq 1 ]; then
    echo "{\"trainer\": ${TRAINER_GPUS:-0}, \"rollout\": ${ROLLOUT_GPUS:-4}, \"libero\": $LIBERO_SERVERS}"
  else
    case $1 in
      trainer) echo '{"trainer": 4}' ;;
      rollout) echo "{\"rollout\": 4, \"libero\": $LIBERO_SERVERS}" ;;
      teacher) echo '{}' ;;
    esac
  fi
}

ray_start() {  # <machine> <role> [--head | --address=...]
  srun --nodes=1 --ntasks=1 -w "$1" bash -c "RAY_TMPDIR=$LOCAL_DIR/ray PYTHONPATH=$PYTHON_PATH \
    $VERL_ENV/bin/ray start ${3:-} --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --resources='$(resources "$2")' --block" &
}

mkdir -p "$LOCAL_DIR"
ray_start "$HEAD" "${ROLE_LIST[0]}" "--head --port=$PORT"
sleep 20
for i in "${!NODES[@]}"; do
  [ "$i" -eq 0 ] || ray_start "${NODES[$i]}" "${ROLE_LIST[$i]}" "--address=$HEAD:$PORT"
done
export RAY_ADDRESS=$HEAD:$PORT
until [ "$("$VERL_ENV/bin/python" -c 'import ray; ray.init(); print(sum(n["Alive"] for n in ray.nodes()))' 2>/dev/null)" = "${#NODES[@]}" ]; do
  sleep 10
done

count() { local n=0 r; for r in "${ROLE_LIST[@]}"; do [ "$r" = "$1" ] && n=$((n + 1)); done; echo $((n > 0 ? n : 1)); }
export ROLLOUT_NNODES=$(count rollout) TRAINER_NNODES=$(count trainer) TEACHER_NNODES=$(count teacher)
trap '"$VERL_ENV/bin/ray" stop --force' EXIT
bash "$STEP" "$@"
