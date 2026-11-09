#!/usr/bin/env bash
#SBATCH --job-name=dqat-plus-mig
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=data/logs/plus-mig-workers-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
experiment="${1:?Usage: cannon_plus_mig_workers.sbatch EXPERIMENT_ROOT}"
# gpu_test caps submitted jobs as well as running jobs at two per user.
case "${SLURM_ARRAY_TASK_ID:?Launch as array 0-1%2}" in
  0) indices=(0 2 4) ;; # AWQ, learner GPTQ, rollout-control GPTQ
  1) indices=(1 3) ;;   # Offline GPTQ, BF16
  *) echo 'Expected worker 0 or 1' >&2; exit 2 ;;
esac
for index in "${indices[@]}"; do
  SLURM_ARRAY_TASK_ID="$index" bash scripts/cannon_plus_mig.sbatch "$experiment"
done
