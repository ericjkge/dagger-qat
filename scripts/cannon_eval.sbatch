#!/usr/bin/env bash
#SBATCH --job-name=dqat-libero-long
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=data/logs/eval-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
nvidia-smi
bash scripts/run_baseline.sh "$@"
