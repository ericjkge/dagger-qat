#!/usr/bin/env bash
#SBATCH --job-name=dqat-gptq-loop
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/loop-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
nvidia-smi
bash scripts/run_onpolicy.sh "$@"
