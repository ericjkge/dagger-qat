#!/usr/bin/env bash
#SBATCH --job-name=dqat-setup
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=data/logs/setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cmake/3.28.3-fasrc01
export PATH="$HOME/vla-calibration/data/tools:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash scripts/setup.sh
