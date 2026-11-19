#!/bin/bash
# The repository's test suite (pytest), in a fresh CPU environment built from its requirements.txt.
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/repo_tests-%j.out
set -euo pipefail
REPO=${REPO:-/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/skeleton}
ENV=/n/scratch/users/e/edk005/vla/on-policy-distillation/test-env
module load conda/miniforge3/26.9.4-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla  # its Python 3.11 only: the tests' packages go into a venv of their own
rm -rf "${ENV:?}"
python -m venv "$ENV"
cd "$REPO"
"$ENV/bin/pip" install -q -r requirements.txt
PATH=$ENV/bin:$PATH python -m pytest -v -p no:cacheprovider
