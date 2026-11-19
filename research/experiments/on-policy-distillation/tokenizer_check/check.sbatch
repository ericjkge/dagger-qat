#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/tokenizer_check-%j.out
set -euo pipefail
module load conda/miniforge3/26.9.4-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla
REPO=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation
python "$REPO/research/experiments/on-policy-distillation/tokenizer_check/check.py" \
  /n/scratch/users/e/edk005/vla/on-policy-distillation/tokenizers "$REPO/research/runs/on-policy-distillation.jsonl"
