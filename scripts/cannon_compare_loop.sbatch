#!/usr/bin/env bash
#SBATCH --job-name=dqat-compare-loop
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=data/logs/compare-loop-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/common.sh
experiment="${1:?Provide the experiment root}"
baseline_run="${DQAT_BASELINE_GPTQ_RUN:?Set the completed baseline GPTQ run.json path}"
exec "$DQAT_PYTHON" -m dqat.compare_calibration --baseline "$(dirname "$baseline_run")" \
  --student "$experiment/student/round-2/evaluation" \
  --expert-control "$experiment/expert_control/round-2/evaluation" \
  --output "$experiment/comparison.json"
