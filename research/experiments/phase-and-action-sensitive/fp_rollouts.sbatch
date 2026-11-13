#!/usr/bin/env bash
#SBATCH --job-name=vla-fp-rollouts
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/fp-rollouts-%j.out
# Download the pinned OFT LIBERO-10 checkpoint (if absent), then run full-precision
# rollouts on the held-out frame-collection states and tag phases. Submit from the worktree root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
OUT=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive
CKPT=$OUT/checkpoints/openvla-oft-libero-10
export HF_HOME=$OUT/cache/huggingface LIBERO_CONFIG_PATH=$OUT/libero-config
export PYTHONPATH=$PWD/src:$PWD/third_party/openvla-oft:$PWD/third_party/LIBERO
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=7 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load conda/miniforge3/26.9.4-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla
nvidia-smi
if [[ ! -f $CKPT/source.json ]]; then
python - <<PY
import json
from pathlib import Path
from huggingface_hub import snapshot_download
mid, rev = "moojink/openvla-7b-oft-finetuned-libero-10", "95220f9a3421a7ff12d4218e73d09ade830fa9a3"
snapshot_download(mid, revision=rev, local_dir="$CKPT", ignore_patterns=["lora_adapter/*"], max_workers=4)
Path("$CKPT/source.json").write_text(json.dumps({"model_id": mid, "model_revision": rev}, indent=2) + "\n")
PY
fi
python research/experiments/phase-and-action-sensitive/fp_rollouts.py \
  --checkpoint "$CKPT" --out "$OUT/fp-rollouts/$SLURM_JOB_ID" \
  --suite libero_10 --tasks 0 1 2 3 4 5 6 7 8 9 --init-states 0 1 2 3 4 --seed 7 \
  --max-steps 520 --settle-steps 10 --open-loop-steps 8 \
  --phase-window 10 --flip-min-hold 5
