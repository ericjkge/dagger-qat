#!/usr/bin/env bash
#SBATCH --job-name=vla-demo-frames
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/demo-frames-%j.out
# Download the pinned OFT LIBERO-10 checkpoint and the OFT fine-tuning RLDS demos
# (if absent), tag phases on the demos and sample the frame set. CPU only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
OUT=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive
CKPT=$OUT/checkpoints/openvla-oft-libero-10
RLDS=$OUT/datasets/modified_libero_rlds
export HF_HOME=$OUT/cache/huggingface TF_CPP_MIN_LOG_LEVEL=2 PYTHONHASHSEED=7 PYTHONUNBUFFERED=1
module load conda/miniforge3/26.9.4-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla
python - <<PY
import json
from pathlib import Path
from huggingface_hub import snapshot_download
pins = [
    ("moojink/openvla-7b-oft-finetuned-libero-10", "95220f9a3421a7ff12d4218e73d09ade830fa9a3", "model", "$CKPT", dict(ignore_patterns=["lora_adapter/*"])),
    ("openvla/modified_libero_rlds", "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551", "dataset", "$RLDS", dict(allow_patterns=["libero_10_no_noops/*"])),
]
for rid, rev, kind, dest, kw in pins:
    if not (Path(dest) / "source.json").exists():
        snapshot_download(rid, revision=rev, repo_type=kind, local_dir=dest, max_workers=4, **kw)
        Path(dest, "source.json").write_text(json.dumps({"id": rid, "revision": rev}, indent=2) + "\n")
PY
cp "$RLDS/source.json" "$RLDS/libero_10_no_noops/1.0.0/source.json"
python research/experiments/phase-and-action-sensitive/demo_frames.py \
  --rlds-dir "$RLDS/libero_10_no_noops/1.0.0" --out "$OUT/demo-frames/$SLURM_JOB_ID" \
  --phase-window 10 --flip-min-hold 5 --frames-per-phase 64 --chunk 8 --seed 7
