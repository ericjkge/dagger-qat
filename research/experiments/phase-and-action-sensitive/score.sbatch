#!/usr/bin/env bash
#SBATCH --job-name=vla-score
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/score-%j.out
# Frames for calibration seeds 8 and 9 (seed 7 is job 59511837), scores for all
# three seeds, worst-phase and phase-agnostic allocations at b̄ = 3, then
# action-head recalibration for the uniform 3-bit backbone.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source research/experiments/phase-and-action-sensitive/env.sh
RLDS=$OUT/datasets/modified_libero_rlds/libero_10_no_noops/1.0.0
declare -A FRAMES=([7]=$OUT/demo-frames/59511837)
for s in 8 9; do
  FRAMES[$s]=$OUT/demo-frames/$SLURM_JOB_ID-s$s
  python $EXP/demo_frames.py --rlds-dir $RLDS --out ${FRAMES[$s]} \
    --phase-window 10 --flip-min-hold 5 --frames-per-phase 64 --chunk 8 --seed $s
done
for s in 7 8 9; do
  python $EXP/score.py --checkpoint $CKPT --frames ${FRAMES[$s]}/frames.npz \
    --out $OUT/scores/s$s.npz --k 8 --seed $s
  for mode in worst-phase phase-agnostic; do
    python $EXP/allocate.py --scores $OUT/scores/s$s.npz --mode $mode --budget 3 \
      --out $OUT/alloc/${mode%%-*}-s$s.json
  done
done
python $EXP/demo_frames.py --rlds-dir $RLDS --out $OUT/demo-frames/$SLURM_JOB_ID-recal \
  --phase-window 10 --flip-min-hold 5 --frames-per-phase 1024 --chunk 8 --seed 10
python $EXP/recalibrate.py --checkpoint $CKPT --frames $OUT/demo-frames/$SLURM_JOB_ID-recal/frames.npz \
  --bits 3 --out $OUT/heads/uniform.pt --val-frac 0.1 --steps 3000 --batch 32 --lr 3e-5 --eval-every 250 --seed 10
