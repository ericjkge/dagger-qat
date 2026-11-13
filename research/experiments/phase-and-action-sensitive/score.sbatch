#!/usr/bin/env bash
#SBATCH --job-name=vla-score
#SBATCH --partition=gpu,gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/score-%j.out
# One calibration seed (SEED in the environment): frames (seed 7 reuses job
# 59511837), scores, then worst-phase and phase-agnostic allocations at b̄ = 3.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source research/experiments/phase-and-action-sensitive/env.sh
FRAMES=$OUT/demo-frames/59511837
if [[ $SEED != 7 ]]; then
  FRAMES=$OUT/demo-frames/$SLURM_JOB_ID-s$SEED
  [[ -f $FRAMES/frames.npz ]] || rm -rf $FRAMES  # requeued: redo an incomplete draw
  [[ -f $FRAMES/frames.npz ]] || python $EXP/demo_frames.py --rlds-dir $OUT/datasets/modified_libero_rlds/libero_10_no_noops/1.0.0 --out $FRAMES \
    --phase-window 10 --flip-min-hold 5 --frames-per-phase 64 --chunk 8 --seed $SEED
fi
python $EXP/score.py --checkpoint $CKPT --frames $FRAMES/frames.npz --out $OUT/scores/s$SEED.npz --k 8 --seed $SEED
for mode in worst-phase phase-agnostic; do
  python $EXP/allocate.py --scores $OUT/scores/s$SEED.npz --mode $mode --budget 3 --out $OUT/alloc/${mode%%-*}-s$SEED.json
done
