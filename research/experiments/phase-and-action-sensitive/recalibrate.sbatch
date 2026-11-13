#!/usr/bin/env bash
#SBATCH --job-name=vla-recal
#SBATCH --partition=gpu,gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1,vram:44G
#SBATCH --exclude=compute-g-16-254,compute-g-16-255,compute-g-17-145,compute-g-17-146,compute-g-17-147,compute-g-17-148,compute-g-17-149,compute-g-17-150,compute-g-17-151,compute-g-17-152,compute-g-17-153,compute-g-17-154,compute-g-17-155,compute-g-17-156,compute-g-17-157,compute-g-17-158,compute-g-17-159,compute-g-17-160,compute-g-17-161,compute-gc-17-245,compute-gc-17-246,compute-gc-17-247  # RTX 8000/6000 and V100 nodes: no BF16
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/recal-%j.out
# Action-head recalibration on top of the worst-phase seed-7 allocation (arm worst-s7-head).
# Submit with --dependency=afterok:<seed-7 score job>.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source research/experiments/phase-and-action-sensitive/env.sh
FRAMES=$OUT/demo-frames/$SLURM_JOB_ID-recal
[[ -f $FRAMES/frames.npz ]] || rm -rf $FRAMES  # requeued: redo an incomplete draw
[[ -f $FRAMES/frames.npz ]] || python $EXP/demo_frames.py --rlds-dir $OUT/datasets/modified_libero_rlds/libero_10_no_noops/1.0.0 --out $FRAMES \
  --phase-window 10 --flip-min-hold 5 --frames-per-phase 1024 --chunk 8 --seed 10
python $EXP/recalibrate.py --checkpoint $CKPT --frames $FRAMES/frames.npz --alloc $OUT/alloc/worst-s7.json --out $OUT/heads/worst-s7.pt \
  --val-frac 0.1 --steps 3000 --batch 32 --lr 3e-5 --eval-every 250 --seed 10
