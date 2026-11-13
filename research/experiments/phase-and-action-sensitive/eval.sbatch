#!/usr/bin/env bash
#SBATCH --job-name=vla-eval
#SBATCH --partition=gpu,gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1,vram:44G
#SBATCH --exclude=compute-g-16-254,compute-g-16-255,compute-g-17-145,compute-g-17-146,compute-g-17-147,compute-g-17-148,compute-g-17-149,compute-g-17-150,compute-g-17-151,compute-g-17-152,compute-g-17-153,compute-g-17-154,compute-g-17-155,compute-g-17-156,compute-g-17-157,compute-g-17-158,compute-g-17-159,compute-g-17-160,compute-g-17-161,compute-gc-17-245,compute-gc-17-246,compute-gc-17-247  # RTX 8000/6000 and V100 nodes: no BF16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/eval-%j.out
# Two evaluation workers share one GPU (any BF16 GPU with >= 44 GB) and one unit queue (see eval.py). The
# first partition with such a GPU runs it; on requeue, this job's
# unfinished claims are freed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source research/experiments/phase-and-action-sensitive/env.sh
export OMP_NUM_THREADS=4
CLAIMS=$OUT/eval/claims
mkdir -p $CLAIMS
for c in $(grep -lx "$SLURM_JOB_ID" $CLAIMS/* 2>/dev/null); do [[ -e $c.done ]] || rm $c; done
nvidia-smi
for w in 1 2; do
  python $EXP/eval.py --checkpoint $CKPT --root $OUT \
    --arms uniform worst-s7 phase-s7 worst-s7-head fp worst-s8 phase-s8 worst-s9 phase-s9 \
    --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 50 --seed 7 --uniform-bits 3 \
    --phase-window 10 --flip-min-hold 5 >> $OUT/logs/eval-$SLURM_JOB_ID-w$w.out 2>&1 &
  sleep 120  # stagger model loads
done
wait
