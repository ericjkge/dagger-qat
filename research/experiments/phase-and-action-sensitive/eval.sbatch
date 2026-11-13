#!/usr/bin/env bash
#SBATCH --job-name=vla-eval
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=30:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/eval-%j.out
# Four evaluation workers share one A100 and one unit queue (see eval.py).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source research/experiments/phase-and-action-sensitive/env.sh
export OMP_NUM_THREADS=4
nvidia-smi
for w in 1 2 3 4; do
  python $EXP/eval.py --checkpoint $CKPT --root $OUT \
    --arms uniform worst-s7 phase-s7 fp worst-s8 phase-s8 worst-s9 phase-s9 \
    --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 50 --seed 7 --uniform-bits 3 \
    --phase-window 10 --flip-min-hold 5 > $OUT/logs/eval-$SLURM_JOB_ID-w$w.out 2>&1 &
  sleep 120  # stagger model loads
done
wait
