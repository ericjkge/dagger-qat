#!/bin/bash
# scripts/setup.sh for the smoke test: PEAR's training environment and LIBERO, the 4B student, PEAR's checks. Built for
# O2's A100s (8.0) and L40S (8.9); MAX_JOBS keeps the compilers within the requested memory.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/setup-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh
CUDA_ARCH="8.0;8.9" MAX_JOBS=12 BENCHES=libero MODEL_FILES=configs/models/qwen3_5_4b.sh bash scripts/setup.sh
