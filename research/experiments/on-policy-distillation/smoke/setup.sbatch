#!/bin/bash
# scripts/setup.sh for the smoke test on a CPU machine (no GPU time spent compiling): PEAR's training environment and
# LIBERO, and the 4B student. Built for the L40S (8.9) the smoke test runs on, one architecture to keep the build short;
# MAX_JOBS keeps the compilers within the requested memory. PEAR's GPU checks run first in pipeline.sbatch.
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/setup-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh
CUDA_ARCH=8.9 MAX_JOBS=12 BENCHES=libero MODEL_FILES=configs/models/qwen3_5_4b.sh bash scripts/setup.sh
