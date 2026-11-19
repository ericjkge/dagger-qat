#!/bin/bash
# What a GPU node offers PEAR's environment: the driver (CUDA 13 wants >= 580), the compute capability, CPUs, memory,
# and access to the package indexes and the Hugging Face Hub.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/probe-%j.out
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
nproc; free -g | head -2; df -h /n/scratch/users/e/edk005 | tail -1
for url in https://pypi.org/simple/ https://download.pytorch.org/whl/ https://huggingface.co/ https://github.com/; do
  printf '%s %s\n' "$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$url")" "$url"
done
gcc --version | head -1; ldd --version | head -1
