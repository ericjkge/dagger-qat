#!/usr/bin/env bash
#SBATCH --job-name=vla-setup-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/setup-%j.out
# Create the `vla` conda env (Python 3.11) from requirements-runtime.txt plus the
# OpenVLA-OFT and LIBERO submodules, then check imports. Submit from the worktree root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
OUT=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive
export CMAKE_POLICY_VERSION_MINIMUM=3.5  # egl_probe declares cmake<3.5; CMake 4 refuses otherwise
export PIP_CACHE_DIR=$OUT/cache/pip LIBERO_CONFIG_PATH=$OUT/libero-config
mkdir -p "$PIP_CACHE_DIR" "$LIBERO_CONFIG_PATH" "$OUT/datasets"
module load conda/miniforge3/26.9.4-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q '^vla ' || conda create -y -n vla -c conda-forge python=3.11 cmake
conda activate vla
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-runtime.txt
pip install --no-deps -e third_party/openvla-oft -e third_party/LIBERO
export PYTHONPATH=$PWD/src  # dqat pins Python 3.10 in pyproject; load it from source instead
python - <<PY
import json; from pathlib import Path
root = Path('third_party/LIBERO/libero/libero').resolve()
Path('$LIBERO_CONFIG_PATH/config.yaml').write_text(json.dumps(dict(
    benchmark_root=str(root), bddl_files=str(root/'bddl_files'), init_states=str(root/'init_files'),
    assets=str(root/'assets'), datasets='$OUT/datasets'), indent=2))
PY
MUJOCO_GL=egl python -c "
import torch, transformers, tensorflow, robosuite, dqat
from libero.libero import benchmark
from experiments.robot.libero.run_libero_eval import initialize_model
print('torch', torch.__version__, 'transformers', transformers.__version__, 'libero_10 tasks', benchmark.get_benchmark_dict()['libero_10']().n_tasks)
print('vla env OK')"
pip freeze > "$OUT/vla-freeze.txt"
