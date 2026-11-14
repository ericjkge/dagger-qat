#!/usr/bin/env bash
# One-time setup of a Runpod (Ubuntu, NVIDIA) pod for the π0.5 contact-guard experiments.
# Installs openpi (PyTorch path) + LIBERO into ONE Python 3.11 venv so the policy and the
# simulator run in the same process (needed for paired counterfactual sim branching).
# Everything persistent lives under $W (/workspace), which survives pod stop.
set -euo pipefail
W=${W:-/workspace}
OPENPI_REV=${OPENPI_REV:-215abfb217dbac7d5f1273282331b9b1866c0479}
export UV_CACHE_DIR=$W/cache/uv UV_PYTHON_INSTALL_DIR=$W/cache/python HF_HOME=$W/cache/hf
export OPENPI_DATA_HOME=$W/cache/openpi
mkdir -p $W/cache $W/logs

log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }

log "apt packages (EGL/OSMesa for MuJoCo offscreen rendering)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs cmake build-essential libegl1 libgl1 libglvnd0 libgles2 \
  libosmesa6 libosmesa6-dev ffmpeg rsync >/dev/null

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH

log "openpi @ $OPENPI_REV"
if [[ ! -d $W/openpi ]]; then
  git clone -q https://github.com/Physical-Intelligence/openpi.git $W/openpi
fi
cd $W/openpi
git fetch -q origin && git checkout -q $OPENPI_REV
git submodule update -q --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync -q
uv pip install -q -e .

log "transformers_replace patch (AdaRMS, activation precision, read-only KV cache)"
TF_DIR=$(uv run python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TF_DIR/"
uv run python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(); print('transformers_replace OK')"

log "LIBERO into the same venv"
uv pip install -q -e third_party/libero
uv pip install -q robosuite==1.4.1 bddl==1.0.1 easydict==1.13 mujoco==3.2.3 \
  gym==0.25.2 future cloudpickle h5py imageio[ffmpeg] opencv-python-headless matplotlib scipy pandas
mkdir -p $W/libero-config
LIBERO_ROOT=$W/openpi/third_party/libero/libero/libero
cat > $W/libero-config/config.yaml <<EOF
{"benchmark_root": "$LIBERO_ROOT", "bddl_files": "$LIBERO_ROOT/bddl_files",
 "init_states": "$LIBERO_ROOT/init_files", "assets": "$LIBERO_ROOT/assets",
 "datasets": "$W/cache/libero-datasets"}
EOF

log "download official pi05_libero (JAX) and convert to PyTorch (bfloat16, as shipped)"
uv run python - <<'PY'
from openpi.shared import download
p = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
print("JAX checkpoint at", p)
PY
JAX_CKPT=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
PT_CKPT=$W/ckpt/pi05_libero_pytorch
if [[ ! -f $PT_CKPT/model.safetensors ]]; then
  uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir $JAX_CKPT \
    --config_name pi05_libero --output_path $PT_CKPT --precision bfloat16
fi
# The policy loader expects the normalization assets next to the weights.
[[ -d $PT_CKPT/assets ]] || cp -r $JAX_CKPT/assets $PT_CKPT/
# `uv run` re-syncs the venv to openpi's lockfile (mujoco 2.3.7). openpi's own LIBERO eval env pins
# mujoco 3.2.3 + robosuite 1.4.1, so restore it LAST and never call `uv run` afterwards
# (experiments use .venv/bin/python directly).
uv pip install -q mujoco==3.2.3
.venv/bin/python -c "import mujoco; assert mujoco.__version__ == '3.2.3' and hasattr(mujoco, 'mj_geomDistance')"
ls -la $PT_CKPT
log "done"
