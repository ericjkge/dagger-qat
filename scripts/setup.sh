#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'The OpenVLA-OFT runtime requires Linux. Use CPU unit tests on macOS.' >&2
  exit 1
fi
command -v cmake >/dev/null || { echo 'Load CMake before setup (needed by egl-probe).' >&2; exit 1; }
git submodule update --init --recursive
if [[ ! -x "$DQAT_PYTHON" ]]; then
  uv venv --python 3.10.14 --seed .venv-runtime
fi
uv pip install --python "$DQAT_PYTHON" --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match -r requirements-runtime.txt
uv pip install --python "$DQAT_PYTHON" --no-deps -e third_party/openvla-oft -e third_party/LIBERO -e .
"$DQAT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path('third_party/LIBERO/libero/libero').resolve()
config = dict(benchmark_root=str(root), bddl_files=str(root / 'bddl_files'),
              init_states=str(root / 'init_files'), assets=str(root / 'assets'),
              datasets=str(Path(os.environ['DQAT_DATA_DIR']) / 'datasets'))
Path(config['datasets']).mkdir(parents=True, exist_ok=True)
# JSON is valid YAML, and avoids importing LIBERO's interactive initializer.
(Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml').write_text(json.dumps(config, indent=2))
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from experiments.robot.libero.run_libero_eval import initialize_model
from libero.libero.envs import OffScreenRenderEnv
from transformers.models.llama.modeling_llama import LlamaAttention
print('OpenVLA-OFT and LIBERO imports passed.', flush=True)
PY
"$DQAT_PYTHON" -m pip freeze > "$DQAT_DATA_DIR/runtime-freeze.txt"
"$DQAT_PYTHON" scripts/download_assets.py
