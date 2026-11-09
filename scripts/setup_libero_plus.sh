#!/usr/bin/env bash
# Add Plus assets/dependencies without changing the completed nominal runtime.
set -euo pipefail
source "$(dirname "$0")/common.sh"
revision=4976dc30028e805ff8094b55501d532c48fec182
# Keep the many small source/asset files on home storage. Scratch is used for
# large downloads/results; its metadata latency can dominate a full checkout.
plus_root="$DQAT_ROOT/data/vendor/LIBERO-plus-$revision"
deps="$DQAT_ROOT/data/libero-plus-deps"
mkdir -p "$(dirname "$plus_root")" "$deps"
if [[ ! -d "$plus_root/.git" ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/sylvestf/LIBERO-plus.git "$plus_root"
fi
git -C "$plus_root" cat-file -e "$revision^{commit}" 2>/dev/null || git -C "$plus_root" fetch --depth 1 origin "$revision"
git -C "$plus_root" checkout --detach "$revision"
[[ "$(git -C "$plus_root" rev-parse HEAD)" == "$revision" ]]
# The base runtime already provides NumPy, SciPy, Pillow, imageio and networkx.
# A separate target directory preserves every package in .venv-runtime.
uv pip install --python "$DQAT_PYTHON" --target "$deps" --no-deps \
  Wand==0.6.13 scikit-image==0.22.0 lazy-loader==0.4 tifffile==2024.2.12
export PYTHONPATH="$deps:$PYTHONPATH"
"$DQAT_PYTHON" "$(dirname "$0")/download_plus_assets.py" --root "$plus_root" --data-dir "$DQAT_DATA_DIR"
export LIBERO_CONFIG_PATH="$DQAT_DATA_DIR/libero-plus-config"
export PYTHONPATH="$plus_root:$PYTHONPATH"
"$DQAT_PYTHON" - <<'PY'
from wand.image import Image
import skimage
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_10"]()
assert suite.n_tasks == 2519, suite.n_tasks
print("Plus setup ready:", suite.n_tasks, "Long variants; skimage", skimage.__version__)
PY
