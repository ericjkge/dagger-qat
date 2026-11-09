#!/usr/bin/env bash
set -euo pipefail
DQAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DQAT_DATA_DIR="${DQAT_DATA_DIR:-$DQAT_ROOT/data}"
export HF_HOME="$DQAT_DATA_DIR/cache/huggingface"
export UV_CACHE_DIR="$DQAT_DATA_DIR/cache/uv"
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR="$DQAT_DATA_DIR/python"
export LIBERO_CONFIG_PATH="$DQAT_DATA_DIR/libero-config"
export PYTHONPATH="$DQAT_ROOT/src:$DQAT_ROOT/third_party/openvla-oft:$DQAT_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=7
DQAT_PYTHON="$DQAT_ROOT/.venv-runtime/bin/python"
mkdir -p "$DQAT_DATA_DIR" "$LIBERO_CONFIG_PATH"
cd "$DQAT_ROOT"
