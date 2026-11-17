#!/usr/bin/env bash
# PEAR spec-decoding study, pod setup (H200). Logs in /workspace/logs/.
set -x
mkdir -p /workspace/logs /workspace/models /workspace/hf
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 UV_CACHE_DIR=/workspace/uvcache
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
# 1. checkpoints (background)
( uv tool install -q "huggingface_hub[cli,hf_transfer]" && export PATH=$HOME/.local/bin:$PATH && \
  hf download incoai/Qwen3.8-27B-DFlash2 --local-dir /workspace/models/dflash2 && \
  hf download mit-han-lab/Qwen3.8-27B-PEAR --local-dir /workspace/models/pear && echo DOWNLOAD_DONE ) \
  > /workspace/logs/download.log 2>&1 &
# 2. SGLang (git main; DFLASH support), CUDA 12.9 driver -> cu128/cu129 torch
( cd /workspace && uv venv -q --python 3.12 sgl && . sgl/bin/activate && \
  uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python" && \
  python -c "import torch, sglang; print('torch', torch.__version__, torch.version.cuda, 'sglang', sglang.__version__, torch.cuda.is_available())" && echo SGL_DONE ) \
  > /workspace/logs/sglang_install.log 2>&1 &
# 3. PEAR repo + LIBERO sim env (python 3.10)
( cd /workspace && git clone -q https://github.com/mit-han-lab/pear.git && git -C pear rev-parse HEAD && \
  git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git && git -C LIBERO checkout -q 8f1084e3132a39270c3a13ebe37270a43ece2a01 && \
  apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libosmesa6 > /dev/null && \
  uv venv -q --python 3.10 sim && . sim/bin/activate && \
  uv pip install -c pear/scripts/install/sim/libero/constraints.txt -e LIBERO robosuite mujoco bddl easydict \
     gym numpy pillow imageio opencv-python-headless pyyaml openai hydra-core termcolor "transformers<5" einops \
     torch --torch-backend cpu && \
  python -c "import libero, robosuite, mujoco; print('mujoco', mujoco.__version__, 'robosuite', robosuite.__version__)" && echo SIM_DONE ) \
  > /workspace/logs/sim_install.log 2>&1 &
wait
