#!/usr/bin/env bash
# Pod setup for the activation-quantization check (no base model needed).
set -x
mkdir -p /workspace/logs /workspace/models /workspace/hf /workspace/results
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 UV_CACHE_DIR=/workspace/uvcache
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
( uv tool install -q "huggingface_hub[cli,hf_transfer]" && export PATH=$HOME/.local/bin:$PATH && \
  hf download mit-han-lab/Qwen3.8-27B-PEAR --local-dir /workspace/models/pear && echo DOWNLOAD_DONE ) \
  > /workspace/logs/download.log 2>&1 &
( cd /workspace && uv venv -q --python 3.12 sgl && . sgl/bin/activate && \
  SGLANG_BUILD_RUST_EXTS=none uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@7476772241#subdirectory=python" ninja && \
  python -c "import sglang; print('sglang', sglang.__version__)" && echo SGL_DONE ) > /workspace/logs/sglang_install.log 2>&1 &
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq libegl1 libgl1 libosmesa6 libegl-mesa0 libgl1-mesa-dri libglvnd0 cuda-compat-13-0 cuda-toolkit-13-0 > /workspace/logs/apt.log 2>&1; echo APT_DONE $? >> /workspace/logs/apt.log
( cd /workspace && git clone -q https://github.com/mit-han-lab/pear.git && git -C pear checkout -q f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087; \
  git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git && git -C LIBERO checkout -q 8f1084e3132a39270c3a13ebe37270a43ece2a01 && \
  uv venv -q --python 3.10 sim && \
  uv pip install -q --python sim/bin/python --torch-backend cpu -c pear/scripts/install/sim/libero/constraints.txt -r pear/pyproject.toml --extra libero && \
  uv pip install -q --python sim/bin/python --no-deps -e LIBERO -C editable_mode=compat && \
  mkdir -p sim/etc/libero /workspace/datasets/libero && P=/workspace/LIBERO/libero/libero && \
  printf "benchmark_root: $P\nbddl_files: $P/bddl_files\ninit_states: $P/init_files\ndatasets: /workspace/datasets/libero\nassets: $P/assets\n" > sim/etc/libero/config.yaml && \
  sim/bin/python -c "import robosuite, mujoco" && echo SIM_DONE ) > /workspace/logs/sim_install.log 2>&1 &
wait
. /workspace/lib.sh; /workspace/sgl/bin/python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" >> /workspace/logs/apt.log 2>&1
echo SETUP_DONE
