#!/usr/bin/env bash
# usage: serve.sh <name> [extra sglang args...]   (spec config passed in extra args)
NAME=$1; shift
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64 HF_HOME=/workspace/hf CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=/workspace/sgl/bin:/usr/local/cuda-13.0/bin:$PATH SGLANG_ENABLE_JIT_DEEPGEMM=0
cd /workspace
exec /workspace/sgl/bin/python -m sglang.launch_server --model-path /workspace/models/pear \
  --host 127.0.0.1 --port 30000 --max-running-requests 1 --context-length 262144 \
  --mem-fraction-static 0.85 --random-seed 0 --enable-metrics "$@" > /workspace/logs/server_$NAME.log 2>&1
