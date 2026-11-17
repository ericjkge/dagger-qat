#!/usr/bin/env bash
# usage: serve.sh <arm_name> <model_dir> [extra sglang args]
NAME=$1; MODEL=$2; shift 2
COMPAT=$(nvidia-smi | grep -q "CUDA Version: 12" && echo /usr/local/cuda-13.0/compat:); export LD_LIBRARY_PATH=${COMPAT}/usr/local/cuda-13.0/lib64 HF_HOME=/workspace/hf CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=/workspace/sgl/bin:/usr/local/cuda-13.0/bin:$PATH SGLANG_ENABLE_JIT_DEEPGEMM=0
cd /workspace
exec /workspace/sgl/bin/python -m sglang.launch_server --model-path $MODEL --tokenizer-path /workspace/models/pear \
  --host 127.0.0.1 --port 30000 --max-running-requests 8 --context-length 262144 \
  --mem-fraction-static 0.85 --random-seed 0 "$@" > /workspace/logs/server_$NAME.log 2>&1
