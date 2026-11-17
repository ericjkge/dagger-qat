export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/workspace/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/cache/python HF_HOME=/workspace/cache/hf
export OPENPI_DATA_HOME=/workspace/cache/openpi LIBERO_CONFIG_PATH=/workspace/libero-config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/workspace/dqat/src:/workspace/openpi/third_party/libero PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /workspace/openpi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
