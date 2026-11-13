# Sourced by every job script: conda env, paths and runtime variables.
OUT=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive
CKPT=$OUT/checkpoints/openvla-oft-libero-10
EXP=research/experiments/phase-and-action-sensitive
export HF_HOME=$OUT/cache/huggingface LIBERO_CONFIG_PATH=$OUT/libero-config
export PYTHONPATH=$PWD/src:$PWD/third_party/openvla-oft:$PWD/third_party/LIBERO
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=7 PYTHONUNBUFFERED=1
module load conda/miniforge3/26.9.4-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla
