# shellcheck shell=bash
# Sourced by every script: the repository's paths and the PEAR run arguments the pipeline's steps share.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PEAR=$ROOT/third_party/pear
WORK_DIR=${WORK_DIR:?the directory scripts/setup.sh installed into (environments, models)}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear}  # node-local compile caches
export WORK_DIR PEAR

# what every process imports: PEAR, its patched verl, and this repository's opd
PYTHON_PATH=$PEAR/pear/utils/sitecustomize:$PEAR:$PEAR/3rdparty/verl/src:$ROOT/src

# merge_args <array name>: keeps the last value of every key, Hydra refusing two: a later entry replaces an earlier one
# of the same key (with or without a leading +) or under it. A step's defaults, then the hardware's, then the user's.
merge_args() {
  local -n merged=$1
  local entries=("${merged[@]}") kept=() entry key earlier earlier_key
  for entry in "${entries[@]}"; do
    key=${entry%%=*}
    key=${key#+}
    local survivors=()
    for earlier in "${kept[@]}"; do
      earlier_key=${earlier%%=*}
      earlier_key=${earlier_key#+}
      [[ $earlier_key == "$key" || $earlier_key == "$key".* ]] || survivors+=("$earlier")
    done
    kept=("${survivors[@]}" "$entry")
  done
  merged=("${kept[@]}")
}

# pear_args: fills RUN_ARGS, to which a step adds its own, with what a run on a LIBERO family needs: the episodes, the model's arguments (from
# MODEL_FILE), the rollout replicas and the Ray workers' environment. Reads GPU, RUN, ROLLOUT_NNODES, MODEL_FILE, MODEL
# and the optional BENCH (libero | libero_plus | libero_pro), EPISODES, VAL_EPISODES, MAX_PROMPT, MAX_RESPONSE.
pear_args() {
  GPU=${GPU:-generic}  # h100 | gb300: PEAR's own recipes; otherwise a profile of configs/hardware
  RUN=$(readlink -m "${RUN:?the run directory}")
  ROLLOUT_NNODES=${ROLLOUT_NNODES:?the Ray machines started as rollout}
  MODEL_FILE=${MODEL_FILE:?a file of configs/models}
  MODEL=${MODEL:?the model directory}
  BENCH=${BENCH:-libero}
  local bench_dir=$PEAR/pear/envs/$BENCH/configs
  if [ "$BENCH" = libero ]; then
    EPISODES=${EPISODES:-$bench_dir/libero_all.json}  # 2,000 episodes: 4 suites x 10 tasks x 50 initial states
  else
    EPISODES=${EPISODES:-$bench_dir/all.json}
  fi
  VAL_EPISODES=${VAL_EPISODES:-$EPISODES}
  MAX_PROMPT=${MAX_PROMPT:-4096}
  MAX_RESPONSE=${MAX_RESPONSE:-160000}
  MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE))

  # MODEL_ARGS, MODEL_ROLLOUT_ARGS, MODEL_TRAINING_ARGS, GPUS_PER_NODE, model_image_tokens, ... PEAR's model file knows
  # H100 and GB300 only: other hardware takes the H100 configuration with its own replacements
  HARDWARE_ARGS=()
  if [ "$GPU" = h100 ] || [ "$GPU" = gb300 ]; then
    # shellcheck source=/dev/null
    source "$MODEL_FILE"
  else
    # shellcheck source=/dev/null
    GPU=h100 source "$MODEL_FILE"
    # shellcheck source=/dev/null
    source "$ROOT/configs/hardware/$GPU.sh"
  fi
  # on one machine the roles split its GPUs (scripts/ray_node.sh)
  ROLLOUT_GPUS=${ROLLOUT_GPUS:-$GPUS_PER_NODE}

  local env_var=+ray_kwargs.ray_init.runtime_env.env_vars
  EXTERNAL_MODULES=pear.rollout.server,$MODEL_MODULES
  START_TIME=$(date -u +%Y%m%dT%H%M%SZ)
  RUN_ARGS=(
    run_dir="$RUN"
    "data.train_files=[$EPISODES]"
    "data.val_files=[$VAL_EPISODES]"
    data.max_prompt_length="$MAX_PROMPT"
    data.max_response_length="$MAX_RESPONSE"
    "${MODEL_ARGS[@]}"
    "${MODEL_ROLLOUT_ARGS[@]}"
    actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES"
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS"
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
    +actor_rollout_ref.rollout.limit_images=$((MAX_MODEL_LEN / $(model_image_tokens 512 512)))
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$bench_dir/tool_config.yaml"
    trainer.validation_data_dir="$RUN/val_dumps"
    trainer.experiment_name="$(basename "$RUN")"
    trainer.logger='["console","file","tensorboard"]'
    hydra.run.dir="$RUN/hydra/$START_TIME"
    run_sizes.sessions_per_replica="$SESSIONS_PER_REPLICA"
    "$env_var.VERL_USE_EXTERNAL_MODULES='$EXTERNAL_MODULES'"
    "$env_var.PYTHONPATH='$PYTHON_PATH'"
    "$env_var.VERL_FILE_LOGGER_PATH=$RUN/metrics_$START_TIME.jsonl"
    "$env_var.LIBERO_CONFIG_PATH=$WORK_DIR/envs/$BENCH/etc/libero"
    "$env_var.LIBERO_PYTHON=$WORK_DIR/envs/$BENCH/bin/python"
    "$env_var.LIBERO_MESA_DIR=$WORK_DIR/envs/mesa"
  )
  if [ "$BENCH" = libero_plus ]; then
    RUN_ARGS+=("$env_var.MAGICK_HOME='$WORK_DIR/envs/imagemagick'")  # LIBERO-Plus's motion blur
  fi
}

# pear_exec <step> [user overrides...]: runs the step's entry point (opd.<step>) on the Ray cluster with its overlay
# (configs/step/<step>.yaml), RUN_ARGS, the hardware's arguments and the user's overrides, merged (merge_args); its log
# is appended to $RUN/train.log.
pear_exec() {
  local step=$1
  shift
  local args=("+step=$step" "${RUN_ARGS[@]}" "${HARDWARE_ARGS[@]}" "$@")
  merge_args args
  mkdir -p "$RUN"
  # shellcheck source=/dev/null
  source "$PEAR/scripts/gpu_process_variables.sh" "$VERL_ENV"
  export RAY_ADDRESS=${RAY_ADDRESS:-auto} PEAR_LOG_DIR=$RUN/sim WANDB_DIR=$RUN
  export VERL_USE_EXTERNAL_MODULES=$EXTERNAL_MODULES PYTHONPATH=$PYTHON_PATH
  # refuses a verl checkout other than PEAR's pinned commit and patches
  bash "$PEAR/3rdparty/verl/scripts/checkout_verl.sh" --check | tee -a "$RUN/train.log"
  python -m "opd.$step" --config-dir "$ROOT/configs" "${args[@]}" 2>&1 | tee -a "$RUN/train.log"
}
