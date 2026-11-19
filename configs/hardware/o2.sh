# shellcheck shell=bash disable=SC2034
# Sourced after a model file: HMS O2's GPU machines, 4 x A100 80GB or 4 x L40S 48GB each. Neither has PEAR's Hopper or
# Blackwell kernels or FlashAttention-3, so the model runs on vLLM's own Qwen3.5 model and verl's FSDP path with
# FlashAttention-2, and no LMCache sidecar. HARDWARE_ARGS replace the model file's arguments of the same keys.
GPUS_PER_NODE=4
SP=${SP:-4}
TOK_PER_GPU=$(((MAX_MODEL_LEN + SP - 1) / SP))
HARDWARE_ARGS=(
  actor_rollout_ref.actor.fsdp_config.use_gpu_backend=false
  actor_rollout_ref.rollout.kernels=null
  "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
  "+actor_rollout_ref.rollout.custom.lmcache={enable: false}"
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="$SP"
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOK_PER_GPU"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOK_PER_GPU"
)
