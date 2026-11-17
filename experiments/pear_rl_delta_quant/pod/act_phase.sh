#!/usr/bin/env bash
# Score arm B (released weights) with activation fake-quant variants. B0 re-scores plain B (hook-path sanity).
. /workspace/lib.sh; cd /workspace; mkdir -p results/pilot
[ -f models/arm_B/arm_stats.json ] || CUDA_VISIBLE_DEVICES=0 sgl/bin/python build_arm.py --arm B --out models/arm_B 2>&1 | tail -n 1
run() {  # name ACTQ
  echo "=== score $1 ACTQ=$2 $(date -u)"
  stop_server; rm -f logs/server_$1.log; ACTQ=$2 PYTHONPATH=/workspace setsid /workspace/serve.sh $1 models/arm_B --disable-cuda-graph < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 300); do curl -s -m 3 127.0.0.1:30000/get_model_info | grep -q model_path && break
    grep -q "kill_process_tree\|Traceback" logs/server_$1.log && { echo SERVER_FAILED $1; return; }; sleep 6; done
  grep -m3 "ACTQ" logs/server_$1.log
  rm -f results/pilot/tf_$1.jsonl
  CUDA_VISIBLE_DEVICES="" sim/bin/python score_tf.py --arm $1 --out results/pilot/tf_$1.jsonl 2>&1 | grep -E "SCORE_DONE|Traceback|Error"
}
# B0 already scored
run Bmlp mlp
run Bmlp_lm mlp,lmhead
run Bfull mlp,lmhead,attn
stop_server; echo ACT_DONE
# context-pruning functional check on BF16 PEAR
echo "=== strip $(date -u)"; start_server A && { rm -f results/pilot/strip_A.jsonl; CUDA_VISIBLE_DEVICES="" sim/bin/python score_strip.py --out results/pilot/strip_A.jsonl 2>&1 | grep -v -i "warn\|gym\|task orders"; }
stop_server; echo STRIP_PHASE_DONE
