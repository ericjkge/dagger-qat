#!/usr/bin/env bash
# Paired replay benchmark across server configs. Results: /workspace/results/bench/<config>.jsonl
SEL=libero_spatial_t0_r0:0,libero_object_t2_r0:0,libero_goal_t2_r0:0,libero_spatial_t2_r0:4,libero_object_t0_r0:5,libero_spatial_t6_r0:8,libero_10_t0_r0:8,libero_spatial_t4_r0:8,libero_10_t2_r0:12,libero_goal_t0_r0:8,libero_10_t2_r0:24,libero_goal_t0_r0:20
EXACT=libero_spatial_t0_r0:0,libero_object_t0_r0:5,libero_10_t2_r0:12
OUT=/workspace/results/bench; mkdir -p $OUT
. /workspace/simenv.sh; export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
stop_server() { pkill -f "[s]glang.launch_server"; for i in $(seq 1 60); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 2; done; sleep 5; }
run_cfg() {
  local name=$1; shift
  stop_server
  echo "=== $name $(date -u)"
  CUDA_VISIBLE_DEVICES=0 PEAR_SPEC_STEP_LOG=$OUT/steps_$name.log setsid /workspace/serve.sh bench_$name "$@" < /dev/null &
  for i in $(seq 1 200); do
    curl -s -m 3 127.0.0.1:30000/get_model_info | grep -q model_path && break
    grep -q kill_process_tree /workspace/logs/server_bench_$name.log 2>/dev/null && { echo "SERVER_FAILED $name"; return; }
    sleep 6
  done
  cd /workspace
  sim/bin/python replay_bench.py --results /workspace/results/dflash2_main --select $EXACT --reps 1 --greedy 1 --max-new 512 --config $name --out $OUT/exact_$name.jsonl 2>&1 | grep -v -i "warn\|gym\|task orders" | tail -n 1
  sim/bin/python replay_bench.py --results /workspace/results/dflash2_main --select $SEL --reps 2 --config $name --out $OUT/$name.jsonl 2>&1 | grep -v -i "warn\|gym\|task orders" | tail -n 1
  echo "DONE_CFG $name $(date -u)"
}
run_cfg dflash2_k7 --speculative-algorithm DFLASH --speculative-draft-model-path /workspace/models/dflash2 --speculative-num-draft-tokens 8
run_cfg ar
run_cfg dflash2_k3 --speculative-algorithm DFLASH --speculative-draft-model-path /workspace/models/dflash2 --speculative-num-draft-tokens 4
run_cfg ngram --speculative-algorithm NGRAM
stop_server
echo BENCH_DONE
