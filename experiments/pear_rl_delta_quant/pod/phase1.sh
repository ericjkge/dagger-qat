#!/usr/bin/env bash
# Phase 1: build arms, teacher-forced scoring of the 14 recorded episodes under A, B, C, D (disk: D built last, then removed).
. /workspace/lib.sh; cd /workspace; mkdir -p results/pilot; rm -rf /workspace/uvcache
build() { [ -f models/arm_$1/arm_stats.json ] || CUDA_VISIBLE_DEVICES=0 sgl/bin/python build_arm.py --arm $1 --out models/arm_$1 2>&1 | tail -n 2; }
score() {
  echo "=== score $1 $(date -u)"; start_server $1 || return
  rm -f results/pilot/tf_$1.jsonl
  CUDA_VISIBLE_DEVICES="" sim/bin/python score_tf.py --arm $1 --out results/pilot/tf_$1.jsonl 2>&1 | grep -v -i "warn\|gym\|task orders"
}
build B; build C
for arm in A B C; do score $arm; done
stop_server; build D; score D; stop_server
cp models/arm_*/arm_stats.json results/pilot/ 2>/dev/null; for a in B C D; do cp models/arm_$a/arm_stats.json results/pilot/arm_stats_$a.json; done
rm -rf models/arm_D
echo PHASE1_DONE
