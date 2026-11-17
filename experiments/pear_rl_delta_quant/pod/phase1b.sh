#!/usr/bin/env bash
# Phase 1b: control arm E = RTN(W_pear) on the checkpoint scales (same quantizer as oracle C).
. /workspace/lib.sh; cd /workspace
until grep -q PHASE1_DONE logs/phase1.log; do sleep 20; done
CUDA_VISIBLE_DEVICES=0 sgl/bin/python build_arm.py --arm E --out models/arm_E 2>&1 | tail -n 1
cp models/arm_E/arm_stats.json results/pilot/arm_stats_E.json
echo "=== score E $(date -u)"; start_server E && { rm -f results/pilot/tf_E.jsonl; CUDA_VISIBLE_DEVICES="" sim/bin/python score_tf.py --arm E --out results/pilot/tf_E.jsonl 2>&1 | grep -v -i "warn\|gym\|task orders"; }
stop_server; echo PHASE1B_DONE
