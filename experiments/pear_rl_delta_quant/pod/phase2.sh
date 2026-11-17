#!/usr/bin/env bash
# Phase 2: closed-loop LIBERO Goal + Long (paired init states) per arm, 5 parallel episode workers.
. /workspace/lib.sh; cd /workspace
ARMS=${ARMS:-"A B C"}; TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}; TRIAL=${TRIAL:-0}; W=${W:-5}
for arm in $ARMS; do
  echo "=== closed loop $arm $(date -u)"
  start_server $arm || continue
  EPS=(); for s in libero_goal libero_10; do for t in $TASKS; do EPS+=("$s:$t:$TRIAL"); done; done
  for w in $(seq 0 $((W-1))); do
    L=""; for i in "${!EPS[@]}"; do [ $((i % W)) -eq $w ] && L="$L,${EPS[$i]}"; done
    sim/bin/python run_episodes.py --out results/pilot/cl_$arm --max-rounds 40 --episodes ${L#,} 2>&1 \
      | grep --line-buffered "end_reason" > logs/cl_${arm}_w$w.log &
  done
  wait; echo "DONE_CL $arm $(date -u)"
done
stop_server; echo PHASE2_DONE
