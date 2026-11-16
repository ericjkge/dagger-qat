#!/bin/bash
# usage: run_arm.sh <rule> <out_dir> <parallel> [extra args for run_anytime.py]
RULE=$1; OUT=$2; P=$3; shift 3
cd /workspace; mkdir -p $OUT/logs
for s in ${SUITES:-libero_spatial libero_object libero_goal libero_10}; do for t in ${TASKS:-0 1 2 3 4 5 6 7 8 9}; do echo "$s:$t:${TRIAL:-0}"; done; done |
  xargs -P $P -I{} bash -c '. /workspace/simenv.sh; e={}; n=${e//:/_}; for try in 1 2; do
     sim/bin/python run_anytime.py --rule '"$RULE"' --out '"$OUT"' --episode $e '"$*"' > '"$OUT"'/logs/$n.log 2>&1 && grep -q "'success'" '"$OUT"'/logs/$n.log && break; echo "retry $e" >> '"$OUT"'/logs/retries.txt; done'
echo "ARM_DONE $RULE"
