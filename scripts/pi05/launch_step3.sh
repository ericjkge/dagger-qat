#!/usr/bin/env bash
# Step 3 launcher (pod). One mechanism job per task; appends one JSON line per state (resumable by
# re-running with a new --seed/--out, results are never overwritten).
set -euo pipefail
source /workspace/env.sh
R=${R:-/workspace/results/2026-11-14-pi05-contact}
PER_PHASE=${PER_PHASE:-25}
for t in spatial_bowl object_soup goal_cheese; do
  setsid nohup .venv/bin/python /workspace/dqat/scripts/pi05/mechanism.py --checkpoint /workspace/ckpt/pi05_libero_pytorch \
    --sweep $R/sweep/$t --out $R/mechanism/$t --per-phase $PER_PHASE > $R/logs/mechanism_$t.log 2>&1 < /dev/null &
  setsid nohup .venv/bin/python /workspace/dqat/scripts/pi05/rescore_events.py --sweep $R/sweep/$t \
    --out $R/rescore/$t > $R/logs/rescore_$t.log 2>&1 < /dev/null &
done
echo launched
