#!/usr/bin/env bash
# Step 2 launcher (run on the pod). Each job appends per-state/per-episode records as it goes, so a
# crashed job can be resumed with --first-episode <next> into the same directory.
set -euo pipefail
source /workspace/env.sh
R=${R:-/workspace/results/2026-11-14-pi05-contact}
CKPT=/workspace/ckpt/pi05_libero_pytorch
EPISODES=${EPISODES:-20}
mkdir -p $R/logs
TASKS=(
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_alphabet_soup_and_place_it_in_the_basket
  put_the_cream_cheese_in_the_bowl
)
SHORT=(spatial_bowl object_soup goal_cheese)
for i in 0 1 2; do
  setsid nohup .venv/bin/python /workspace/dqat/scripts/pi05/contact_sweep.py --checkpoint $CKPT \
    --task ${TASKS[$i]} --episodes $EPISODES --out $R/sweep/${SHORT[$i]} \
    > $R/logs/sweep_${SHORT[$i]}.log 2>&1 < /dev/null &
  setsid nohup .venv/bin/python /workspace/dqat/scripts/pi05/contact_sweep.py --checkpoint $CKPT \
    --task ${TASKS[$i]} --episodes $EPISODES --closed-loop a4_all --out $R/closed_a4_all/${SHORT[$i]} \
    > $R/logs/closed_a4_all_${SHORT[$i]}.log 2>&1 < /dev/null &
done
echo launched
