#!/usr/bin/env bash
# Closed-loop launcher for extra configs (pod): CFGS="a4_rotdown a4_rot" bash launch_closed.sh
# Same 20 initial states, env seed and per-call noise schedule as the BF16 sweep (paired episodes).
set -euo pipefail
source /workspace/env.sh
R=${R:-/workspace/results/2026-11-14-pi05-contact}
CKPT=/workspace/ckpt/pi05_libero_pytorch
EPISODES=${EPISODES:-20}
TASKS=(
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_alphabet_soup_and_place_it_in_the_basket
  put_the_cream_cheese_in_the_bowl
)
SHORT=(spatial_bowl object_soup goal_cheese)
for c in ${CFGS:?}; do
  for i in 0 1 2; do
    setsid nohup .venv/bin/python /workspace/dqat/scripts/pi05/contact_sweep.py --checkpoint $CKPT \
      --task ${TASKS[$i]} --episodes $EPISODES --closed-loop $c --out $R/closed_$c/${SHORT[$i]} \
      > $R/logs/closed_${c}_${SHORT[$i]}.log 2>&1 < /dev/null &
  done
done
echo launched
