#!/bin/bash
. /workspace/lib.sh; cd /workspace; rm -f results/pilot/cardsf_*.jsonl
for f in results/dflash2_main/libero_*.jsonl; do e=$(basename $f .jsonl)
  for t in 1 2 3 4; do sim/bin/python calib_fast.py $e > logs/cf_$e.log 2>&1 && grep -q CALIB_DONE logs/cf_$e.log && break; done
  grep -h "units/m" logs/cf_$e.log || echo "FAILED $e"
done
cat results/pilot/cardsf_*.jsonl > results/pilot/cards.jsonl
sim/bin/python make_transform.py
until grep -q "summ rc" logs/summ.log; do sleep 15; done; tail -n 2 logs/summ.log
CUDA_VISIBLE_DEVICES="" sim/bin/python score_ctx.py 2>&1 | grep -v -i "warn\|gym\|task orders"
stop_server; echo RUN5_DONE
