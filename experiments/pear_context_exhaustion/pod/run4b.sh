#!/bin/bash
. /workspace/lib.sh; cd /workspace; mkdir -p results/pilot logs/calib
for f in results/dflash2_main/libero_*.jsonl; do e=$(basename $f .jsonl)
  ( sim/bin/python calib.py $e > logs/calib/$e.log 2>&1; echo "rc=$?" >> logs/calib/$e.log ) &
done
wait
for f in logs/calib/*.log; do e=$(basename $f .log)
  for t in 1 2 3; do grep -q "rc=0" logs/calib/$e.log && break
    echo "retry $e $t"; sim/bin/python calib.py $e > logs/calib/$e.log 2>&1; echo "rc=$?" >> logs/calib/$e.log; done
done
cat results/pilot/cards_*.jsonl > results/pilot/cards.jsonl
grep -h "state-units\|rc=" logs/calib/*.log; tail -n 2 logs/summ.log
UPC=$(grep -h "state-units" logs/calib/*.log | head -1 | sed 's/.*(x,y,z): \[ *\([0-9.]*\).*/\1/'); UPC=$(python3 -c "print(float('$UPC')/100)")
sim/bin/python make_transform.py $UPC
CUDA_VISIBLE_DEVICES="" sim/bin/python score_ctx.py 2>&1 | grep -v -i "warn\|gym\|task orders"
stop_server; echo RUN4_DONE
