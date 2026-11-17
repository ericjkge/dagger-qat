#!/bin/bash
cd /workspace && bash setup3.sh > logs_setup.txt 2>&1
. /workspace/lib.sh; cd /workspace; mkdir -p results/pilot
cp /workspace/strip_A.jsonl results/pilot/
( sim/bin/python calib.py > logs/calib.log 2>&1 ) &
CP=$!
start_server A || { echo SERVER_FAIL; exit 1; }
wait $CP; grep -E "state-units|CALIB_DONE|Error|Traceback" logs/calib.log | tail -n 20
UPC=$(grep "state-units" logs/calib.log | head -1 | sed 's/.*(x,y,z): \[ *\([0-9.]*\).*/\1/'); UPC=$(python3 -c "print(float('$UPC')/100)")
sim/bin/python make_transform.py $UPC
CUDA_VISIBLE_DEVICES="" sim/bin/python summarize.py 2>&1 | tail -n 2
CUDA_VISIBLE_DEVICES="" sim/bin/python score_ctx.py 2>&1 | grep -v -i "warn\|gym\|task orders"
stop_server; echo RUN3_DONE
