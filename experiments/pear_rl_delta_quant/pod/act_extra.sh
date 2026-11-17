#!/usr/bin/env bash
# Rerun MLP+lm_head after the name-matching fix (the first Bmlp_lm run hooked lm_head only -> saved as Blm).
. /workspace/lib.sh; cd /workspace
until grep -q STRIP_PHASE_DONE logs/act_phase2.log; do sleep 20; done
mv results/pilot/tf_Bmlp_lm.jsonl results/pilot/tf_Blm.jsonl 2>/dev/null
stop_server; rm -f logs/server_Bmlp_lm.log; ACTQ=mlp,lmhead PYTHONPATH=/workspace setsid /workspace/serve.sh Bmlp_lm models/arm_B --disable-cuda-graph < /dev/null > /dev/null 2>&1 &
for i in $(seq 1 300); do curl -s -m 3 127.0.0.1:30000/get_model_info | grep -q model_path && break; sleep 6; done
grep -m1 "ACTQ" logs/server_Bmlp_lm.log
CUDA_VISIBLE_DEVICES="" sim/bin/python score_tf.py --arm Bmlp_lm --out results/pilot/tf_Bmlp_lm.jsonl 2>&1 | grep -E "SCORE_DONE|Traceback"
stop_server; echo EXTRA_DONE
