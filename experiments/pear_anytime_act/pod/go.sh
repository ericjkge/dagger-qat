#!/bin/bash
# full pipeline after upload: setup, server, E1 probe (offline convergence), then baseline closed-loop arm with probes
cd /workspace; bash setup.sh > logs/setup.log 2>&1
uv pip install -q --python sim/bin/python tokenizers >> logs/setup.log 2>&1 || /root/.local/bin/uv pip install -q --python sim/bin/python tokenizers >> logs/setup.log 2>&1
mkdir -p results; tar xzf dflash2_main.tgz -C results --no-same-owner
. /workspace/lib.sh; start_server || exit 1
( . /workspace/simenv.sh; sim/bin/python probe_converge.py > logs/probe.log 2>&1 )
echo GO_PROBE_DONE
