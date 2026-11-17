#!/bin/bash
cd /workspace && bash setup2.sh > logs_setup.txt 2>&1 && bash act_phase.sh > logs/act_phase.log 2>&1
