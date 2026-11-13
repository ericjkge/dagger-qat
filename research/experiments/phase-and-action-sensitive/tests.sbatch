#!/usr/bin/env bash
#SBATCH --job-name=vla-tests
#SBATCH --partition=gpu,gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/tests-%j.out
# Correctness checks (tests.py), then every pipeline stage end to end on a tiny
# scale in a scratch root: score -> allocate -> recalibrate -> eval (1 episode per arm).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source research/experiments/phase-and-action-sensitive/env.sh
T=$OUT/tests/$SLURM_JOB_ID
rm -rf $T  # requeued: start clean
python $EXP/tests.py --checkpoint $CKPT --frames $OUT/demo-frames/59511837/frames.npz --out $T
python $EXP/score.py --checkpoint $CKPT --frames $T/frames.npz --out $T/scores/s0.npz --k 2 --seed 0
for mode in worst-phase phase-agnostic; do
  python $EXP/allocate.py --scores $T/scores/s0.npz --mode $mode --budget 3 --out $T/alloc/${mode%%-*}-s0.json
done
python $EXP/recalibrate.py --checkpoint $CKPT --frames $T/frames.npz --alloc $T/alloc/worst-s0.json \
  --out $T/heads/worst-s0.pt --val-frac 0.25 --steps 20 --batch 4 --lr 3e-5 --eval-every 10 --seed 0
python $EXP/eval.py --checkpoint $CKPT --root $T --arms fp uniform worst-s0 worst-s0-head \
  --tasks 0 --episodes 1 --seed 7 --uniform-bits 3 --phase-window 10 --flip-min-hold 5
python - "$T" <<'PY'
import json, sys
from pathlib import Path
T = Path(sys.argv[1])
for mode in ("worst", "phase"):
    a = json.loads((T / "alloc" / f"{mode}-s0.json").read_text())
    assert a["mean_bits"] <= 3 and len(a["bits"]) == 420, (mode, a["mean_bits"], len(a["bits"]))
    print(f"PASS alloc {mode}: mean_bits {a['mean_bits']:.3f}, layers_per_bit {a['layers_per_bit']}, worst_phase {a['worst_phase_counts']}")
for arm in ("fp", "uniform", "worst-s0", "worst-s0-head"):
    rows = [json.loads(l) for l in (T / "eval" / f"{arm}.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["arm"] == arm, arm
    print(f"PASS eval {arm}: success {rows[0]['success']}, steps {rows[0]['action_steps']}, {rows[0]['seconds']} s, last_phase {rows[0]['last_phase']}")
r = json.loads((T / "heads" / "worst-s0.json").read_text())
print(f"PASS recalibration ran: val L1 {r['val_l1_quantized_head']:.4f} -> {r['val_l1_recalibrated_head']:.4f}")
print("END-TO-END PASSED")
PY
