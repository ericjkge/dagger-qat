---
slug: phase-and-action-sensitive
status: running
hypothesis: At a matched average of 4 bits per weight, worst-phase action-sensitive mixed-precision allocation raises OpenVLA-OFT LIBERO-Long success over uniform AWQ W4/g128, with the gain concentrated in grasp/place episodes.
kill_criterion: At b̄=4, mean success over ≥3 seeds fails to beat AWQ W4/g128 by more than 2 SE, or at b̄=3 it fails to beat uniform 3-bit AWQ by more than 2 SE.
best_metric:
started: 2026-11-13
---

# phase-and-action-sensitive

## Hypothesis

At a matched average of 4 bits per quantized weight, allocating bits from
{2,3,4,8} by worst-phase action sensitivity
(s = max_p mean_p ‖g_j (Q_b(W_j) − W_j)‖², phases approach/grasp/transport/place
tagged by gripper flips) gives higher LIBERO-Long success than uniform AWQ
W4/g128, and the gain comes mainly from grasp and place phases.

## Kill criterion

Fixed 2026-11-13, before any run. Comparisons use the paired protocol in the
top-level `README.md` (OpenVLA-OFT `libero_10`, per-episode seeding):

- At b̄ = 4: mean success over ≥ 3 seeds is not above AWQ W4/g128 by more than
  2 SE of the paired difference — killed. (The repo's single-seed reference is
  93.4% for both FP BF16 and AWQ W4/g128, so headroom at 4 bits may be ~0;
  hence the second arm.)
- At b̄ = 3: same test against uniform 3-bit AWQ (g128). If neither budget
  shows a gain, the direction is dead.
- Diagnostic, not a kill test: per-phase failure attribution, and an ablation
  replacing max_p with the phase-agnostic mean, to tell whether any gain comes
  from the phase split or merely from sensitivity-weighted allocation.

## Current state

2026-11-13: first turn. Read the brief and `baseline.md` from `origin/eddy`.
Redirection (human, 2026-11-13): ignore `research/baseline.md` for this
direction. Baseline arms are taken from the top-level `README.md` protocol: FP
BF16 and AWQ W4/g128, plus uniform 3-bit AWQ g128 for the b̄=3 arm. The kill
criterion is now fixed against these.
Redirection (human, 2026-11-13): the project direction has changed; the
Cannon-era repo is now only a skeleton to repurpose for O2. On the human's
instruction, `research/bin/submit` on this branch no longer has a per-job time
limit (it still requires `--time`), and its RAM cap went from 8G to 128G.
Submodules `third_party/openvla-oft` and `third_party/LIBERO` are checked out in
this worktree.

Jobs:
- 59508441, `setup_env.sbatch`: creates the conda env `vla` (Python 3.11,
  `requirements-runtime.txt`, editable OFT/LIBERO/dqat from this worktree) and
  checks imports. Header: `--partition=short --cpus-per-task=4 --mem=16G
  --time=03:00:00`. Log:
  `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/logs/setup-59508441.out`.

`runs/phase-and-action-sensitive.jsonl` is empty.

## Results

None.

## Next step

Once the blockers are cleared: a single CPU job that tags phases on the
LIBERO-Long training HDF5 demos by gripper-state flips and reports phase-length
statistics per task, to confirm the four-phase split is well defined before any
gradient or quantization work.

## Blocked on

1. **New direction.** The human says the direction has changed completely. The
   hypothesis and kill criterion above follow the current brief and will be
   replaced once the brief on `origin/eddy` is updated.
2. **No weights or demos on O2.** The OFT checkpoint and LIBERO HDF5 demos still
   need downloading (`scripts/download_assets.py`) once the new direction
   confirms they are needed.

## Dead ends

None yet.
