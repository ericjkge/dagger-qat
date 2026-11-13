---
slug: phase-and-action-sensitive
status: running
hypothesis: At matched average bits b̄, worst-phase action-sensitive bit allocation over {2,3,4,8} gives higher LIBERO-Long success than the same allocation using the phase-agnostic mean, with the gain concentrated in grasp/place.
kill_criterion: (pending baseline in brief)
best_metric:
started: 2026-11-13
---

# phase-and-action-sensitive

## Hypothesis

At matched average bits b̄, allocating bits from {2,3,4,8} by worst-phase
action sensitivity (s = max_p mean_p ‖g_j (Q_b(W_j) − W_j)‖², phases
approach/grasp/transport/place tagged by gripper flips) gives higher
LIBERO-Long success than the same allocation driven by the phase-agnostic mean,
and the gain comes mainly from grasp and place.

## Kill criterion

Not yet fixed. The brief's Baseline section is empty and the AWQ baseline is
out of scope (see current state). The criterion will be fixed against the
brief's baseline before the first evaluation run. Proposed: at each b̄ tested,
if mean success over ≥ 3 seeds is not above the phase-agnostic-mean allocation
and uniform b̄-bit quantization by more than 2 SE of the paired difference, the
direction is dead.

## Current state

2026-11-13: first turn. Read the brief and `baseline.md` from `origin/eddy`.
Redirection (human, 2026-11-13): ignore `research/baseline.md` for this
direction.
Redirection (human, 2026-11-13): ignore the AWQ baseline too; it is an artifact
of the old project. The earlier kill criterion against AWQ W4/g128 is
withdrawn. No run had used it.
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
  FAILED after 4 min: CMake 4 refuses to build `egl_probe`, which declares
  CMake < 3.5.
- 59510142, the same script with `CMAKE_POLICY_VERSION_MINIMUM=3.5` and the same
  header (reuses the partly built env).

`runs/phase-and-action-sensitive.jsonl` is empty.

## Results

None.

## Next step

Once the blockers are cleared: a single CPU job that tags phases on the
LIBERO-Long training HDF5 demos by gripper-state flips and reports phase-length
statistics per task, to confirm the four-phase split is well defined before any
gradient or quantization work.

## Blocked on

1. **Unspecified parts of the brief.** Baseline and evaluation sections are
   empty. Also open: unit granularity of W_j, what A is (8×7 chunk? gripper
   weight?), the quantizer Q_b, phase-window definitions, per-unit vs global
   worst phase and phase normalization, greedy vs exact allocation, the meaning
   of b̄ and its values, and per-phase failure attribution.
2. **No weights or demos on O2.** The OFT checkpoint and LIBERO HDF5 demos still
   need downloading (`scripts/download_assets.py`) once the new direction
   confirms they are needed.

## Dead ends

None yet.
