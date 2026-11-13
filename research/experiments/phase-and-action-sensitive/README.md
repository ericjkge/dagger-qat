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
Specification (human, 2026-11-13):
- A is the rolled-out action. I read this as the 8×7 action chunk the
  full-precision policy outputs at frames of its own closed-loop LIBERO
  rollouts. OFT executes all 8 steps per call. So phase tags and sampled frames
  come from FP rollouts, not training demos.
- Quantizable layers: the vision backbone (the fused DINOv2 + SigLIP ViT
  linears) and the Llama linear layers. The projector, embeddings, LM head,
  proprio projector and action head stay unquantized.
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
  header (reuses the partly built env). FAILED after 9 min: `egl_probe` built,
  but `pip install -e .` refused because `dqat`'s pyproject pins Python 3.10.
- 59510797, the same script without installing `dqat`, which is imported from
  `src/` via PYTHONPATH. Same header. FAILED at the import check after 46 s:
  the editable LIBERO install is not importable as `libero`. The packages were
  installed.
- 59511042, the same script with `src`, `third_party/openvla-oft` and
  `third_party/LIBERO` on PYTHONPATH, as in `scripts/common.sh`. Same header.

Proposed first-pass choices (awaiting confirmation):
- Unit W_j = one whole linear layer: Llama 32 blocks × 7, DINOv2 ViT-L 24 × 4,
  SigLIP SO400M 27 × 4, about 430 units. Each layer is quantized by RTN with
  groups of 128 at its assigned bit-width.
- A = the full 8×7 action chunk in OFT's normalized action space (q01/q99 to
  [−1, 1]), all 56 entries weighted equally. Random projections u ~ N(0, I_56).

`runs/phase-and-action-sensitive.jsonl` is empty.

## Results

None.

## Next step

Once the env check passes and the blockers are cleared: download the OFT
checkpoint, run a few full-precision LIBERO-Long rollouts on one GPU, and tag
phases from their gripper traces. Report phase-length statistics per task to
confirm the four-phase split is well defined before any gradient or
quantization work.

## Blocked on

1. **Unspecified parts of the brief.** Baseline and evaluation sections are
   empty. Also open: unit granularity of W_j; weighting within A (steps,
   gripper dimension, normalized units); the quantizer Q_b; phase-window
   definitions; per-unit vs global worst phase and phase normalization; greedy
   vs exact allocation; the meaning of b̄ and its values; per-phase failure
   attribution; k and frames per phase; rollout count and seeds for collecting
   frames.
2. **No weights on O2.** The OFT checkpoint still needs downloading. Training
   demos are no longer needed for frames, because A comes from rollouts.

## Dead ends

None yet.
