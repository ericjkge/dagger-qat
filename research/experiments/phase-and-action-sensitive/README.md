---
slug: phase-and-action-sensitive
status: running
hypothesis: At matched average bits b̄, worst-phase action-sensitive bit allocation over {2,3,4,8} gives higher LIBERO-Long success than the same allocation using the phase-agnostic mean, with the gain concentrated in grasp/place.
kill_criterion: At b̄ = 3, if mean LIBERO-Long success over ≥3 seeds does not exceed both uniform b̄-bit RTN and the phase-agnostic-mean allocator by more than 2 SE of the paired per-episode difference, the direction is dead.
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

Fixed 2026-11-13, before any run. Baselines (human, 2026-11-13), all using the
same symmetric RTN g128 quantizer and the same quantizable layers:

- **uniform**: every quantizable layer at b̄ bits;
- **phase-agnostic**: the same allocator with max_p replaced by the mean over
  all sampled frames (same frames, same k);
- **full precision**: unquantized BF16, the ceiling.

Episodes are paired across arms. Budget b̄ = 3 bits (human, 2026-11-13; briefly 4 earlier that day, before any run). If mean success over ≥ 3
seeds does not exceed **both** uniform and phase-agnostic by more than 2 SE of
the paired per-episode difference, the direction is dead. Beating uniform but
not phase-agnostic means sensitivity allocation helps but the phase split does
not; that also kills this direction, which is about the phase split.

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
  COMPLETED in 2.5 min: `vla env OK`, torch 2.2.0+cu121, transformers 4.40.1,
  libero_10 has 10 tasks. Jobs must export that PYTHONPATH.

Proposed first-pass choices (awaiting confirmation):
- Unit W_j = one whole linear layer: Llama 32 blocks × 7, DINOv2 ViT-L 24 × 4,
  SigLIP SO400M 27 × 4, about 430 units. Each layer is quantized by RTN with
  groups of 128 at its assigned bit-width.
- Quantizer (human, 2026-11-13): symmetric. Per group of 128,
  s = max|w| / (2^(b−1) − 1), q ∈ [−(2^(b−1)−1), 2^(b−1)−1]. 2-bit is
  therefore ternary {−s, 0, s}. One scale per group, no zero point.
- 8 bits = real symmetric INT8 with the same quantizer (human, 2026-11-13). It
  costs 8 bits per weight in b̄. Every quantizable layer is quantized in every
  arm.
- A = the full 8×7 action chunk in OFT's normalized action space (q01/q99 to
  [−1, 1]), all 56 entries weighted equally. Random projections u ~ N(0, I_56).

- 59511606, `fp_rollouts.sbatch` (FP rollouts as the frame source). Cancelled
  while still pending, after the redirection below. It never ran.
- 59511837, `demo_frames.sbatch`: downloads the pinned checkpoint and the OFT
  RLDS demos, tags phases on the demos and samples 64 frames per phase. Header:
  `--partition=short --cpus-per-task=4 --mem=32G --time=04:00:00`.

- 59512608 `score.sbatch` and 59512609 `eval.sbatch`: cancelled while pending
  to add head recalibration. Resubmitted as 59513187 (score + recalibration)
  and 59513188 (eval worker job A); headers in `results.md`.

Human (2026-11-13): run action-head recalibration and see whether it works.
Arm `uniform-head`: the uniform 3-bit backbone with only the L1 head fine-tuned
on demo frames to match BF16 action chunks. Paired against `uniform`. This is
a candidate method beyond the brief, not a phase test.

Human (2026-11-13): results needed by Saturday 2026-11-14 early morning.
Agent's reading: the policy and simulator are deterministic, so "seed" means
the calibration seed (frame sample + projections), 7/8/9. The kill criterion's
"≥ 3 seeds" means 3 calibration seeds.

Redirection (human, 2026-11-13): the project is moving away from on-policy, so
frames come from the true demonstrations, not FP rollouts. This replaces the
earlier rollout frame source. A is now the policy's predicted 8×7 chunk at
demonstration frames. The demos are OFT's own fine-tuning data,
`openvla/modified_libero_rlds` `libero_10_no_noops` @ `6ce6aaa` (256 px, model
orientation, no-op steps removed). Phases are tagged from the demo gripper
actions. No init states need holding out, so evaluation can use all 50.

Human (2026-11-13): use reasonable first-pass defaults for frame collection and
the score estimator, and record the exact setup of every run. That record is
`results.md` in this directory. Suite: LIBERO-Long, the agent's recommendation;
the human did not choose otherwise.

Related work: QVLA (arXiv 2602.03782, ICLR 2026) already allocates bits by
action sensitivity in OpenVLA-OFT on LIBERO. It works per channel, uses bits
{0,2,4,8,16}, greedy demotion, a σ²·‖J‖_F² first-order proxy, and frames from
training demos averaged over the data. It has no phase split. What this
direction adds is the worst-phase max over gripper-tagged phases, on-policy
and per-phase evaluation. Frames now come from demos, as in QVLA. The phase-agnostic arm is the
QVLA-like control at layer granularity.

## Results

None.

## Next step

Check the first worker and score logs for errors. When 59513187 ends, submit
worker job B (`eval.sbatch`).

## Blocked on

1. **Still open (needed before allocation/evaluation, not before the current
   job):** evaluation episodes/seeds; per-phase failure attribution. Worst
   phase (per layer, unnormalized) and greedy allocation were fixed by the human
   on 2026-11-13. First-pass defaults are listed in `results.md`.
2. **No weights on O2.** The OFT checkpoint still needs downloading. Training
   demos are no longer needed for frames, because A comes from rollouts.

## Dead ends

None yet.
