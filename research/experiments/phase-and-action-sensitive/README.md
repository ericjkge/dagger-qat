---
slug: phase-and-action-sensitive
status: blocked
hypothesis: At a matched average of 4 bits per weight, worst-phase action-sensitive mixed-precision allocation raises OpenVLA-OFT LIBERO-Long success over uniform AWQ W4/g128, with the gain concentrated in grasp/place episodes.
kill_criterion: (proposed, awaiting baseline.md) At b̄=4, mean success over ≥3 seeds fails to beat AWQ W4/g128 by more than 2 SE, or at b̄=3 it fails to beat uniform 3-bit AWQ by more than 2 SE.
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

Proposed, not yet fixed — it depends on a baseline that `baseline.md` does not
yet define. To be fixed before the first run:

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
No jobs submitted; `runs/phase-and-action-sensitive.jsonl` is empty. The
direction cannot start; see Blocked on.

## Results

None.

## Next step

Once the blockers are cleared: a single CPU job that tags phases on the
LIBERO-Long training HDF5 demos by gripper-state flips and reports phase-length
statistics per task, to confirm the four-phase split is well defined before any
gradient or quantization work.

## Blocked on

1. **No baseline or evaluation protocol.** `research/baseline.md` on
   `origin/eddy` is empty, and the brief's "Baseline" and "Evaluation data"
   sections are unfinished ("We will test the method on", "----"). Please
   specify the baseline arms (proposal: FP BF16, AWQ W4/g128, uniform AWQ 3-bit,
   as in the top-level README protocol) and the bit budget(s) b̄.
2. **No runnable environment.** The conda env `vla` named in
   `research/CLAUDE.md` does not exist (`~/.conda/envs` has only `big_things`,
   `nrfm`). `third_party/openvla-oft` and `third_party/LIBERO` are empty in this
   worktree and in the main checkout. No `.venv-runtime`, OFT checkpoint or LIBERO
   demos were found, and `/n/scratch/users/e/edk005/vla/` does not exist. The
   `scripts/train.py`/`sample.py`/`eval.py`/`make_data.py` and `configs/` named in
   `research/CLAUDE.md` do not exist; the repo's entry point is
   `scripts/run_baseline.sh` via `cannon_eval.sbatch`.
3. **Memory limit vs. a 7B model.** `research/bin/submit` caps jobs at 8G RAM.
   The existing eval sbatch requests 64G. Loading OpenVLA-7B (≈15 GB BF16) and
   running LIBERO under 8G of host RAM is unlikely to work. Gradient estimation
   (step 2 of the brief) also needs the BF16 model and backprops on one GPU.
   Raising the RAM cap, or confirming that direct-to-GPU loading is expected, is
   needed.
4. **Compute for evaluation.** One full LIBERO-Long evaluation (500 episodes) is
   ~12 h on one A100, which is at the per-job limit. Three seeds × several arms is
   many such jobs at 2 concurrent. Please confirm the episode budget per arm, or
   whether a reduced subset (e.g. 10 tasks × 20 states) is acceptable.

## Dead ends

None yet.
