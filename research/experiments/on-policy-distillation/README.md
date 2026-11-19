---
slug: on-policy-distillation
status: blocked
hypothesis: On-policy reverse-KL distillation from the PEAR 27B teacher makes a Qwen3.5-9B student succeed more often on LIBERO than SFT alone, and more often than the teacher at low effort, while taking less time per episode than the teacher at xhigh.
kill_criterion: After on-policy distillation, the 9B student's LIBERO success rate (PEAR protocol, 2,000 episodes) is not more than 2 SE above its SFT-only checkpoint, or not above 21.9%, or its mean time per episode is not below the teacher's at xhigh.
best_metric:
started: 2026-11-19
---

# on-policy-distillation

## Hypothesis

On-policy reverse-KL distillation from Qwen3.8-27B-PEAR-LIBERO_only gives a Qwen3.5-9B student a higher LIBERO success rate than its SFT-only checkpoint and than the teacher at low effort (21.9%), at lower mean time per episode than the teacher at xhigh.

## Kill criterion

The direction is dead if the on-policy-distilled 9B student, evaluated with PEAR's protocol (2,000 episodes: 40 tasks × 50 initial states), fails any of these:
1. Its success rate is more than 2 SE above its own SFT-only checkpoint.
2. Its success rate is above 21.9%, the teacher's paper number at low effort.
3. Its mean wall-clock time per episode is below the teacher's at xhigh, measured on the same hardware.

The 4B student is reported alongside the 9B student but does not decide the direction.

## Current state

- **2026-11-19, brief updated (origin/eddy 2fa8b2c).** The placeholder brief was replaced with a concrete pipeline, and this note now follows it:
  - **Goal:** distill the PEAR robot-reasoning model (arXiv 2611.18562) to cut reasoning latency.
  - **Teacher:** Qwen3.8-27B-PEAR-LIBERO_only.
  - **Students:** Qwen3.5-9B, then Qwen3.5-4B.
  - **Pipeline:**
    1. Evaluate the base students. The teacher's numbers come from the paper: 74.4% at xhigh, 21.9% at low effort.
    2. Collect about 1,000 successful teacher episodes across the 4 LIBERO suites, keeping the full multi-round contexts.
    3. Run SFT on all generated tokens.
    4. Starting from the SFT checkpoint, run on-policy distillation. Each step, the student runs episodes, the teacher prefills them, and the per-token advantage is the reverse KL. There is no task reward, one update per step, and an evaluation every 10 steps.
    5. Run the final evaluation on LIBERO, plus LIBERO-Pro (teacher 57.4%) and LIBERO-Plus. The main plot is success rate against time per episode.
- I wrote the hypothesis and kill criterion above from this brief, before any run.
- No jobs have been submitted, because nothing can run yet; see "Blocked on".

## Results

None yet.

## Next step

When the PEAR code and teacher checkpoint are available, submit the Step 1 evaluation of base Qwen3.5-9B on LIBERO (2,000 episodes, PEAR protocol), recording time per episode. In the same job, time a teacher episode at xhigh so the latency axis has a measured reference.

## Blocked on

- **Baselines file:** `research/baseline.md` on origin/eddy is still empty. The brief's Step 1 names the baselines: base 9B and 4B, and teacher at 74.4% (xhigh) and 21.9% (low). Please confirm these are the binding baselines, or fill in `baseline.md`.
- **Code and checkpoints:**
  - The PEAR repo (LIBERO env, prompt format, RL trainer) is not on the cluster. I searched `/n/data1/.../edk005` and `/n/scratch/users/e/edk005`.
  - The teacher checkpoint is not on the cluster either.
  - This branch has no `scripts/`, `configs/` or `src/`.
  - I need the repo URL or location and the checkpoint path. I'm not guessing either one.
- **Latency on the main plot:** time per episode has to be measured on our hardware for the teacher as well as the students, because the paper doesn't report it. Confirm one teacher inference GPU fits within the submit limits.

## Dead ends
