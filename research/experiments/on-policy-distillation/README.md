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
- **2026-11-19, human request in session:** set up a clean repo skeleton for `main`, with none of the agent orchestration. It is on branch `skeleton` (5e4e3d9), off `origin/main`:
  - PEAR is a submodule at `third_party/pear`, pinned to faff474. The models are pinned at Hub revisions.
  - `configs/` holds the model files and one Hydra config per step. `scripts/` holds one script per step. `src/opd/` holds the eval, collect, SFT and distill entry points and the SFT dataset.
  - Step 4 uses verl's built-in on-policy distillation: `loss_mode=k1`, `use_policy_gradient=true`, `use_task_rewards=false`.
  - **2026-11-19, human:** compute limits for the skeleton lifted, as long as jobs stay sized for fairshare priority. Run no experiments yet. Push straight to `main` with no agent setup and no Claude co-author line.
  - `main` is now a4330ed, one squashed commit. It adds `configs/hardware/o2.sh`, because O2 has A100-80GB and L40S nodes but no H100/GB300, so PEAR's Hopper/Blackwell kernels, FA3 and LMCache are off. It also adds `scripts/slurm.sbatch`: a Ray cluster per Slurm job, single-node on gpu_quad or up to 5 nodes on gpu_mpi_quad, with notes on sizing requests.
  - The pipeline is renumbered without the baselines step: collect, SFT, distill, eval.
  - The remote `skeleton` branch is deleted.
- **2026-11-19, human:** make the pipeline transferable to any cluster, and make the README concise and outward-facing. `main` is now b6f46dd:
  - `scripts/run.sh` runs a step on one machine, inside a Slurm allocation, or on an existing Ray cluster. `scripts/ray_node.sh` starts one machine's Ray node by role.
  - `configs/hardware/generic.sh` covers any CUDA GPU; `GPU=h100` and `GPU=gb300` keep PEAR's tuned recipes.
  - The O2-specific `slurm.sbatch` is gone.
  - Target clusters: HMS O2 (A100, L40S, so `GPU=generic`) and Kempner on Cannon (H100, so `GPU=h100`). Both are reached through `sbatch [site flags] scripts/run.sh <step>`.
- **2026-11-19, brief edited (origin/eddy):** reworded the brief into Setup and Workflow sections; the teacher, students and steps are unchanged.
  - "Step 1: Baselines" (base Qwen3.5-9B/4B under PEAR's protocol) is still in the brief. In the repo that is `scripts/eval.sh` on a base student; there is no separate step.
- **2026-11-19, human:** asked for clean, reproducible code checked by rigorous tests, plus a `requirements.txt`. The CPU test suite (`repo_tests/run.sbatch` builds a fresh environment from `requirements.txt`) found three pipeline bugs:
  1. `distill.sh` failed to parse: an apostrophe inside `${VAR:?message}`.
  2. Hydra refused PEAR's configs as defaults of ours, because they set `hydra.searchpath`. The steps are now overlays (`configs/step/*.yaml`, `+step=<step>`) on PEAR's/verl's configs as the top-level config.
  3. Generic hardware inherited H100's 8 GPUs per node and 8-way sequence parallelism.
- Collection now saves the rollout's own stored tensors (`input_ids`, `position_ids`, `loss_mask`, `multi_modal_inputs`), because verl's transfer queue does not keep raw images. This also removes a wrong Qwen2-VL rotary-position recomputation.
- The suite passes, 70/70 with no warnings (job 60178211), on `main` 3871b03.
- **2026-11-19, brief rewritten as prose:** setup with π_teacher and π_θ, base-model baselines, teacher data, an SFT-only baseline, the method with the reverse-KL formula (k1, as in `configs/step/distill.yaml`), and generalization evaluation. The "Our method" section is still being written.
- **2026-11-19, brief:** "Our method" now holds the formal algorithm and is named the project's main contribution. A new ablation: on-policy distillation from the untuned base student as well as from the SFT checkpoint, to test whether the SFT warm start is needed. This needs no code: `scripts/distill.sh` with `MODEL=` set to the base model.
- **2026-11-19, human:** run a short GPU smoke test; keep the code clean; this is a shared codebase.
  - Raised `research/bin/submit` to one GPU machine per job (4 GPUs, 192 GB; origin/eddy defa9d4), per the earlier instruction to lift compute limits while staying fair to priority.
  - `smoke/` holds the jobs. Each runs `main`'s code cloned on scratch at a pinned commit.
    - `probe.sbatch` (60181075): driver, compute capability, network.
    - `setup.sbatch` (60181374; 60181101 was cancelled and resubmitted with --time=05:00:00 to fit the GPU-minute cap): `scripts/setup.sh` at 2dec0ee. Header: gpu / l40s:1 / 16 CPU / 64G / 5h. It installs PEAR's verl environment built for CUDA arch 8.0 and 8.9, LIBERO, and Qwen3.5-4B, then runs PEAR's checks.
    - `eval.sbatch`: 4 episodes, eval then collect with the 4B, then `check.py`.
  - The `setup.sh` fix (2dec0ee, not yet on `main`): it stops calling PEAR's `libero_run`, which downloads PEAR's 27B base model.
- Correction: PEAR's continuous-token context keeps every round's reasoning under either chat template. The teacher's template matters for its reasoning-effort system instruction.
- **2026-11-19, human:** removed `research/baseline.md` and the baseline requirement (origin/eddy 90000b2). Comparisons are stated per result.
- **Tokenizer check (jobs 60171018, 60171577):** a 15-minute CPU job, `tokenizer_check/check.py`; two records in `runs/on-policy-distillation.jsonl`.
  - Teacher and both students share the vocabulary (248,320 entries), the merges and the normalizer.
  - The teacher adds 7 audio/TTS special tokens (ids 248070–248076) that the students lack. LIBERO episodes never use them.
  - The pre-tokenizer regex differs in one place: the students' regex keeps Unicode combining marks (`\p{M}`) attached to letters, and the teacher's does not. `trim_offsets` also differs, but it changes character offsets only, not ids.
  - So ASCII text such as LIBERO instructions, action JSON and English reasoning gets the same ids under both. In distillation the teacher scores the student's token ids directly, so ids carry the same meaning on both sides.
  - **Verdict:** compatible for this task.
  - The vla environment's `tokenizers` could not load the teacher's file, so the check compares the `tokenizer.json` fields rather than encoding sample texts.
- **The chat template is the real mismatch** (compared the Hub files directly). The teacher's template differs from Qwen3.5's in two ways:
  1. It adds a reasoning-effort instruction to the system prompt: xhigh by default, or medium or low.
  2. It keeps every earlier round's reasoning in the context (`preserve_thinking`, default true). Qwen3.5's template drops reasoning from turns before the last user query.

  **2026-11-19, human decision:** the students use the teacher's chat template (`main` 9bc3873). The student model files set `chat_template_file` to the teacher's pinned `chat_template.jinja`, and `opd` passes it to verl's `actor_rollout_ref.model.custom_chat_template`.
- No jobs have been submitted, because nothing can run yet; see "Blocked on".

## Results

None yet.

## Next step

When the PEAR code and teacher checkpoint are available, submit the Step 1 evaluation of base Qwen3.5-9B on LIBERO (2,000 episodes, PEAR protocol), recording time per episode. In the same job, time a teacher episode at xhigh so the latency axis has a measured reference.

## Blocked on

- **Compute:** `research/bin/submit` allows 1 GPU, 8 GB RAM and 12 h per job. PEAR's recipes assume multi-node Ray clusters, and the 27B teacher alone needs about 54 GB in bf16. No pipeline step fits the current limits.
- **Code and checkpoints (superseded by the `skeleton` branch for the code):**
  - The PEAR repo (LIBERO env, prompt format, RL trainer) is not on the cluster. I searched `/n/data1/.../edk005` and `/n/scratch/users/e/edk005`.
  - The teacher checkpoint is not on the cluster either.
  - This branch has no `scripts/`, `configs/` or `src/`.
  - I need the repo URL or location and the checkpoint path. I'm not guessing either one.
- **Latency on the main plot:** time per episode has to be measured on our hardware for the teacher as well as the students, because the paper doesn't report it. Confirm one teacher inference GPU fits within the submit limits.

## Dead ends
