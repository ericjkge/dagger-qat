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
  - **Human: make the GPU test as short as possible while still covering the full pipeline.** `smoke/` now holds two jobs, each running `main`'s code cloned on scratch at a pinned commit (a478227):
    - `setup.sbatch` (60185384). Header: short / 16 CPU / 64G / 4h, no GPU. It installs PEAR's verl environment built for CUDA arch 8.9 only, plus LIBERO and Qwen3.5-4B. It replaced GPU setup jobs (60181670 and earlier), which every GPU partition estimated would start the next day; `setup.sh` now builds without a GPU given `CUDA_ARCH` (bc23adb).
    - Setup completed in 57 min with a peak RSS of 48 GB. PEAR's CPU environment tests passed (18). Qwen3.5-4B is pinned at 851bf6e (8.8 GB); the verl environment is 14 GB.
    - **Human: split the test and run it as soon as possible without priority penalties.** The single 3-GPU pipeline job (60185385) is replaced by two jobs on `gpu_requeue`, which is preemptible and billed at 0.1 per GPU instead of 5. Both run code bc23adb and share outputs in `runs/smoke-bc23adb`.
      - `part1.sbatch` (60188766). Header: gpu_requeue / l40s:1 / 8 CPU / 64G / 1h15. PEAR's GPU and LIBERO checks, collect 2 episodes (all kept), SFT 2 updates with the optimizer offloaded to CPU, then eval of the SFT model.
      - `part2.sbatch` (60188767, after part 1). Header: gpu_requeue / l40s:3 / 12 CPU / 80G / 1h15. Distill 2 updates plus a validation pass, export, then eval of the distilled model.
      - `check.py` verifies each part's steps.
  - Cancelled the probe (60181075) and the earlier setup jobs (60181101, 60181374). The account's GPU-minute cap (`GrpTRESRunMins gres/gpu=12000`, nearly all held by learned-hierarchy jobs) makes small, short requests start sooner.
  - Fixes found while building the smoke test (a478227):
    - `setup.sh` no longer calls PEAR's `libero_run`, which downloads PEAR's 27B base model.
    - SFT now also saves a Hugging Face model, which distillation and eval need.
    - `collect.min_score` sets which episodes are kept (default 1, successes only).
    - TensorBoard logs go to the run directory, not the repo.
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

- **2026-11-20, brief updated (origin/eddy, Algorithm 1 reformatted).** The algorithm is unchanged: the brief now numbers it as "Algorithm 1. On-policy distillation" and fixes the "epidoes" typo. I audited `main`'s distill step against it, reading verl `trainer/distillation/losses.py` and PEAR's `configs/{base,training}.yaml`:
  - **Matches:**
    - The advantage: `k1` gives log π_θ − log π_teacher, and the advantage is its negative, detached, per token, with no discounting.
    - No task reward, a token-mean loss, one update per step (mini-batch equal to the batch), and B episodes per step (`rollout.n=1`).
    - The teacher scores the student's own token sequence.
    - An evaluation every 10 steps, plus one before training.
    - Both student initialisations: `MODEL` is either the SFT checkpoint or the base model.
  - **Gaps:**
    1. PEAR's separate-async trainer trains on episodes up to 4 policy versions old (`max_off_policy_threshold: 5`), and an episode can continue across a weight update. Algorithm 1 samples every episode from π_θk. `trainer.v1.sampler.max_off_policy_threshold=1` would keep only episodes generated entirely under the current version, at a throughput cost. This needs a human decision.
    2. `vanilla` is PPO's clipped ratio against the sampler's log-probs. With fresh rollouts and one update, the ratio is about 1, differing only by the vLLM/FSDP numeric mismatch, so the gradient is A_t ∇log π_θ up to that factor.
    3. Still untested on GPU: the teacher prefill on multimodal inputs. The smoke test's part 2 covers it.

- **2026-11-20, brief updated again (origin/eddy).** Algorithm 1's inner steps are renumbered 1–5 (rollout, teacher prefill, advantages, update, evaluate). Nothing else changed, and neither does the audit above.
- **2026-11-21, smoke part 1 failed on a config error; fixed and resubmitted.** Part 1 (60188766) started at 02:55 on an L40S. Its PEAR GPU and LIBERO checks passed, but collect stopped at config composition after 3.5 minutes: `scripts/common.sh` passed `+…env_vars.TENSORBOARD_DIR`, and PEAR's `base.yaml` already sets that key (to `${run_dir}/tensorboard`, which is already the run directory), so Hydra refused the `+`. The raylet SIGKILL in the log is only Ray's teardown. Part 2 (60188767) was cancelled by its dependency.
  - The CPU suite does catch it (`test_eval_and_collect_run_one_pass`). The earlier passing run predates the override.
  - Fix: the override is removed, and a test now checks that `TENSORBOARD_DIR` is the run's `tensorboard/`. I folded the fix into the unpushed skeleton commit; `skeleton-local` is now 026aef1 (ebf7349, 58a51e7, 026aef1 on origin/main bcaded2).
  - Resubmitted on 026aef1 with unchanged headers: CPU suite 60335813 (short), part 1 60335814 (gpu_requeue, l40s:1), part 2 60335815 (gpu_requeue, l40s:3, afterok).
- **2026-11-21, CPU suite passed; part 1 hung on Ray CPUs; fixed and resubmitted.** The CPU suite (60335813) passed 70 of 70 on 026aef1. Part 1 (60335814) started at 03:05. PEAR's checks passed in 146 s, then collect hung with the GPU idle: PEAR's `base.yaml` gives the data queue 32 storage units (verl's default is 8), and each takes one Ray CPU (`@ray.remote(num_cpus=1)` in `transfer_queue/storage/simple_storage.py`). The job has 8 CPUs, so 7 units started, 24 stayed `PENDING_CREATION`, and the runner waited on them forever. I cancelled 60335814 and 60335815 at 03:40, 35 minutes in.
  - Smoke fix (here): `env.sh` sets `UNITS=transfer_queue.backend.SimpleStorage.num_data_storage_units=2`, and every Ray stage passes it. It composes in both `embodied` and `training`.
  - Main: a README note that Ray needs more than 32 CPUs, or a lower `num_data_storage_units` (65c78b8). Full runs should request more than 32 CPUs per Ray cluster or set this override. A snapshot resumes only with the same number of units, so a run keeps one value throughout.
  - Resubmitted on 65c78b8 (026aef1 plus the README): part 1 60338005, part 2 60338006 (afterok), headers unchanged.
- **2026-11-21, part 1 got past the queue, then failed on a relative path; fixed and resubmitted.** Part 1 (60338005) ran its checks (103 s), and collect's storage units all started. It then failed after 3.4 minutes, at 03:40, creating the dataset: `FileNotFoundError: Custom module file not found: module_path='pear/agent_loop/dataset.py'`. PEAR's `base.yaml` names `data.custom_cls.path` and `actor_rollout_ref.rollout.agent.agent_loop_config_path` relative to PEAR's root, because PEAR's own `start_ray_node.sh` starts Ray from there. Our launcher does not, and these are the only two such paths in PEAR's configs.
  - Fix (main, a8fe07f): `scripts/common.sh` passes both as absolute paths under `$PEAR`, like `tool_config_path`. `test_every_referenced_repository_file_exists` now covers them. The scripts and configs tests pass (44 of 44).
  - Resubmitted on a8fe07f: part 1 60338213, part 2 60338214 (afterok).
- **2026-11-21, part 1 reached vLLM's launch, then failed on Slurm's GPU variables; fixed and resubmitted.** Part 1 (60338213) got past the dataset. It failed at 03:47 starting the rollout replica's `CheckpointEngineWorker`, with verl's `ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set` (`single_controller/base/worker.py`). Slurm's GPU allocation exports AMD's `ROCR_VISIBLE_DEVICES` beside `CUDA_VISIBLE_DEVICES`, and Ray's workers inherit both.
  - Fix (main, e642cd7): `scripts/ray_node.sh` unsets `ROCR_VISIBLE_DEVICES` when `CUDA_VISIBLE_DEVICES` is set. This is a no-op off Slurm. The static checks pass (14 of 14).
  - Resubmitted on e642cd7: part 1 60338413, part 2 60338414 (afterok).
- **2026-11-21, collect passed; SFT hit the same variable; fix moved and resubmitted.** Part 1 (60338413) passed the checks (115 s) and collect (443 s), writing both episodes. SFT then failed at 04:00 with the same `ROCR_VISIBLE_DEVICES` error: it starts through torchrun, not `ray_node.sh`.
  - Fix: the unset moved to `scripts/common.sh`, which every step sources. It replaces e642cd7 as 6a64ffc. The offline tests pass (64 of 64, network tests skipped).
  - Resubmitted on 6a64ffc: part 1 60338905, part 2 60338906 (afterok).
- **2026-11-21, SFT ran out of GPU memory; offload fixed and resubmitted.** Part 1 (60338905) passed the checks (163 s) and collect (446 s). SFT then started and ran out of memory at 04:24, in Adam's first step (`_init_group`), at 42.3 GB allocated on the 44 GB L40S. A 4B's fp32 parameters, gradients and Adam state are about 64 GB, and `engine.optimizer_offload` moves the state off the GPU only between steps. SFT also ran FSDP1, verl's SFT default, where FSDP2's `offload_policy` does nothing.
  - Main (2055b84): `configs/step/sft.yaml` sets `engine.strategy: fsdp2`, as PEAR trains its actor, with a test. The offline tests pass (64 of 64).
  - Smoke: SFT passes `engine.offload_policy=true` in place of `optimizer_offload`, and part 1's `--mem` goes from 64G to 128G for that host-side state. This is smoke-only, since full runs use A100-80GB or H100 GPUs, several per run.
  - Resubmitted on 2055b84: part 1 60339780, part 2 60339781 (afterok).

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
