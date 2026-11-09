# Changelog

## Unreleased
- Repo skeleton.

## 2026-11-08 — Sophie / Codex: OpenVLA-OFT baseline setup

- Read current `main` (`4d1c289`) and the previous `PROPOSAL.md` from `a035129`.
  The overhaul removed that proposal and the previous π0.5 runtime; this session
  follows the new immediate scope: OpenVLA-OFT on LIBERO-Long, unquantized vs. AWQ.
  GPTQ, on-policy refresh and a new loop-aware quantizer remain later work.
- Implemented the OFT loader, paired LIBERO evaluation, per-episode JSONL and
  comparison tooling. Preserve the official preprocessing and action semantics;
  propagate infrastructure failures instead of counting them as policy failures.
- Implemented simulated Llama W4/g128 AWQ with activation scale search and clipping,
  adapted from MIT HAN Lab `llm-awq` at `d6e797a`. Sensitive non-decoder components
  remain unquantized. Calibration is 32 task-balanced original training-demo frames.
- Added a pinned Linux/CUDA runtime, asset download/provenance, Cannon setup and
  evaluation jobs. Checkpoint revision: `95220f9a3421a7ff12d4218e73d09ade830fa9a3`;
  dataset revision: `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- Local validation: CPU numerical/integration tests use the pinned Transformers
  fork with the installed macOS Torch 2.11; GPU runtime uses Torch 2.2 as upstream.
  Actual benchmark scores must come from Cannon, not these tests.
- Cannon: authenticated through the user's terminal; new checkout
  `/n/home13/sophiezeng/dagger-qat`, scratch data at
  `/n/netscratch/amin_lab/Everyone/sophiezeng/dagger-qat`.
  Initial CPU setup job `59049206` caught a Protobuf/TF Metadata version conflict;
  pinned Protobuf 3.20.3 for the TF 2.15/TFDS 4.9.3 stack. Setup retry: `59051313`.
  Dependent one-episode FP smoke job: `59051490`.
  Serialize upstream's checkpoint-file updates with a lock so concurrent model
  loads cannot race on the shared checkpoint directory.
  Match BF16 scaled-weight rounding during scale search; test both FP32 and BF16
  search against the RTN candidate. Local validation: 18 tests passed, Ruff and
  shell syntax/diff checks passed.
  Setup job `59051313` completed successfully in 39m49s, including real model and
  simulator imports. Parallel asset job `59055904` completed in 4m04s.
  CPU tests on the exact runtime: job `59057158`, 18 passed in 83.45s.
  GPU smoke job `59051490` completed successfully on an A100-SXM4-80GB in 7m12s:
  task 0 / initial state 0 succeeded in 256 actions (32 policy calls, 35.28s).
  Saved JSONL, provenance and a rollout video under scratch
  `results/20261109T033843Z-59051490-fp/`. This single episode is a pipeline check,
  not a benchmark score.
  Full paired 500-episode jobs: FP `59063995`, AWQ `59063997`. Separated AWQ
  calibration plus one rollout into job `59064488` (two-hour limit); the full
  AWQ run depends on its successful completion and reuses the saved artifact.
  FP has an eight-hour reservation; AWQ has five hours after measuring rollout
  throughput. FP and AWQ validation started at 22:54:39 EST on separate
  A100-SXM4-80GB GPUs on `holygpu8a27601` (`gpu_requeue`). Full AWQ started at
  23:06:40 EST on the same node/GPU type. Restoring the artifact reproduced the
  calibration smoke's 384-action success for initial state 0.
  AWQ validation `59064488` completed 0:0 in 9m28s. All 32 decoder layers were
  calibrated; the smoke episode succeeded in 384 actions / 48 calls (16.81s).
  Records/video: `results/20261109T035441Z-59064488-awq/`; calibration artifact:
  `calibration/awq-w4g128-32frames-seed7.pt` (164MiB), SHA256
  `af08ac980fc3276d820baa93ad054509b51c831469ef4fee9a71a4388a10e7e6`.
  Full-run output directories: `results/20261109T035441Z-59063995-fp/` and
  `results/20261109T040641Z-59063997-awq/`.
  Full benchmark completion and scores are recorded in the 2026-11-09 entry below.
  For future preemptions, retain partial runs and rerun to completion before
  reporting a score.
  The comparison also rejects mismatched simulator seeds or image resolutions.
- Incorporated the team's additional on-policy/OOD notes and read the shared
  discussion. Added `docs/research-notes.md` with primary-source reading notes
  for DAgger, VLAQuantBench, QAIL, SQIL and recent LoopQ; a data/objective ablation
  plan; and qualifications on DAgger guarantees, GPTQ curvature and QAT/QLoRA.
  These are proposed next experiments, not completed GPTQ, DAgger, QAT or OOD runs.

## 2026-11-09 — Sophie / Codex: completed paired LIBERO-Long baselines

- Both full Cannon jobs completed 0:0 without restarts: BF16 in about 1h32m;
  AWQ in about 1h28m. Both used A100-SXM4-80GB GPUs, clean source
  `ef1fa1608b8299b44307f1ae6c1ca2167decd246`, and 500 episodes (10 tasks × 50 states).
- **BF16: 467/500 (93.4%). AWQ W4/g128: 467/500 (93.4%).** AWQ minus FP is
  0.0 percentage points; descriptive paired bootstrap 95% interval [-2.2, +2.2]
  pp. There are 452 shared successes, 15 successes unique to each method, and
  18 shared failures. Equal aggregate scores do not establish equivalence.
- Independently checked all 1,000 episode identities, seeds, task names, success
  flags, action/call counts, summaries and provenance. Zero infrastructure errors.
  Rechecked the cached AWQ artifact SHA256 against the run metadata.
- Published the aggregate comparison and per-task report under
  `results/2026-11-09-openvla-oft-libero-long/`. Complete episode records,
  calibration selections, environment inventory and scheduler provenance are
  retained locally and on Cannon. The public result files contain aggregate data.
- Clarified the seeding difference from upstream: this evaluator reseeds before
  every episode; `env.seed(0)` also resets NumPy globally, while Python/Torch retain
  `7 + 1000 * task_id + initial_state_id`. The two arms share that protocol. These
  scores are not an exact reproduction of upstream's three-seed paper result.
- Interpretation: nominal task accuracy is tied under this protected decoder-only
  W4 scope. This does not measure packed INT4 efficiency or OOD robustness. Next
  steps remain matched RTN/GPTQ, a controlled on-policy data refresh, then a new
  inner method if the earlier gates support it; see `docs/research-notes.md`.
- Contribution is in PR #8 (`codex/libero-long-baselines`); the immediate paired
  baseline task is complete. Monday slides and DAgger/QAT implementation remain
  separate TODOs.

## 2026-11-09 — Sophie / Codex: matched GPTQ baseline preparation

- Added a GPTQ arm with the same OpenVLA-OFT checkpoint, 32 task-balanced training
  observations, seven decoder projections per layer, W4/g128 asymmetric format,
  and 500-episode paired LIBERO-Long protocol as the completed FP/AWQ experiment.
- Adapted original GPTQ revision `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` with
  natural column order, 128-column blocks, 1% Hessian damping and true-sequential
  calibration. Subsequent decoder groups/blocks use quantized activations; no
  environment rollout data is used for calibration yet.
- Added packed-code calibration artifacts that reconstruct BF16 weights, explicit
  cache provenance, the GPTQ launcher option and paired comparison support.
  This remains a simulated-quantization accuracy experiment.
- Numerical port matched the original implementation exactly on synthetic FP32
  and BF16 checks. Local suite: 28 tests passed. Fixtures preserve original GPTQ
  outputs; integration tests cover multimodal capture, sequential propagation,
  protected modules and artifact replay. Allow small FP32 tolerance across BLAS
  implementations while requiring exact BF16 fixture agreement.
- Cannon authentication renewed. GPU calibration/smoke validation and the full
  benchmark are the next steps; no GPTQ success rate is available yet. Publish
  only aggregate results and public protocol details; keep raw records and
  internal cluster metadata in the local run archive.
- Follow-up log audit identified one shared MuJoCo acceleration-instability case
  in each previous baseline, counted as a failure without a Python exception.
  Clarified this in the baseline report: raw scores remain 467/500 for both;
  excluding the flagged pair gives 467/499 (93.6%) for both. Keep warning-based
  sensitivity analysis separate from the original raw scores. Added the generated
  MuJoCo log to `.gitignore` without deleting it.
