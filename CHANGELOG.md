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
  flags, action/call counts, summaries and provenance. Zero recorded Python
  exceptions; the later simulator-warning audit below qualifies this result.
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

## 2026-11-09 — Sophie / Codex: matched GPTQ baseline launched

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
- Cannon's exact-runtime suite passed all 28 tests in 61.82 seconds. All 32
  decoder layers calibrated on an A100-SXM4-80GB, and the one-episode smoke
  rollout succeeded in 256 actions / 32 policy calls. The calibration/smoke job
  completed 0:0 in 9m20s, without restarts or simulator-instability warnings.
  This is a pipeline check, not a GPTQ benchmark score.
- Verified that the 32 calibration observations, checkpoint and dataset
  revisions, normalization statistics, protected scope and evaluation protocol
  match the earlier baselines. Run source is clean at
  `a90c67c9eac5ddf9ba44a9c03ac2b4240c39cd93`. Saved GPTQ artifact SHA256:
  `d97fba21c6b297b248b6997477e811dafc5cc82dab07cd5dd7030db818ecef42`.
- The full 500-episode GPTQ evaluation has started on the same GPU type using
  the validated cache. Cache reload reproduced the smoke episode's success and
  256-action / 32-call outcome; the first eight episodes completed without
  recorded errors or simulator-instability warnings. Its score is pending.
  PR #9 is a draft stacked on #8;
  after completion, validate all episode pairs, inspect simulator warnings,
  and compare with both completed baselines before reporting results. Job
  identifiers and output paths are in the ignored local `data/gptq-submission.json`.
  Publish only aggregate results and public protocol details; keep raw records
  and internal cluster metadata in the local run archive.
- Follow-up log audit identified one shared MuJoCo acceleration-instability case
  in each previous baseline, counted as a failure without a Python exception.
  Clarified this in the baseline report: raw scores remain 467/500 for both;
  excluding the flagged pair gives 467/499 (93.6%) for both. Keep warning-based
  sensitivity analysis separate from the original raw scores. Added the generated
  MuJoCo log to `.gitignore` without deleting it.
- Follow-up status check at 10:47 EST: the GPTQ full run is still running, with
  63/500 episodes completed and no recorded episode errors or simulator-instability
  warnings so far. No final summary exists yet; a complete three-way score
  comparison is pending. The completed BF16 and AWQ scores remain 467/500 each.
- Follow-up at 11:45 EST: 412/500 episodes completed after 1h13m; the run is still
  active and has no final summary. Zero recorded Python errors. Its one MuJoCo
  instability warning matches the same paired episode flagged in both earlier
  baselines; all three count that episode as unsuccessful. Preserve the planned
  raw-score and shared-warning-exclusion comparisons when GPTQ finishes.

## 2026-11-09 — Sophie / Codex: completed GPTQ and three-method comparison

- GPTQ completed at 12:02 EST, exit 0:0, in 1h29m37s with zero restarts and zero
  recorded Python exceptions. **GPTQ: 465/500 (93.0%)**, versus **467/500 (93.4%)**
  for both unquantized BF16 and AWQ W4/g128.
- GPTQ minus either baseline is **−0.4 percentage points**, with a descriptive
  paired-bootstrap 95% interval of **[−2.6, +1.8] pp**. In each pairwise comparison
  GPTQ wins 15 episodes and loses 17; 450 succeed in both and 18 fail in both.
  Across all three methods, 439 episodes succeed in every arm and 14 fail in
  every arm. Close totals do not establish equivalence or a clear regression.
- Validated all 1,500 records for expected identities, task names, seeds,
  action/call counts, status and summary agreement. Confirmed matching checkpoint,
  runtime, GPU type, protocol, calibration observations and normalization;
  independently rechecked the saved GPTQ cache SHA256. The original BF16/AWQ
  aggregate comparison reproduces exactly.
- The final GPTQ log contains one simulator-instability warning in the same
  paired episode flagged in both prior baselines. All three record a failure.
  Excluding that shared pair gives BF16/AWQ **467/499 (93.6%)** and GPTQ
  **465/499 (93.2%)**; the difference remains about −0.4 pp. Keep this sensitivity
  result separate from the primary 500-episode scores.
- Added `results/2026-11-09-openvla-oft-gptq/` with aggregate pairwise comparisons,
  per-task counts, uncertainty, simulator sensitivity and public provenance.
  Raw records, logs and scheduler evidence remain in ignored local data and on
  Cannon. PR #9 now includes the completed results, stacked on the still-open #8.
- This completes fixed-demonstration BF16/AWQ/GPTQ baseline evaluation. No
  on-policy refresh, DAgger/QAT update, new quantizer, OOD result or packed INT4
  efficiency claim follows from these measurements. The next proposed experiment
  holds GPTQ and the calibration budget fixed while changing calibration data;
  calibration rollouts must remain separate from evaluation episodes.

## 2026-11-09 — Sophie / Codex: first controlled GPTQ calibration loop

- Implemented the pre-set two-round experiment in `docs/onpolicy-gptq-protocol.md`:
  learner-controlled collection versus a full-precision-rollout control, each
  retaining 16 original demo anchors and 16 rollout observations per calibration.
  Round 2 aggregates eight observations from each round. The unchanged GPTQ
  method is rebuilt from the original BF16 checkpoint before each update.
- Collection uses two fresh seeded resets per task per round, checks physical
  initial configurations against all evaluation states and prior collection
  states, and samples at eight-action boundaries with independent reservoirs.
  Valid unsuccessful trajectories are retained. Nonfinite values or simulator
  warnings abort collection. Hashes bind the stored arrays, actor and round lineage.
- Two final-round evaluations of 500 episodes each will compare with the
  completed offline 93.0% GPTQ baseline and with each other. Hyperparameters and
  the final round are fixed before these evaluations; pipeline-check episodes
  do not choose updates. No teacher action labels enter this data-only update,
  and no DAgger imitation-learning guarantee is claimed.
- Added collection/bundle utilities, calibration-bundle support, Cannon stage
  launchers and a matched final comparison. Original GPTQ numerics and rollout
  evaluation remain unchanged. Local validation: 37 tests passed, including
  exclusion of evaluation states, chunk-boundary collection, warning rejection,
  replay balance/lineage, corruption detection and paired comparison. Ruff,
  formatting, shell syntax and diff checks passed. Cannon validation is next.
- Answered the inner-method question in `docs/research-notes.md`: prioritize AWQ
  under the same outer loop, an RTN control, OmniQuant, then a carefully compared
  action-loss update of quantizer parameters. SmoothQuant changes the activation
  format; QLoRA is an adapter-training baseline. No extra quantizer runs launched.

## 2026-11-09 — Sophie / Codex: completed controlled nominal calibration loop

- Both pre-set round-2 evaluations completed: learner-calibrated GPTQ **464/500
  (92.8%)**, BF16-rollout-calibrated GPTQ control **474/500 (94.8%)**, versus offline
  GPTQ **465/500 (93.0%)**. Learner minus offline is −0.2 pp (descriptive paired
  bootstrap 95% interval [−2.4, +2.0]); learner minus control is −2.0 pp
  ([−4.0, −0.2]). This first experiment does not demonstrate an on-policy gain.
- All four collection/calibration stages completed and all 37 tests passed on
  the actual runtime. Audited all 1,000 new evaluation records and reproduced the
  strict comparator. The same single failed simulator-instability episode occurs
  in all five methods; excluding that shared pair leaves the conclusion unchanged.
- Published aggregate results and caveats under `results/2026-11-09-onpolicy-gptq/`.
  Raw records, calibration selections and cluster identifiers remain private.
- User requested LIBERO-Plus to test robustness beyond the nearly saturated
  nominal benchmark. Preparing a fixed, category/task-balanced 70-case Long
  pilot across all five existing artifacts, with parallel GPU shards. It measures
  transfer of nominal calibration; no Plus data will be used to fit these models.

## 2026-11-09 — Sophie / Codex: LIBERO-Plus pilot prepared

- Added a fixed 70-case Plus-Long manifest: one case per original task/category,
  with all seven perturbation categories, author difficulty levels and all five
  noise operators represented. The five nominal artifacts are reused unchanged.
- Added isolated Plus setup, verified pinned assets, paired per-case seeding,
  simulator-warning capture and 20 deterministic GPU shards (up to eight at once).
  Private mutable checkpoint metadata permits concurrent model loading while
  shared model tensors remain unchanged. Actual BDDL instructions avoid unintended
  perturbation suffixes in language input; this is a documented paired pilot,
  not an exact reproduction of the official leaderboard protocol.
- Local validation: 40 tests passed; Ruff, formatting, shell syntax and diff
  checks passed. GPU smoke and pilot scores are still pending. The nominal result
  report is complete in PR #10; this Plus extension is a separate contribution.

## 2026-11-09 — Sophie / Codex: Plus setup complete; deadline pilot queued

- Completed pinned asset verification and CPU construction of all 70 fixed Plus
  scenes. Selective extraction materialized 3,674 required files without changing
  their bytes. Setup-only extraction hooks are absent during scored evaluation.
- Added a predeclared 14-case subset before observing any Plus outcomes: two cases
  per category, all ten original tasks, unchanged 520-action horizons and paired
  seeds. This urgent subset is separate from the planned 70-case pilot. It cannot
  establish small success-rate differences: one episode is 7.1 percentage points.
- Submitted parallel evaluations of BF16, AWQ, offline GPTQ, learner-calibrated
  GPTQ and the BF16-rollout calibration control. Prioritized the three quantized
  methods requested for the presentation. The urgent subset permits full Ampere
  GPU models and records hardware per arm; the larger pilot retains A100 hardware.
- At the final pre-deadline check, scored jobs were still awaiting scheduler
  allocation and no Plus success rates were available. Queue delay is not a model
  failure. Do not substitute nominal scores, published scores or illustrative
  targets for measured quantized Plus results. Private submission manifests retain
  scheduler details and the exact source commit; comparisons run after completion.
- Validation: the expanded suite passed 49 local tests; the latest hardware
  reporting adjustment passed all 12 Plus tests. An earlier 46-test version also
  passed on Cannon. Nominal results remain BF16/AWQ 93.4%, offline GPTQ 93.0%,
  learner GPTQ 92.8%, and BF16-rollout-calibrated GPTQ 94.8% (500 episodes each).

## 2026-11-09 — Sophie / Codex: validated the test-GPU route for Plus

- Full-GPU allocations remained queued. Prepared an isolated, signature- and
  hash-verified Mesa/OSMesa renderer to use A100 MIG test GPUs without changing
  the existing Python packages or frozen quantization artifacts.
- AWQ loaded on a 20 GB A100 MIG and completed all seven short category checks,
  including glass blur, without Python errors or simulator warnings. These are
  eight-action infrastructure checks, not scored full-horizon benchmark episodes.
- Added a separately identified 14-case OSMesa pilot with the same frozen cases,
  seeds and 520-action limits. All five arms require matching MIG hardware and
  renderer settings; the comparator rejects pooling with EGL results. Two jobs
  run concurrently, with AWQ and GPTQ before the reference arms.
- All 50 local tests pass, including rejection of mixed rendering protocols and
  hardware. Ruff, shell syntax and diff checks pass. Full-horizon scores remain
  pending; submission identifiers and raw logs stay in ignored private data.
- The test partition also caps submitted jobs at two, so a five-task array was
  rejected before execution. Two worker jobs now schedule the five methods
  sequentially within their allocations, preserving separate per-method records.
- At 16:55 EST, the first scored worker had started AWQ; the offline-GPTQ worker
  was still queued. No full-horizon Plus episode scores were available yet. The
  running checkout is frozen at `f76cc2c9cfb1f5fac3336767573db19ef7d86928`;
  subsequent documentation-only commits are not deployed into active evaluations.
