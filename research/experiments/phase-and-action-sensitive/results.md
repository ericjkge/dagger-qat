# phase-and-action-sensitive — runs and exact setup

Every run is listed with its exact setup, so it can be reproduced from this file
alone. Numbers come from `research/runs/phase-and-action-sensitive.jsonl` and the
per-run `config.json` saved in the run's output directory. Each script commits
before it is submitted, so the `git_commit` in `config.json` identifies the code.

## Fixed first-pass setup (2026-11-13)

Choices marked (human) were made by the human. The rest are first-pass defaults
chosen by the agent under the human's instruction to "use whatever seems the
most reasonable for a first-pass setup".

| item | setting |
|---|---|
| model | OpenVLA-OFT `moojink/openvla-7b-oft-finetuned-libero-10` @ `95220f9a3421a7ff12d4218e73d09ade830fa9a3`, `lora_adapter/*` excluded |
| policy inputs | L1 action head, 2 images (agentview + wrist), proprio, 90% center crop, 8-step action chunk, all 8 executed (open-loop), no FiLM |
| suite | LIBERO-Long (`libero_10`), 10 tasks, env resolution 256, 10 settle steps (no-op, gripper open), 520-step horizon |
| seeding | per episode `set_seed_everywhere(7 + 1000·task + init_state)`, then `env.seed(0)` |
| A | the 8×7 chunk the BF16 policy predicts at a demonstration frame |
| A weighting | all 56 entries, OFT normalized action space (q01/q99 → [−1, 1]), equal weight; projections u ~ N(0, I₅₆) |
| quantizable layers (human) | vision backbone linears (DINOv2 ViT-L + SigLIP SO400M) and Llama linears; projector, embeddings, LM head, proprio projector, action head stay BF16 |
| unit W_j | one whole linear layer; 420 layers (each ViT's last block is unused by OpenVLA and excluded) |
| quantizer (human) | symmetric RTN, group 128 along input dim, s = max\|w\|/(2^(b−1)−1), 2^b−1 levels (2-bit = ternary); simulated (dequantized to BF16) |
| bits (human) | {2, 3, 4, 8}; 8 = real INT8 |
| frame source (human) | true demonstrations: OFT fine-tuning data `openvla/modified_libero_rlds` / `libero_10_no_noops` @ `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`, train split, all episodes; 256 px, model orientation |
| phases | demo gripper action, raw RLDS (−1 open / +1 close); flip counted if held ≥ 5 steps; grasp = ±10 steps around a close flip, place = ±10 around an open flip, otherwise approach (open) / transport (closed) |
| frames for gradients | 64 per phase, split evenly over tasks having that phase, uniform without replacement within task, seed 7; a frame needs 8 following steps in its episode |
| score | s^(p)(W_j, b) = mean over frames in p and k = 8 projections of ⟨∇_{W_j} uᵀA, Q_b(W_j) − W_j⟩² (first-order, keeps cancellation within the layer) |
| baselines (human) | uniform b̄-bit RTN; phase-agnostic allocator (mean over all frames instead of max_p); FP BF16 |
| budget b̄ (human) | 3 bits per weight (changed from 4 the same day, before any run), parameter-weighted average over quantizable layers; group scales not counted |
| worst phase (human) | s(W_j,b) = max_p s^(p)(W_j,b) per layer, no phase normalization; log the argmax phase per layer |
| allocation (human) | greedy: all layers start at 8; repeatedly demote the layer/target bit with the lowest (Δs)/(Δbits·#params), any lower bit allowed, until parameter-weighted mean ≤ b̄; same code for the phase-agnostic arm |
| seeds | "seed" = calibration seed (frame sample + projections u): 7, 8, 9 for both allocator arms. The policy, simulator and init states are deterministic, so uniform and FP are run once |
| evaluation | LIBERO-Long 10 tasks × 50 init states, episode seed 7 + 1000·task + init, env.seed(0); arms run in priority order uniform, worst-s7, phase-s7, worst-s7-head, fp, worst-s8, phase-s8, worst-s9, phase-s9 |
| per-phase attribution | same gripper tagger on executed eval actions; per episode: last phase reached, close/open flip counts, steps per phase |

Environment: conda `vla` (Python 3.11.18, torch 2.2.0+cu121, transformers fork
4.40.1 @ `bc339d9`) built by `setup_env.sbatch` (job 59511042);
`vla-freeze.txt` is under `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/`.
Jobs export `PYTHONPATH=src:third_party/openvla-oft:third_party/LIBERO`.
Submodules: openvla-oft `e4287e9`, LIBERO `8f1084e`.

## Runs

### tests / job 59513531

- Purpose: correctness checks (`tests.py`), then every stage end to end at tiny
  scale in `$OUT/tests/59513531/`. It appends test lines (job 59513531) to the
  runs file; these are smoke tests, not results.
- Header: `--partition=gpu_quad --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=02:00:00`.
- Status: submitted; production jobs held until it passes.

### score / jobs 59513219, 59513220, 59513221 (seeds 7, 8, 9)

- Purpose: per calibration seed (env `SEED`), frames (seed 7 reuses job
  59511837), scores (k = 8), and worst-phase and phase-agnostic greedy
  allocations at b̄ = 3. Replaces 59512608 and 59513187, both cancelled while
  pending.
- Script: `score.sbatch` → `demo_frames.py --frames-per-phase 64 --seed $SEED`,
  `score.py --k 8 --seed $SEED`, `allocate.py --budget 3`.
- Header: `--partition=gpu_quad --gres=gpu:l40s:1 --cpus-per-task=4 --mem=64G --time=06:00:00`.
- Output: `$OUT/scores/s$SEED.npz`, `$OUT/alloc/{worst,phase}-s$SEED.json`, where
  `OUT=/n/scratch/users/e/edk005/vla/phase-and-action-sensitive`.
- Status: submitted.

### recalibration / job 59513471 (after seed-7 score job 59513219)

- Purpose (human: "run 2 and see whether it works", on top of our method):
  quantize the backbone with the worst-phase seed-7 allocation, then fine-tune
  only the L1 action head (float32, AdamW lr 3e-5, batch 32, 3000 steps, L1
  loss) to reproduce the BF16 policy's normalized 8×7 chunks from quantized
  features. Replaces 59513222 (uniform backbone, cancelled while pending:
  the human wanted it on our method).
- Frames: `demo_frames.py --frames-per-phase 1024 --seed 10` (phase-balanced,
  separate draw from the scoring seeds). Split 90/10 train/val by demo episode
  (not by frame), seed 10. The head with the lowest val L1 (checked every 250 steps) is saved in BF16 to
  `$OUT/heads/worst-s7.pt`, with metrics in `worst-s7.json`.
- Script: `recalibrate.sbatch` (`--dependency=afterok:59513219` at submission).
- Header: same as the score jobs.
- Evaluated as arm `worst-s7-head`, paired against `worst-s7`.
- Status: submitted.

### eval / jobs 59513472, 59513473, 59513474 (worker jobs A, B, C)

These replace 59512609, 59513188, 59513223 and 59513224, all cancelled while
pending to change the arm list.

- Purpose: evaluation. Four `eval.py` workers per A100, 12 in total, share one queue of
  (arm, task) units. A unit waits until its allocation file exists.
- Header: `--partition=gpu_quad --gres=gpu:a100:1 --cpus-per-task=16 --mem=120G --time=30:00:00`.
- Command per worker: `--arms uniform worst-s7 phase-s7 worst-s7-head fp worst-s8 phase-s8
  worst-s9 phase-s9 --tasks 0..9 --episodes 50 --seed 7 --uniform-bits 3
  --phase-window 10 --flip-min-hold 5`.
- Output: `$OUT/eval/<arm>.jsonl` (per episode); worker logs `$OUT/logs/eval-<job>-w*.out`.
- Status: submitted.

### demo-frames / job 59511837

- Purpose: diagnostic. Phase tagging on the demos and the frame set for the
  gradient step. Also downloads the pinned checkpoint and the RLDS shard set.
- Script: `demo_frames.sbatch` → `demo_frames.py`.
- Header: `--partition=short --cpus-per-task=4 --mem=32G --time=04:00:00`.
- Command: `--rlds-dir .../libero_10_no_noops/1.0.0 --phase-window 10
  --flip-min-hold 5 --frames-per-phase 64 --chunk 8 --seed 7`.
- Output: `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/demo-frames/59511837/`
  (`config.json`, `phases.json`, `frames.npz`).
- Status: COMPLETED in 5 min. 379 demos, 10 tasks. Steps per phase:
  approach 42278, grasp 14191, transport 32603, place 12397. 64 frames per phase.
- Notes: the knob turn (task 3) and drawer/microwave handles (tasks 2, 8)
  register as extra grasps. In task 9, 40 of 41 demos never release the book,
  so its place frames come from one demo.

### fp-rollouts / job 59511606 — cancelled while pending, never ran

Superseded when the frame source changed from FP rollouts to demos.
