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
| unit W_j | one whole linear layer |
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
| not yet fixed | evaluation seeds, per-phase failure attribution |

Environment: conda `vla` (Python 3.11.18, torch 2.2.0+cu121, transformers fork
4.40.1 @ `bc339d9`) built by `setup_env.sbatch` (job 59511042);
`vla-freeze.txt` is under `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/`.
Jobs export `PYTHONPATH=src:third_party/openvla-oft:third_party/LIBERO`.
Submodules: openvla-oft `e4287e9`, LIBERO `8f1084e`.

## Runs

### demo-frames / job 59511837

- Purpose: diagnostic. Phase tagging on the demos and the frame set for the
  gradient step. Also downloads the pinned checkpoint and the RLDS shard set.
- Script: `demo_frames.sbatch` → `demo_frames.py`.
- Header: `--partition=short --cpus-per-task=4 --mem=32G --time=04:00:00`.
- Command: `--rlds-dir .../libero_10_no_noops/1.0.0 --phase-window 10
  --flip-min-hold 5 --frames-per-phase 64 --chunk 8 --seed 7`.
- Output: `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/demo-frames/59511837/`
  (`config.json`, `phases.json`, `frames.npz`).
- Status: submitted.
- Results: pending.

### fp-rollouts / job 59511606 — cancelled while pending, never ran

Superseded when the frame source changed from FP rollouts to demos.
