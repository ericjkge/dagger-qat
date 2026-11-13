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
| A (human) | rolled-out action: the 8×7 chunk the FP policy outputs at states of its own closed-loop rollouts |
| A weighting | all 56 entries, OFT normalized action space (q01/q99 → [−1, 1]), equal weight; projections u ~ N(0, I₅₆) |
| quantizable layers (human) | vision backbone linears (DINOv2 ViT-L + SigLIP SO400M) and Llama linears; projector, embeddings, LM head, proprio projector, action head stay BF16 |
| unit W_j | one whole linear layer |
| quantizer (human) | symmetric RTN, group 128 along input dim, s = max\|w\|/(2^(b−1)−1), 2^b−1 levels (2-bit = ternary); simulated (dequantized to BF16) |
| bits (human) | {2, 3, 4, 8}; 8 = real INT8 |
| frame source | FP rollouts on init states 0–4 of every task; these states are held out of evaluation (eval uses 5–49) |
| frames used | successful rollouts only |
| phases | executed gripper command (−1 open / +1 close); flip counted if held ≥ 5 steps; grasp = ±10 steps around a close flip, place = ±10 around an open flip, otherwise approach (open) / transport (closed) |
| frames for gradients | policy-call steps only (one per 8 env steps); 64 per phase, sampled uniformly across tasks |
| score | s^(p)(W_j, b) = mean over frames in p and k = 8 projections of ⟨∇_{W_j} uᵀA, Q_b(W_j) − W_j⟩² (first-order, keeps cancellation within the layer) |
| baselines (human) | uniform b̄-bit RTN; phase-agnostic allocator (mean over all frames instead of max_p); FP BF16 |
| not yet run | allocation (exact knapsack vs greedy), phase normalization, b̄ values, evaluation |

Environment: conda `vla` (Python 3.11.18, torch 2.2.0+cu121, transformers fork
4.40.1 @ `bc339d9`) built by `setup_env.sbatch` (job 59511042);
`vla-freeze.txt` is under `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/`.
Jobs export `PYTHONPATH=src:third_party/openvla-oft:third_party/LIBERO`.
Submodules: openvla-oft `e4287e9`, LIBERO `8f1084e`.

## Runs

### fp-rollouts / job 59511606

- Purpose: diagnostic. FP rollouts on the held-out frame states, phase tagging,
  and saved policy-call observations for the gradient step.
- Script: `fp_rollouts.sbatch` → `fp_rollouts.py`.
- Header: `--partition=gpu_quad --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=06:00:00`.
- Command: `--suite libero_10 --tasks 0..9 --init-states 0 1 2 3 4 --seed 7
  --max-steps 520 --settle-steps 10 --open-loop-steps 8 --phase-window 10
  --flip-min-hold 5`.
- Output: `/n/scratch/users/e/edk005/vla/phase-and-action-sensitive/fp-rollouts/59511606/`
  (`config.json`, `phases.json`, `episodes/*.npz`).
- Status: submitted.
- Results: pending.
