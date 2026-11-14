---
slug: phase-and-action-sensitive
status: dead
hypothesis: At matched average bits b̄, worst-phase action-sensitive bit allocation over {2,3,4,8} gives higher LIBERO-Long success than the same allocation using the phase-agnostic mean, with the gain concentrated in grasp/place.
kill_criterion: At b̄ = 3, if mean LIBERO-Long success over ≥3 seeds does not exceed both uniform b̄-bit RTN and the phase-agnostic-mean allocator by more than 2 SE of the paired per-episode difference, the direction is dead.
best_metric: worst-phase 3-bit LIBERO-Long success 47.7% mean over 3 seeds (uniform 3-bit 73.2%, BF16 95.0%)
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

**Dead (2026-11-13 22:25): the kill criterion is met.** At b̄ = 3, worst-phase
allocation is far below uniform 3-bit RTN on every seed. Its edge over the
phase-agnostic allocator is not consistent across seeds. All runs, configs and
job IDs are in `results.md`. The exact setup is in its first table.

How the run went:
- Setup choices: some made by the human (A is the policy's action chunk;
  frames are true demos; quantize the vision backbone and Llama linears;
  symmetric RTN; INT8 as 8 bits; per-layer worst phase; greedy; b̄ = 3;
  baselines uniform, phase-agnostic and BF16), the rest agent defaults
  recorded in `results.md`.
- `research/bin/submit` on this branch: no per-job time limit, RAM cap 128G,
  no job cap (human instructions).
- Tests (`tests.py`, job 59537943) passed before any production run.
- Recalibration (action-head fine-tuning) was skipped by the human; its code
  is kept, unrun.
- GPUs were the bottleneck all day. The human paused their learned-hierarchy
  runs at 17:45 to free GPUs. That direction's note later records the human's
  decision not to resume them, so the agent's resubmissions (59587922/23) were
  cancelled before they started.
- One eval record (worst-s9, task 0, init state 29) was lost by concurrent appends
  from workers on different nodes to a shared jsonl on scratch. It was
  restored from that worker's log. Every arm was checked against the logs; no
  other record was missing.

## Results

LIBERO-Long, OpenVLA-OFT, 500 paired episodes per arm (10 tasks × 50 init
states, deterministic policy and simulator). Calibration seeds 7, 8, 9.

| arm | success |
|---|---|
| BF16 | 95.0% |
| uniform 3-bit | 73.2% |
| worst-phase s7 / s8 / s9 | 55.8% / 46.0% / 41.4% (mean 47.7%) |
| phase-agnostic s7 / s8 / s9 | 50.0% / 30.0% / 51.4% (mean 43.8%) |

| paired difference | s7 | s8 | s9 | pooled (1500 ep.) | seed-level (n = 3) |
|---|---|---|---|---|---|
| worst − uniform | −17.4 ± 2.3 | −27.2 ± 2.4 | −31.8 ± 2.5 | −25.5 ± 1.4 | −25.5 ± 4.2 |
| worst − phase-agnostic | +5.8 ± 2.2 | +16.0 ± 2.3 | −10.0 ± 2.3 | +3.9 ± 1.3 | +3.9 ± 7.6 |

Points, ± SE. Kill test: worst − uniform is far below zero, so the direction
is dead. The phase max beats the mean in the pooled episode-level test
(3 SE), but not at the seed level, and its sign flips on seed 9. That is not
a result.

Failures by last phase reached are mostly approach for every quantized arm
(uniform 79/134, worst 109–133 per seed), not grasp or place.

## Next step

None for this direction. Possible follow-ups for the human to decide: score
damage by actually measuring ‖ΔA‖ per layer at each bit-width (first-order
reads 2-bit damage 1–2× low); drop ternary 2-bit from the bit set; or use
GPTQ-style error compensation before allocating.

## Blocked on

Recording the retirement in the `research/README.md` table is for the human;
this agent may not edit that file.

## Dead ends

- Worst-phase greedy layer allocation at b̄ = 3 over {2,3,4,8}: 47.7% mean,
  25.5 points below uniform 3-bit. Pushing about 60–80 layers to ternary
  2-bit costs more than the 8-bit layers it pays for.
- Phase max vs mean: +5.8, +16.0, −10.0 points on 3 calibration seeds.
  Calibration noise (64 frames per phase, k = 8) changes 113–155 of 420
  layers' bits between seeds, more than max vs mean does (43–53).
