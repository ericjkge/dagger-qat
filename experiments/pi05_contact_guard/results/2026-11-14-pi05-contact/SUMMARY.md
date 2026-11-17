# π0.5 contact-guard activation quantization: summary (2026-11-14) — **NO-GO**

**Hypothesis:** in π0.5 on LIBERO, 4-bit activation quantization errors near contact transitions change whether contact
happens, even when numerical action error is small, and contact-guard-aware rounding can prevent this.

**Verdict: NO-GO**, for two reasons:
- **The hypothesis is not supported.** Contact damage under 4-bit activations (A4) is no larger than random error of
  the same size, and the guard-aware rounding oracle gains about 0.1 mm.
- **The real failure mode is already solved by published methods.** It is a speed-shrinking bias from `down_proj`
  outliers, and Hadamard rotation (HoloQ-VLA, FoldQuantVLA) fixes it. A follow-up idea, sigma-delta / error-feedback
  quantization across denoising and control steps, also lacks novelty: Sigma-Delta Quantized Networks (2017), MoDiff,
  and a 2026 noise-shaping quantizer for neural-network controllers already cover it.

The full report with tables is [REPORT.md](REPORT.md). Raw logs and JSONL are in `../2026-11-14-pi05-contact-raw/`.

## Setup
- **Model:** openpi PyTorch `pi05_libero` (BF16).
- **Quantization:** fake-quantized activations only (weights stay BF16) on the inputs of all 7 linears in the 18
  action-expert layers, across all 10 flow steps. The grid is per-token asymmetric min-max: 4-bit (A4) or 8-bit (A8).
- **Tasks:** LIBERO bowl (Spatial), soup (Object) and cheese (Goal), 20 paired episodes each.
- **Pairing:** every comparison uses the same snapshot, inputs and flow noise.
- **Hardware:** 1× A100 80GB (Runpod), about 2.7 GPU-hours, about $4.10 total.

## Runs and results

| # | Run | n | Result |
|---|---|---|---|
| 1 | One-step counterfactuals (A4, A8, `down` only, without `down`) plus a random perturbation of the same norm | 1,397 states | Contact flips occur only in transition states. A4 = 0.176 vs. random 0.162. In a logistic regression, the quantization-vs-random coefficient is ≈ 0 |
| 2 | 10-step re-simulation: missed vs. delayed contacts | 464 states | A4 misses as often as random (7.4% vs. 5.9%) but is biased toward *delay* (27% vs. 16%). Changes are mostly ±1-step shifts |
| 3 | Closed loop, quantized policy driving | 60 episodes per arm | Success: BF16 59/60, A4 55/60 (4 soup timeouts with the object held). Median grasp delay with A4: +6.5 to +21 steps |
| 4 | Step 3: linearized predictor Δg ≈ M(ĥ − h) vs. simulated Δg | 313 states | Single quantized site: accurate (cos 0.98, sign 0.88). Full A4 at transitions: Spearman 0.42, underestimates Δa by about 40%. Most crossings are within 2 mm of the boundary, where a linear predictor can't decide |
| 5 | Step 4 oracle: nearest vs. guard-aware vs. action-aware rounding on the same grid, top-12 coordinates | 75 states | Isolated site: guard-aware rounding improves 88% of states, but the gain is only about 0.1 mm. Inside full A4: no effect (47% better / 52% worse) |
| 6 | Diagnosis: 4-bit grid statistics per input group | 12 states | `down_proj` inputs: kurtosis 130–264 and 89–92% of entries rounded to 0, so the layer output shrinks to 0.83–0.87×. Other groups: 0.95–0.99× |
| 7 | One-step speed check (incl. Hadamard-rotated A4) | 90 states | Speed ratio: A4 0.77, A4 without `down` 1.01, A4 + rotation 1.00. Rotation cuts action MSE 380× |
| 8 | Closed loop with fixes | 60 episodes per arm | A4 without `down`, A4 + rotation on `down`, and A4 + rotation on all groups: each 58/60, with BF16 timing (grasp delay 0 to ±1 step) |
| 9 | Overhead of the fake-quant simulation (eager PyTorch) | 20 calls | BF16 310 ms, A4 546 ms, A4 + rotation 1,579 ms per call; peak memory 7.64 GB for all. These are simulation costs, not int4 kernels |

## Why NO-GO
1. **Contact is not the weak point.** Near contact, quantization error behaves like random error of the same size, so
   rounding that targets contact has nothing specific to fix. The oracle confirms this: about 0.1 mm of gain, which
   disappears in full A4.
2. **The failure that does exist is solved.** It is a systematic shrinkage of action magnitude from the dead zone on
   `down_proj` outliers. Hadamard rotation (QuaRot / HoloQ-VLA / FoldQuantVLA) or keeping `down_proj` at higher
   precision removes it, and BF16-level success leaves no headroom on these tasks.
3. **Follow-up idea: also not novel.** Error-feedback (sigma-delta) quantization across denoising and control steps
   amounts to combining existing techniques.

**If we continue:** first find where rotation-based A4 still fails in closed loop, for example A3, W4A4 or
LIBERO-Long. Then design a method around that failure.

## Limitations
- Analysis-only use of privileged simulator information: guards, phases, ∂g/∂a and the oracle.
- Fake quantization (no integer arithmetic).
- Jacobians computed in BF16, not FP32.
- Min-distance guard discontinuities.
- n = 20 per task, so the success differences are underpowered; the paired timing shifts are the robust signal.
- Novelty check was a quick search, not an exhaustive review.

## Reproduce
Code is in `src/dqat/pi05/` and `scripts/pi05/`; tests are in `tests/test_pi05_actquant.py`. Commands are in
[REPORT.md §9](REPORT.md#9-reproduction). The replay snapshots (`snaps_*.npz`) are kept only on the Runpod volume of
pod `htq79y1t9mrswe`.
