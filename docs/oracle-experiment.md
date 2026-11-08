# Oracle kill-check: is calibration data a lever at all?

**Question.** Under a distribution shift, can *any* choice of calibration data shrink the gap
between the quantized and FP π0.5? If calibrating on the test states themselves does not help,
no data-selection method can, and we should say so and change direction.

**Answer either way is useful.** If it helps, we have a cheap, training-free robustness fix.
If it doesn't, we have a clean negative result that data selection is the wrong lever.

## Design

Shift: LIBERO-Plus **camera**, the largest π0.5 quantization gap reported so far (PIVOT-Q:
FP 82.1 → W4A8 57.1). Camera episodes are split into two disjoint halves, A and B (see
`docs/teammate-tasks.md`). B is the test set.

Every arm uses the same quantizer (SmoothQuant-style W4A8 with static per-tensor activation
scales, QuantVLA's 180 layers), the same number of calibration frames (≤256, 4 per episode,
seed 0), and is evaluated on the same frames with the same flow-matching noise. **Only the
calibration data changes.**

| Arm | Calibration data | Role |
|---|---|---|
| `q_clean` | FP rollouts on clean LIBERO | standard practice (reference) |
| `o1_camera_other` | FP rollouts on camera split A | same shift type, different instances |
| `o2_test_states` | Q-clean rollouts on camera split **B** (the test states) | oracle upper bound for data selection |
| `q_dynamic` | clean (smoothing only); activation scales per token at run time | no calibrated activation ranges |

Decomposition arms (where does the gap come from?):

| Arm | What it isolates |
|---|---|
| `w4_rtn_only` | 4-bit weights only, no calibration, no smoothing |
| `a8_static_clean` / `a8_static_o2` | 8-bit static activations only, clean vs oracle data |
| `gptq_clean` / `gptq_o2` | GPTQ 4-bit weights (Hessian from calibration data), clean vs oracle |

Paired arms (`*_o2`) are compared to their own clean twin, not to `q_clean`.

## Two stages

1. **Offline (cheap, hours).** `scripts/offline_gap.py --plan configs/killcheck/camera.json`.
   Metric: per-frame mean |a_quant − a_FP| over the 6 arm dimensions and the 10-step horizon,
   plus the gripper sign-flip rate. Each arm gets a paired bootstrap 95% CI vs its baseline.
   `recovered` = 1 − L1(arm) / L1(baseline).
2. **Closed loop (expensive, only if stage 1 says so).** Task success on camera split B for
   FP, `q_clean`, `o1`, `o2`, `q_dynamic`, 200–300 paired episodes each (same init states and
   seeds across arms).

## Decision rules (set before seeing results)

First, the gap must exist: `q_clean` must have clearly higher L1 on `camera_b` than on
`clean_heldout`, and a closed-loop success gap vs FP on split B whose CI excludes 0. If W4A8 has
no gap, rerun with `configs/quant/sq_w4a4_static.json` (OnPTQ found calibration effects mainly at
lower activation bit widths).

| Result | Interpretation | Next step |
|---|---|---|
| `o2` recovers ≈0 (CI includes 0, or <10%) | Data selection is the wrong lever for this quantizer. | Report as a negative result. Use the decomposition arms to say why (e.g. the gap is all in the weights). |
| `o2` recovers a lot, `o1` also does | Shift-type data is enough. No test-time access is needed. | Closed loop. Then held-out perturbation types (does it transfer?). |
| `o2` recovers, `o1` does not | Only on-policy / test-specific states help. | On-policy calibration is the method. Closed loop. |
| `q_dynamic` recovers as much as `o2` | Static activation ranges are the whole story, and a known fix removes it. | Novelty must come from elsewhere: weights/GPTQ arms, or consequence weighting. |
| `w4_rtn_only` ≈ `q_clean` | Gap comes from weights, not activation ranges. | The only data lever is the GPTQ Hessian: read `gptq_o2` vs `gptq_clean`. |

The ~10% threshold is a judgment call. Report the full CIs, not just the verdict.

## Caveats

- `o2` calibrates on the very frames it is evaluated on. This is intentional: it is an upper
  bound, not a method.
- Offline L1 is a proxy. A small action change at a critical state can flip success. That is
  why stage 2 exists and why the proposal's consequence weighting could, in principle, beat the
  `o2` bound (the bound is on distribution matching, not on weighting).
- GPTQ here is non-sequential (every layer's Hessian comes from FP inputs) and has no act-order.
  This is the same for every arm, so comparisons stay fair, but absolute GPTQ numbers may be
  a little worse than the reference implementation's sequential mode.
- One shift type. Repeat on language (largest gap after camera) and lighting (no gap reported,
  a negative control) before generalizing.
