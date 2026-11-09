# On-policy quantization: discussion notes

Updated 2026-11-08 from the team's notes, shared discussion, and primary papers.
These are research hypotheses and proposed controls, not implemented methods or
novelty claims. The current experiment is the nominal OpenVLA-OFT LIBERO-Long
comparison described in the [README](../README.md); its run status is in the
[changelog](../CHANGELOG.md).

The question is whether fitting a quantized policy on observations it actually
visits improves task success and robustness compared with fitting on fixed
demonstrations. Keep two choices explicit: the outer loop controls data
collection and aggregation; the inner loop controls the objective, quantization
grid, granularity, rounding, and trainable parameters. A modular, swappable
implementation is compatible with a substantive quantization contribution.
Evidence for that contribution should come from controlled ablations.

The literature constrains the claim:

| Source | What matters for this project |
| --- | --- |
| [DAgger, Ross et al. (2011)](https://proceedings.mlr.press/v15/ross11a.html) | Collect learner-induced observations, query expert labels, aggregate data, and update the learner. Its guarantees require the stated no-regret/surrogate-loss assumptions. Recomputing GPTQ statistics alone does not establish those assumptions or a linear task-cost bound. |
| [VLAQuantBench (2026)](https://arxiv.org/html/2609.25376v1) | Quantization scope, numerical format, calibration, and layer interactions affect closed-loop success. A small OFT output projection can dominate failure. Match these choices across methods; distinguish simulated quantization from real-kernel measurements. Our initial scope protects the entire action head. |
| [QAIL (2024), §4 / Algorithm 1](https://arxiv.org/html/2412.01034v1) | Combines quantization-aware imitation learning with FP-policy action-distribution matching. Algorithm 1 collects FP-policy data, while Eq. 3 writes the discrepancy expectation under the quantized policy. Audit the actual collection/training procedure before claiming that learner-state distillation is absent from prior work. |
| [SQIL (2025), Algorithm 1 / §4.3](https://arxiv.org/html/2505.15304v2) | Uses saliency-weighted action distillation and QAT over an existing expert dataset. State importance and teacher matching already have close precedents; adding them alone is insufficient to establish novelty. |
| [LoopQ (Nov. 3, 2026), §7](https://arxiv.org/html/2611.05093v1) | A recent comparison of generic and VLA-specific quantizers on matched scopes/engines, including paired LIBERO-plus perturbations. Nominal success can hide robustness losses. Its protocols and implementations deserve inspection before defining the next baseline or claiming deployment speedups. |

The proposed experiment sequence preserves the original gates:

1. Finish the paired FP/AWQ nominal baseline. Add RTN and GPTQ at the same W4,
   group-128 decoder scope, with a matched calibration budget and seeds.
2. Test a data-only outer loop: roll out the quantized policy on a separate
   collection split, retain a task-balanced mixture of demonstrations and prior
   learner observations, and recalibrate the unchanged quantizer. Rebuild from
   the original unquantized checkpoint each iteration. Record rollout steps,
   teacher calls, calibration frames, and total GPU time.
3. If that helps, choose an inner update aligned with the observed failure mode.
   One candidate is to learn group scales/clipping using teacher action-chunk
   loss on the aggregated observations, with the weight-bit budget and protected
   modules fixed. Start with ordinary rounding and an explicit straight-through
   gradient approximation; test learned rounding or mixed precision separately.
   This candidate still needs a prior-work audit and comparison against standard
   distillation/QAT. It is not yet the selected method.

Use a factorial comparison to identify the source of improvement:

| Collection data | Existing inner method | Proposed inner method |
| --- | --- | --- |
| Fixed demonstrations | Baseline | Objective/update effect |
| Aggregated learner observations | Data-only loop | Combined method |

Match bit width, quantized layers, sample count, teacher/environment budget, and
optimization effort. Include an extra-compute control that refreshes only
expert-policy observations. Otherwise additional data or optimization could
explain the apparent benefit. Select iterations/hyperparameters on validation
episodes and evaluate once on separate nominal and perturbed test episodes.

For OFT, collect observations at eight-action chunk boundaries and query the
teacher for the corresponding complete chunk. If using expert/learner mixing,
choose the controlling policy at those boundaries and record the schedule.
Compare action targets in a consistent normalized coordinate system, specify
gripper weighting, and check teacher behavior on states far from demonstrations.
The existing evaluation wrapper calls upstream inference-only code; QAT needs a
separate differentiable forward path.

Several distinctions from the shared discussion need to stay explicit:

- **GPTQ curvature is a reconstruction proxy.** For fixed inputs, its quadratic
  layer loss has Hessian `2 X X^T`; it is not the Hessian of rollout success.
  [GPTQ, §3–4](https://arxiv.org/html/2210.17323v2)
- **Scalar weighting is an approximation.** Our derivation: weighting samples
  by nonnegative `w(s)` gives `H = 2 sum_s w(s) X_s X_s^T` for weighted layer
  reconstruction. An action-space linearization instead contains the downstream
  Jacobian and generally couples output channels and tokens. Replacing that
  Jacobian by a scalar sensitivity loses directional information. Independent
  per-row Hessians are not exact when cross-row terms remain.
- **On-policy states and quantized internal activations are separate choices.**
  Collecting states with the learner does not logically require a particular
  teacher/student activation path during reconstruction. Ablate that choice.
- **A Jacobian norm is not generally one ordinary backward pass.** Exact
  action-Jacobian evaluation needs multiple output derivatives; random-projection
  vector-Jacobian products can estimate its squared Frobenius norm. State the
  estimator, cost, and variance if using this signal.
- **QAT and QLoRA need separate table entries.** QAT is a training approach;
  QLoRA trains low-rank adapters through a frozen quantized base. Specify whether
  an experiment changes base weights, adapters, or quantizer parameters.
  [QLoRA](https://arxiv.org/abs/2305.14314)

OOD robustness remains a hypothesis. Learner-induced drift and an externally
shifted camera, lighting, language, or object layout are different sources of
distribution shift. Define held-out perturbation types/severities and pair the
FP and quantized evaluations on them. Report the FP policy's own degradation as
well as the additional quantization gap. The current nominal baseline cannot
establish this claim.

For Monday's discussion, the open decisions are the collection/validation/test
split, the first inner-loop candidate, the resource budget, and the OOD protocol.
Slides and the DAgger/QAT implementation remain separate TODOs.
