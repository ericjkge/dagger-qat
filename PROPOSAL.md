# Consequence-Weighted On-Policy Calibration for Training-Free VLA Quantization

## 1. Problem Statement and Motivation

**Question:** in training-free post-training quantization (PTQ) of a vision-language-action (VLA)
model, which states should the quantizer see, and how much should each count?

Under standard LIBERO evaluation, VLA PTQ already looks nearly lossless. QuantVLA reports π0.5 at
97.1% average success in FP16 and 97.6% at W4A8, so there is no in-distribution gap to close. The
gap shows up elsewhere: recent work finds that quantized VLAs stay comparable in-distribution but
become fragile under the environmental perturbations in LIBERO-Plus.

We test a concrete mechanism for that fragility. Methods like GPTQ and SmoothQuant fit
quantization parameters to a calibration distribution through the layer-input Hessian
H = Σ XXᵀ and activation ranges. That distribution is normally clean offline frames. At
deployment, the quantized policy visits different states: perturbed scenes, plus states its own
action errors produce. If the calibration Hessian and activation ranges don't reflect those
states, reconstruction error grows exactly where the policy is deployed.

Two findings shape our approach:

- **Errors are sparse, not gradual.** SQIL and PIVOT-Q both report that action discrepancies
  between quantized and FP policies concentrate in a small number of critical states, rather
  than accumulating smoothly along a trajectory.
- **On-policy coverage alone can hurt.** In multimodal LLMs, OnPTQ found that uniformly
  reconstructing on the quantized model's own trajectories lowered accuracy by about 2.5 points.
  Selecting states by their downstream consequence was necessary.

So our object of study is not "on-policy data" but a **consequence-weighted calibration
distribution**: which states define the quantizer's objective, and how heavily each one is
weighted.

## 2. Related Work and Positioning

The on-policy idea has already been applied to MLLM PTQ (OnPTQ) and to VLA distillation
(PIVOT-Q). The open gap is training-free, closed-form PTQ on a continuous-action, flow-matching
VLA, with the calibration distribution chosen by downstream consequence.

| Work | Domain | Updates weights by gradient? | Calibration / training states | Uses downstream consequence? |
|---|---|---|---|---|
| QuantVLA (CVPR 2026) | VLA (π0.5, GR00T) | No | Small unlabeled offline buffer | No |
| QAIL | VLA / driving | Yes (QAT) | Demos + FP-policy rollouts | No |
| SQIL | VLA (OpenVLA) | Yes (QAT) | Demos, saliency-weighted | Saliency of critical states |
| PIVOT-Q (Oct 2026) | VLA, LIBERO-Plus | Yes (on-policy distillation) | Quantized-student rollouts | Short-horizon accumulated discrepancy |
| OnPTQ (Sep 2026) | MLLM (discrete tokens) | Calibration params only | Quantized-model trajectories, refreshed | Counterfactual branch rollouts |
| Self-calibration (NAACL 2025) | LLM | No | Model-generated text | No |
| **This project** | VLA (π0.5), LIBERO-Plus | No (closed-form GPTQ / SmoothQuant) | Offline, FP16, quantized and perturbed rollouts | Consequence-weighted Hessian |

What we add beyond this work:

- **Diagnosis before method.** An oracle bound and a Hessian cross-evaluation (Section 5) that
  measure whether calibration data can matter at all for VLA PTQ, and in which layers. None of
  the papers above reports this.
- **A cheap-alternative test.** Does the expensive quantized-policy loop beat FP16 rollouts,
  noise-injected rollouts, or perturbed demos? If not, that is a practical, publishable answer.
- **Continuous actions and closed-form solvers.** OnPTQ's consequence score is built on token
  margins and learnable calibration parameters. We adapt consequence weighting to action chunks
  and to GPTQ's closed-form Hessian, where it becomes a weighted second-moment matrix.

## 3. Formulation: Calibration as Reweighting, Not Imitation

We treat calibration as covariate-shift correction of the quantizer's objective. We do not treat
it as DAgger-style imitation, for a specific reason.

GPTQ solves, layer by layer:

$$
\hat{W} = \arg\min_{\hat{W} \in \mathcal{Q}} \; \mathrm{tr}\big((W-\hat{W})\, H\, (W-\hat{W})^{\top}\big),
\qquad H = \sum_{s \in \mathcal{D}} X_s X_s^{\top}
$$

The FP16 weights W are already the "teacher" at every state. No action labels enter the
objective. The calibration set affects the result only through H, and, for activation
quantization, through the activation ranges and smoothing scales. In DAgger, the expert's
corrective labels carry the signal. Here there are none to add, so querying the FP16 model for
reference actions (Step 3 of our earlier draft) changes nothing in GPTQ. Calling the method
DAgger invites the "where does the supervision come from?" circularity our TA raised.

The correct framing is a choice of state distribution and weights. Deployment error is an
expectation over the states the quantized policy actually visits, d_Q. Our target is therefore:

$$
H_w = \sum_{s \in \mathcal{D}_{\mathrm{mix}}} w(s)\, X_s X_s^{\top},
\qquad w(s) = \underbrace{\rho(s)}_{\text{coverage}} \cdot \underbrace{\big(1 + \lambda\, c(s)\big)}_{\text{consequence}}
$$

- **D_mix** pools offline demos, FP16 rollouts, quantized-policy rollouts and perturbed
  observations.
- **ρ(s)** corrects coverage toward the deployment distribution. This includes the uniform
  on-policy case.
- **c(s)** is a consequence score: how much a quantization error at s changes the outcome. We
  estimate it from short simulator rollouts (Section 4).
- **λ = 0** recovers on-policy calibration with uniform weights. Ablating λ separates *which
  states* from *how much they count*.

For activation quantization (A8/A4), we apply the same weights to the statistics that set
clipping ranges and SmoothQuant scales (weighted quantiles and weighted per-channel maxima).

Three honest limits of this framing:

- **The ceiling is FP16.** Matching FP16 at off-distribution states cannot teach recovery if
  FP16 also fails there. Our goal is to close the FP16-vs-quantized gap, not to beat FP16.
- **No convergence guarantee.** DAgger's guarantees rely on a no-regret learner. GPTQ's map from
  data to rounded weights is discontinuous, so iterated refresh may oscillate. OnPTQ saw more
  refresh rounds perform worse. We treat refresh as an empirical question.
- **Stochastic policy.** π0.5 samples action chunks via flow matching. Every FP16-vs-quantized
  comparison fixes the initial noise, so discrepancy measures quantization, not sampling
  variance.

## 4. Methodology

The pipeline produces one quantized π0.5 from frozen FP16 weights. There are no gradient steps.
Only the calibration distribution and its weights change.

> *Pipeline diagram: see the project Google Doc (embedded there, not reproduced here).*

The highlighted stage is the contribution. Everything to its left decides which states enter the
quantizer's statistics and how much each one counts.

1. **Initial quantization.** Quantize with GPTQ (W4) or SmoothQuant (W4A8 / W4A4) on the
   standard clean offline calibration set. This uses the existing `src/vlacal/quant`
   implementations.
2. **Collect candidate states.** Roll out the policies on LIBERO-Plus training-split
   perturbations (never the evaluation split) and pool the states into D_mix. Rollouts use a
   β-mixed policy: at each step, FP16's action runs with probability β, and the current
   quantized policy's action otherwise. β = 1 gives FP16 rollouts, β = 0 gives pure on-policy
   rollouts, and values in between interpolate. The pool includes:
   - offline demos
   - FP16-policy rollouts
   - FP16 rollouts with injected action noise (DART-style)
   - perturbed demos
   - quantized-policy rollouts
3. **Score consequence c(s).** For a subset of candidate states, reset the simulator to s, then
   run two short branches with the same flow-matching noise: one where the quantized policy acts
   at s and one where FP16 acts at s. Both then continue under FP16 for H steps (H = 1–2 action
   chunks). c(s) is the divergence between the branches' end-effector and object states after H
   steps. A cheaper proxy, discounted accumulated action discrepancy (as in PIVOT-Q), serves as a
   baseline scorer.
4. **Build the weighted calibration set.** Either select the top-K states by c(s) under a fixed
   sample budget, or keep a uniform mix and weight each state's contribution to H and to the
   activation-range statistics by w(s).
5. **Co-design: where calibration matters.** For each layer l, measure shift sensitivity as the
   ratio of reconstruction error of the offline-calibrated weights evaluated under the on-policy
   Hessian to the same error under the offline Hessian. Layers with high sensitivity get the
   weighted Hessian, and are candidates for higher precision under a fixed average bit budget.
   Insensitive layers keep cheap offline calibration. This links calibration data, the
   reconstruction objective and bit allocation, instead of treating them separately.
6. **Optional refresh (at most 2 rounds).** Re-roll with the new quantized policy (β decaying
   across rounds, as in DAgger), aggregate the states with the existing pool, re-score, and
   re-quantize. Every round requantizes from the original FP16 weights, never from the previous
   quantized iterate, so rounding errors do not stack across rounds. We stop at two rounds
   because OnPTQ found more rounds harmful, and we report whether round 2 helps.
7. **Model selection.** We return the iterate and setting (round, β, λ) with the highest success
   on a held-out selection split of LIBERO-Plus rollouts. This split is disjoint from both the
   calibration states and the evaluation episodes, so selection does not leak into the reported
   numbers.

All quantization is simulated (quantize-dequantize), as in the current repo. We measure task
success and calibration cost, not kernel latency.

## 5. Preliminary Go / No-Go Experiments

These four experiments come before any method work. Each one can end the project cheaply, and
their results are what we bring to HT. We run them in order and stop at the first failed gate.

| # | Question | Measurement | Go if | No-go response |
|---|---|---|---|---|
| **P1** Headroom | Is there a FP16-vs-quantized gap to close? | Success of FP16 vs. GPTQ W4, SmoothQuant W4A8 and W4A4 on a LIBERO-Plus subset, paired seeds | Gap ≥ 3 points in at least one setting | Move to lower bits (W3, W4A4) before rethinking |
| **P2** Oracle | Can *any* calibration data close the gap? | Calibrate on held-out rollouts drawn from the evaluation perturbation itself (upper bound) | Oracle recovers ≥ 1/3 of the P1 gap | Calibration data is not the lever; report this as the finding |
| **P3** Hessian shift | Do the statistics GPTQ uses actually move, and where? | Per-layer cross-evaluation: quantize with H_offline, score under H_onpolicy, and the reverse; activation-range shift per layer | Some layers show a clear increase in cross-error | Effect is in activation ranges only, or absent; narrow scope accordingly |
| **P4** Sparsity | Are discrepancies concentrated in a few states? | Distribution of fixed-noise action discrepancy along rollouts | A small share of states holds most of the discrepancy | Uniform on-policy weighting may suffice; λ becomes secondary |

P2 is the most important. It bounds every method in this proposal. If oracle calibration on the
test distribution barely helps, no clever choice of calibration states will. That is still a
clean, useful result.

The go thresholds are our working choices for a course-project budget. We will fix them before
running P1.

## 6. Main Experiments

We use one model (π0.5), one primary quantizer and bit width chosen by P1, and LIBERO-Plus. The
second quantizer is a secondary check. Every arm uses the same quantizer, bit width and
calibration sample budget. Only the calibration distribution and weights differ.

| Arm | Calibration states | Weights | What it isolates |
|---|---|---|---|
| A0 Standard PTQ | Clean offline demos | Uniform | Baseline |
| A1 Larger random set | Clean offline demos, 4× budget | Uniform | Is it just more data? |
| A2 Perturbed demos | Demos with LIBERO-Plus-style perturbations | Uniform | Visual shift alone |
| A3 FP16 rollouts | FP16-policy states | Uniform | Policy-induced shift, no quantized loop |
| A4 Noisy FP16 rollouts | FP16 + injected action noise (DART-style) | Uniform | Cheap stand-in for quantized drift |
| A5 On-policy | Quantized-policy states | Uniform (λ = 0) | Value of the quantized loop |
| A6 Consequence-weighted | Pooled D_mix | w(s), branch-rollout score | Full method |
| A7 Proxy-weighted | Pooled D_mix | w(s), accumulated-discrepancy proxy | Is the expensive scorer needed? |
| A8 + refresh | A6 after one refresh round | w(s) | Value of iteration |
| Oracle | Evaluation-perturbation states | Uniform | Upper bound (from P2) |

**Protocol.**

- Evaluation is paired: every arm sees the same initial states, perturbations and noise seeds, so
  we compare per-episode differences.
- Calibration states never come from the evaluation split or the held-out selection split.
- A β sweep (β ∈ {1, 0.5, 0}) under uniform weights connects A3 (β = 1) to A5 (β = 0), showing
  how much quantized-policy states matter as a function of mixing.
- We report 95% confidence intervals on paired differences. At about 90% success, one arm with
  500 episodes has a standard error near 1.3 points, so we need roughly 1,000+ paired episodes
  per arm on the main split to resolve a 2–3 point difference.

**Metrics.**

- Task success on LIBERO-Plus, by perturbation type, plus clean LIBERO as a no-regression check.
- Fixed-noise action discrepancy vs. FP16, overall and at the top-scored states.
- Per-layer reconstruction error under each Hessian, and per-layer shift sensitivity.
- Calibration cost in GPU-hours (rollouts + scoring + quantization), plotted against success
  gained over A0.

We drop memory footprint and inference latency as comparison metrics. Every arm produces the same
quantized format, so these numbers are identical, and simulated quantization cannot measure real
latency.

## 7. Hypotheses and Falsification Criteria

Each hypothesis is tested on its own and has a stated way to fail. A negative result on any of
them is reported as a finding.

| Hypothesis | Supported if | Falsified if |
|---|---|---|
| **H1 Statistic shift.** The calibration statistics GPTQ/SmoothQuant depend on (H, activation ranges) shift meaningfully between offline and deployment states, concentrated in identifiable layers. | P3 shows elevated cross-error in a subset of layers, larger under perturbations | Cross-error is flat across layers and sources |
| **H2 Calibration can matter.** Calibrating on deployment-distribution states closes a meaningful part of the quantization gap. | Oracle recovers ≥ 1/3 of the P1 gap | Oracle ≈ A0 |
| **H3 The loop is necessary.** Quantized-policy states (A5) beat cheaper proxies (A2–A4). | A5 > max(A2, A3, A4) with paired CI excluding 0 | A cheaper arm matches A5, so the loop isn't needed |
| **H4 Consequence matters.** Weighting by c(s) beats uniform on-policy coverage. | A6 > A5 with paired CI excluding 0 | A6 ≈ A5, or A6 < A5 |
| **H5 Cheap scoring suffices.** The discrepancy proxy is as good as branch rollouts. | A7 ≈ A6 at lower cost | A6 ≫ A7 |

H3 and H5 are framed so that either outcome is useful. Either the expensive component earns its
cost, or practitioners can skip it.

## 8. Expected Contributions and Significance

The main contribution is an answer to *when* calibration data matters for VLA quantization, and
*what it is worth*, backed by a measured upper bound.

- **A diagnostic.** The oracle bound plus per-layer Hessian cross-evaluation measures how much any
  calibration-data method can help a quantized VLA, and in which layers. It is reusable by anyone
  proposing a calibration method.
- **Consequence-weighted Hessians for closed-form PTQ.** A training-free way to fold downstream
  task consequence into GPTQ and SmoothQuant, adapted to continuous action chunks.
- **A cost-benefit map.** Success gained per GPU-hour of calibration, across offline, perturbed,
  FP16, noisy-FP16, on-policy and consequence-weighted sources. It shows whether the closed loop
  is worth running.
- **A co-design result.** Whether calibration-shift sensitivity identifies the layers that
  deserve both the better calibration data and the higher precision.

**Significance.** Deployment on resource-constrained hardware happens under shifted conditions,
not on clean benchmark frames. If calibration data is a cheap, training-free lever for
robustness, this tells practitioners how to spend it. If it is not, the oracle bound settles the
question and points effort back toward rotation, mixed precision or quantization-aware training.

## 9. Scope, Compute, Risks and Timeline

**Scope.** One model (π0.5), one primary quantizer and bit width, LIBERO-Plus with clean LIBERO
as a regression check. A second quantizer is added only if time allows. A real-robot demo is out
of scope: LIBERO-Plus perturbations already stand in for real-world shift, and a demo would not
strengthen the central claim.

**Compute.** Rollouts dominate cost. During P1 we will measure wall-clock time per episode on our
hardware, then size the episode counts in Section 6 to fit our GPU budget. If the budget is
tight, we cut perturbation types before cutting episodes per arm. Underpowered comparisons are
worse than narrower ones.

| Risk | Likelihood | Mitigation |
|---|---|---|
| No headroom at W4A8 | High (QuantVLA is near-lossless in-distribution) | Use LIBERO-Plus and W4A4 / W3 from the start (P1) |
| Oracle calibration barely helps | Medium | Report it as the main finding; pivot the analysis to why (P3 per-layer results) |
| Branch-rollout scoring is too expensive | Medium | Score a subsample; use the discrepancy proxy (A7) |
| Refresh oscillates | Medium | Aggregate the state pool; cap at two rounds; report it |
| Differences inside noise | Medium | Paired seeds, CIs, fewer arms with more episodes |
| Simulator reset to arbitrary states is unreliable | Low–medium | Branch from saved LIBERO sim states only; verify determinism with FP16-vs-FP16 branches |

**Timeline (relative weeks).**

- **Week 1:** P1 headroom and P4 sparsity, using the existing quantizers and rollout server.
  Measure episode cost.
- **Week 2:** P2 oracle and P3 Hessian cross-evaluation. Meet HT with these results.
- **Weeks 3–4:** Arms A0–A5, then A6–A7 if the gates pass.
- **Week 5:** A8 refresh, co-design layer analysis, second quantizer if time allows.
- **Week 6:** Analysis and write-up.

## References

- Zhang et al. (2026). QuantVLA: Scale-Calibrated Post-Training Quantization for
  Vision-Language-Action Models. CVPR 2026.
- Fan et al. (2026). Calibrate the Decisions That Change the Future: On-Policy Post-Training
  Quantization for Multimodal Large Language Models (OnPTQ).
- Ruan et al. (2026). Beyond In-Distribution Preservation: Recovering Generalization in Quantized
  VLAs via Vulnerability-Oriented Tuning (PIVOT-Q).
- Park et al. (2024). Quantization-Aware Imitation-Learning for Resource-Efficient Robotic
  Control (QAIL).
- Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control (SQIL), 2025.
- Williams, Chrysostomou and Aletras (2025). Self-calibration for Language Model Quantization and
  Pruning. NAACL 2025.
- Frantar et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers.
- Xiao et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
  Language Models.
- Ross, Gordon and Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning (DAgger).
- Ross and Bagnell (2014). Reinforcement and Imitation Learning via Interactive No-Regret
  Learning (AggreVaTe).
- Laskey et al. (2017). DART: Noise Injection for Robust Imitation Learning.
- Xu et al. (2026). VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for VLA
  Models.

Project code: [ericjkge/vla-calibration](https://github.com/ericjkge/vla-calibration).
