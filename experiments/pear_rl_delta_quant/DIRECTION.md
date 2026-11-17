# Next research direction after PEAR: keeping a robot skill learned by RL alive at 4 bits (2026-11-15)

**Recommendation: GO to a cheap behavioral pilot.** The weight-level evidence is measured. The behavioral claim is not
yet tested.

## 1. The problem

**PEAR cannot ship its own 4-bit model.** The paper says: "NVFP4 quantization reduces our model's accuracy and lengthens
its chain of thought, so all evaluations use BF16 weights" (§4.6). It gives no numbers.

**Why it matters for deployment:**
- PEAR's consumer-hardware speeds (221 tok/s on an RTX 5090, 52 tok/s on an M5 Max) are NVFP4 numbers.
- A 27B model in BF16 is 54 GB and does not fit on a 32 GB RTX 5090.
- So the accuracy PEAR reports and the speed it reports come from two different models.

**What the released NVFP4 checkpoint quantizes** (`hf_quant_config.json`, ModelOpt 0.48):
- All MLPs and the lm_head use NVFP4 W4A4.
- All attention projections, both Gated DeltaNet and full attention, use FP8.
- The vision tower is left unquantized.

This is already a conservative mixed-precision recipe from the lab that wrote AWQ, and the paper says it still hurts.

## 2. What we measured

**Why the skill is suspect.** In PEAR, the robot skill comes almost entirely from RL: LIBERO success goes from 1.6% for
the base model to 74.4% after RL. So the skill should live in the weight difference Δ = W_PEAR − W_base.

**Method.**
- I pulled individual tensors from three checkpoints by HTTP range requests:
  - `Qwen/Qwen3.8-27B` (the base model);
  - `mit-han-lab/Qwen3.8-27B-PEAR` (BF16);
  - `mit-han-lab/Qwen3.8-27B-PEAR-NVFP4`.
- I decoded the checkpoint's own FP4 codes and scales.
- I re-rounded the base and the PEAR weights **on the deployed grid**: the checkpoint's own block scales plus
  round-to-nearest. This reproduces 91–99% of the checkpoint's stored values exactly.
- Scripts are in `scripts/`; raw outputs are in `results/`.

| Matrix (format) | RL update ‖Δ‖/‖W‖ | Quantization error ‖E‖/‖W‖ | Signal-to-noise ‖Δ‖/‖E‖ | Weights whose quantized value RL changed | Similarity of quantized update to true update (cosine) | Share of the RL update kept |
|---|---|---|---|---|---|---|
| L10 `mlp.up_proj` (NVFP4) | 0.16% | 8.4% | 0.019 | 0.31% | 0.095 | 0.69 |
| L10 `mlp.down_proj` (NVFP4) | 0.16% | 8.5% | 0.019 | 0.31% | 0.097 | 0.70 |
| L40 `mlp.gate_proj` (NVFP4) | 0.15% | 8.6% | 0.018 | 0.29% | 0.095 | 0.70 |
| L63 `mlp.down_proj` (NVFP4) | 0.14% | 11.9% | 0.012 | 0.27% | 0.095 | 0.72 |
| `lm_head`, rows 1k–21k (NVFP4) | 0.07% | 8.4% | 0.009 | 0.11% | 0.085 | 0.73 |
| L10 Gated DeltaNet `in_proj_qkv` (FP8) | 0.09% | 2.7% | 0.035 | 3.8% | 0.26 | 1.02 |
| L11 full-attention `q_proj` (FP8) | 0.09% | 2.7% | 0.033 | 3.7% | 0.27 | 1.00 |

How to read the last three columns:
- "Weights whose quantized value RL changed" is the share of weights that land on a different 4-bit (or 8-bit) value
  in the PEAR model than in the base model.
- The cosine compares the quantized update Q(W_PEAR) − Q(W_base) with the true update Δ. A value of 1 would mean the
  quantized model kept the update intact.
- "Share of the RL update kept" is the projection ⟨Q(W_PEAR) − Q(W_base), Δ⟩ / ‖Δ‖².

**What this shows:**
1. **RL's update is about 50–100× smaller than NVFP4's rounding error.** The median per-weight change is 0.5% of one FP4
   grid step. RL left about 65% of BF16 weights bit-identical, which matches the finding that RL updates a sparse
   subnetwork ([2507.17107](https://arxiv.org/abs/2507.17107)).
2. **At 4 bits the update survives only as rare jumps.** Only about 0.3% of weights change their 4-bit value because of
   RL. The resulting quantized update has cosine 0.09 with the true one, meaning it is about 90% noise. About 30% of
   the update is also lost systematically (column "share kept" ≈ 0.70), because large weights are clipped and the FP4
   grid is uneven.
3. **FP8 attention also scrambles it** (cosine 0.26), although nothing is lost on average.
4. **Adjusting scales cannot fix this.** The update is a fraction of a percent of one grid step. Only the per-weight
   rounding choices have enough freedom to carry it.
   - A naive test, which rounds near-tie weights toward the sign of Δ, raises the cosine from 0.095 to 0.26 for a 1%
     increase in reconstruction error (`results/delta_directed_rounding.txt`).
   - But that test overshoots badly (share kept up to 6.3 instead of 1). The flips have to be chosen against a
     functional target, not by sign.

**Hypothesis to test.** NVFP4 PEAR loses accuracy and reasons longer because 4-bit rounding drowns the RL update. The
quantized model then drifts back toward the base model, which succeeds only 1.6% of the time.

This weight-level evidence does **not** prove the behavioral effect. The decisive test is the pilot in §5.

## 3. Prior work and the open gap

| Work | What it does | Why it does not close the gap |
|---|---|---|
| DAQ ([2603.22324](https://arxiv.org/abs/2603.22324)) | Delta-aware post-training quantization (PTQ): picks scales that preserve the sign and direction of Δ, without data | Searches scales only, FP8 only, on one toy SFT style task. Its own "future work" lists learned rounding and lower bit widths. Our numbers show scales cannot carry Δ at 4 bits |
| QuRL ([2602.13953](https://arxiv.org/abs/2602.13953), ICLR 2026) | Finds RL updates are much smaller than quantization error during quantized RL *training*; rescales weights so updates cross rounding boundaries | Training-time only. Does not address deploying an RL-trained model after the fact |
| QUAIL ([2601.15538](https://arxiv.org/abs/2601.15538)), [2602.13151](https://arxiv.org/abs/2602.13151) | 4-bit quantization undoes LLM *unlearning*, because the unlearning update is too small to cross rounding boundaries | Same mechanism, different domain. Fixes it by changing training |
| BitDelta ([2402.10193](https://arxiv.org/abs/2402.10193)), DeltaLLM | Compress the fine-tuning delta on top of a base kept in high precision | Extra memory and two weight sets. Does not keep the deployed 4-bit format |
| Q-resafe / Q-realign | Restore safety alignment after quantization with DPO-style patching | Training-based, for safety rather than an RL skill |
| QAD ([2601.20088](https://arxiv.org/abs/2601.20088)), SteerDistill ([2611.13410](https://arxiv.org/abs/2611.13410)) | Recover quantized accuracy by distillation, including on-policy and multi-turn agents | Training a 27B model on 94k-token episodes means PEAR-scale infrastructure. Not a post-training method |
| Overthinking penalty ([2606.00206](https://arxiv.org/abs/2606.00206)), ReSET ([2606.13233](https://arxiv.org/abs/2606.13233)) | Decoding-time fixes for quantized models that reason longer | Treat the symptom. Tested on single-turn math and code. Complementary baselines |
| Quantized RL reasoning ([2511.15694](https://arxiv.org/abs/2511.15694)) | Finds PTQ after math RL keeps reasoning intact down to 4 bits | **Counterpoint.** Math RL refines a base that is already capable; PEAR's skill is almost entirely the delta. The pilot has to show the difference |

**The gap:** no published post-training method preserves an RL-learned skill at 4 bits, and none has been tested in a
closed-loop embodied setting.

## 4. Proposed method: delta-anchored rounding

**Fixed constraints.** The deployed artifact is a standard NVFP4 checkpoint: same bit width, same kernels, same speed.
Only the per-weight rounding choices (and optionally the block scales) change.

**Objective, per linear layer:**

min over Q:  ‖(Q − W_P) X‖²  +  λ ‖U_kᵀ (Q − W_P) X_s‖²

- X: activations from general calibration text plus robot contexts.
- X_s: activations from **on-policy robot contexts**. These are collected by running the quantized policy in LIBERO,
  which is the team's DAgger outer loop.
- U_k: the top-k output directions of the RL update's effect on those contexts, i.e. the top-k left singular vectors of
  ΔW X_s.
- The second term penalizes rounding error that cancels or distorts the change RL made, on the contexts where the robot
  actually acts. The first term treats all other error as the generic noise the base model already tolerates.
- The penalty has rank k, so the problem stays in the GPTQ / optimal-brain-surgeon family: row-wise updates plus k
  coupled projection rows. AdaRound-style learned rounding on each block is the fallback solver.

**Why this is one coherent method and not a bag of tricks:**
- It follows from a measured mechanism: the skill update is a fraction of one grid step, so only rounding has the
  capacity to carry it.
- It exploits two features of robot RL that generic LLM quantization does not have:
  - the skill is an explicit delta from an open base model;
  - a simulator supplies the contexts where the skill acts, generated by the quantized policy itself.
- The DAgger-style on-policy calibration from `docs/research-notes.md` becomes the data source of the method rather
  than a separate add-on.

## 5. Plan and stopping rules

**Stage 1: attribution pilot.** Runs on our existing H200 SGLang + LIBERO stack, about $50.

To isolate the effect of weights, each arm runs **fake-quantized weights in BF16**:

| Arm | Weights |
|---|---|
| A | PEAR BF16 |
| B | PEAR NVFP4 weights, dequantized from the checkpoint (activations unquantized) |
| C | Oracle: dequantized RTN(W_base) on the same grid, plus Δ in BF16. Same noise magnitude as B, exact skill update |
| D | Quantized base model only (sanity check) |

**Metrics:**
- KL divergence from BF16 on our 14 recorded episodes (teacher-forced, so no new sampling).
- Thinking tokens per round.
- Success and context-exhaustion rate on LIBERO Goal and Long subsets, with paired seeds.

**Stopping rules:**
- **GO** if B is worse than A (success, or ≥ 10% longer reasoning) **and** C recovers at least half of the gap.
- **NO-GO** if C ≈ B: restoring the exact update does not help, so the mechanism is wrong.
- **Pivot** if B ≈ A: weight rounding is not the cause. Test FP4 activations and the W4A4 lm_head next.

**Caveat:** 40 episodes per arm gives success-rate intervals of about ±15 points, so the reasoning-length and KL
metrics carry most of the weight.

**Stage 2: method.**
- Implement delta-anchored rounding on the NVFP4 grid and export a ModelOpt-format checkpoint.
- Baselines: ModelOpt RTN (the released checkpoint), GPTQ-NVFP4, AWQ-NVFP4, DAQ scale search, and naive
  delta-directed rounding.
- Decoding fixes ([2606.00206](https://arxiv.org/abs/2606.00206), ReSET) can be combined with any arm.

**Stage 3: real hardware.**
- Real W4A4 kernels on a B200 or B300.
- Show the same tok/s as the released NVFP4 checkpoint.
- Measure closed-loop success and context exhaustion on LIBERO.
- Run general benchmarks to show the base model's capabilities are unchanged.

**Risks:**
1. The NVFP4 harm may come mostly from FP4 activations or the W4A4 lm_head. Arms B vs. C vs. real W4A4 separate these.
2. The PEAR authors (MIT Han lab) may release a distilled (QAD) NVFP4 model. Our differentiators: post-training only
   (minutes, no RL infrastructure), and it applies to any skill added by RL.
3. Success-rate variance is large. The paired design and the reasoning-length metric mitigate this.

## 6. Other directions examined and rejected

| Direction | Evidence | Verdict |
|---|---|---|
| Structure-aware speculative decoding | `../pear_specdec/RESEARCH_DECISION.md`: < 5% headroom over DFlash2 | NO-GO |
| Early exit from reasoning | `results/action_commit_point.txt`: the final action's values are first all stated at a median of **93%** through the reasoning; at most 31% of reasoning tokens (token-weighted) come after that point. The area is crowded (DEER, Dynasor, SABER, [survey 2611.00166](https://arxiv.org/abs/2611.00166)) | Too little headroom, and published |
| Racing parallel reasoning chains | `results/race_lengths.txt`: per-context reasoning length varies widely (coefficient of variation 0.17–1.53 over 9 samples). Taking the first of 4 parallel chains cuts the expected sequential tokens **1.61×** (lengths capped at 8,192). Real headroom, but already published as First Finish Search ([2505.18149](https://arxiv.org/abs/2505.18149)), short-m@k, and M1-Parallel | Would only be an application. Possibly a later add-on |
| Context exhaustion (PEAR: 19% / 33% of Goal / Long episodes, and 68.5% of RoboDojo episodes, end at the 256k-token budget) | The largest failure mode, but agent KV eviction and context management are crowded (SideQuest, AgentKV, CommitKV, IntentKV), and hybrid-model eviction exists (GateSight [2610.14754](https://arxiv.org/abs/2610.14754), DeltaS). The simplest fix is a chat-template flag that drops earlier thinking | Second choice; the method would be a trick |
| Quantization error accumulating in the recurrent state | LeapQuant, StepQuant, and [2609.04098](https://arxiv.org/abs/2609.04098) (Gated DeltaNet survives NVFP4 W4A4) | Published |
| Long-context degradation of the model | `results/surprisal_vs_context.txt`: the surprisal of sampled reasoning tokens falls slightly from 0.27 to 0.22 nats from 0–8k to 128k+ context; the acceptance drop is a drafter effect | No evidence of a problem |

## 7. Pilot result (2026-11-15): NO-GO

**Run:**
- One Community H200, about 35 minutes, roughly $2.
- Pod scripts: `pod/`. Analysis: `scripts/analyze_pilot.py`. Raw data: `results/pilot/`.
- Phase 2 (closed loop) was not run, because phase 1 already met the NO-GO rule.

**Method.**
- Teacher-forced scoring of all 14 recorded BF16-PEAR episodes (410k generated tokens), one prefill per episode.
- Metric: dNLL, the per-token NLL increase over arm A, measured on A's own samples. This estimates KL(A‖arm).
- 95% CIs come from an episode bootstrap.
- Arm E was added after C turned out worse than B: **E = RTN(W_PEAR) on the checkpoint's scales**. It is the fair
  control for C, since it uses the same quantizer and only the RL update differs.
- My RTN reproduces 86–98% of checkpoint values. Its weight error equals or slightly undercuts the checkpoint's, which
  suggests the released checkpoint uses activation-aware rounding.

| Arm | Weights | dNLL thinking | dNLL action | Top-1 agreement (action) |
|---|---|---|---|---|
| A | PEAR BF16 | 0 | 0 | 99.85% |
| B | released NVFP4/FP8 weights (dequantized) | +0.0093 [0.0081, 0.0115] | +0.0027 [0.0013, 0.0042] | 99.75% |
| E | RTN(W_PEAR) | +0.0186 [0.0161, 0.0230] | +0.0042 [0.0027, 0.0053] | 99.63% |
| C | oracle: RTN(W_base) + exact Δ | +0.0423 [0.0351, 0.0536] | +0.0118 [0.0072, 0.0172] | 99.47% |
| D | RTN(W_base), base everywhere | +0.1607 [0.1352, 0.2020] | +0.0180 [0.0141, 0.0218] | 99.28% |

**Reading:**
1. **The mechanism is wrong.** The oracle that restores the exact RL update (C) is 2.3× *worse* than plainly
   quantizing PEAR (E). Per tensor, the two have identical weight-error norms (median ratio 1.00000) and identical
   clipping.
   - So the 0.09 cosine and the "erasure" measured at the weight level do not carry over to function.
   - Rounding PEAR's own weights preserves the skill better than adding back the exact update.
   - Plausibly, the fine-tuned weights sit where their own rounding is benign. This was not tested further.
2. **Weight quantization costs little.** The released weights cost 0.009 nats/token on thinking and 0.003 on actions
   (action top-1 agreement 99.75% vs. 99.85%).
   - This is already better than RTN, so headroom for any better rounding is at most about 0.009 nats/token.
   - Weights alone therefore cannot explain the NVFP4 accuracy drop the paper reports. If there is a drop, it would
     come from FP4 activations or kernels, which this pilot did not test.
3. The stopping rule in §5 ("NO-GO if C ≈ B") is met in a stronger form: C is worse than B.

**Pivot left open (not run):** test whether FP4 activations or the W4A4 lm_head cause the paper's NVFP4 drop. That
needs real NVFP4 kernels (B200) and would be a new direction, not this one.

## 8. Follow-up checks (2026-11-16): where the NVFP4 cost comes from, and whether old thinking can be dropped

**Run:**
- Same teacher-forced pipeline on one H200, about $5.
- A first Community pod had a faulty GPU (25 uncorrectable remapped rows) and was terminated after about 10 minutes.
- Activation fake quantization uses the released checkpoint's own recipe (`pod/actq.py`):
  - NVFP4 per-16 dynamic block scales with the checkpoint's static global `input_scale`, on the MLPs and lm_head;
  - static per-tensor FP8 on all attention and Gated DeltaNet projections.
- B0 re-scored plain B through the hook path and reproduced it exactly (+0.00932 both times).

| Variant (all on released weights B) | dNLL thinking | dNLL action | Top-1 agreement (action) |
|---|---|---|---|
| B, weights only | +0.0093 | +0.0027 | 99.75% |
| + lm_head FP4 activations (Blm) | +0.0108 | +0.0028 | 99.76% |
| + MLP FP4 activations (Bmlp) | +0.0273 | +0.0043 | 99.69% |
| + MLP and lm_head (Bmlp_lm) | +0.0287 | +0.0043 | 99.69% |
| Full deployment: + FP8 attention activations (Bfull) | **+0.0314** [0.028, 0.038] | **+0.0059** [0.004, 0.008] | 99.63% (A: 99.85%) |

**Reading:**
- The NVFP4 cost is about 70% from **FP4 MLP activations**, about 30% from weights, and little from the lm_head.
- It falls on **reasoning tokens** (about 13% higher NLL), not on actions.
- This fits the paper's report of a longer chain of thought, and it is the generic "quantized reasoners overthink"
  effect ([2606.00206](https://arxiv.org/abs/2606.00206), ReSET).
- Generic FP4-activation fixes (rotations, microscaling-aware GPTQ) apply directly. Nothing here is specific to robots.
- **Not promising as a novel direction.**

**Context-pruning check** (`pod/score_strip.py`, `results/pilot/strip_A.jsonl`).
- Setup:
  - BF16 PEAR, 67 rounds from 14 episodes.
  - Teacher-force each round's recorded tokens with and without the thinking of earlier rounds, rendered the way the
    template renders `preserve_thinking=false`.
  - Context shrinks from a median 14.8k to 6.7k tokens.
- Thinking tokens shift by +0.29 nats/token [0.24, 0.43]. This is partly inflated, because the recorded thinking
  refers back to earlier thinking.
- **Actions shift by +0.016 nats/token [0.012, 0.019].** That is 2.6× the whole NVFP4 deployment cost. Action top-1
  agreement drops from 99.80% to 99.22%, and 26 of 67 rounds have at least one action token flip. Rounds with more
  than 40k tokens of context show the same pattern.
- So PEAR does use its old thinking, and the trivial flag is not free. Context exhaustion is a real problem that needs
  a real method.
- The prior-art space is crowded:
  - MEM1 ([2506.15841](https://arxiv.org/abs/2506.15841)): learned constant memory, trained with RL.
  - Context Folding ([2510.11967](https://arxiv.org/abs/2510.11967)).
  - StateComp ([2609.27298](https://arxiv.org/abs/2609.27298)).
  - WorldLines/ObsMem ([2606.18847](https://arxiv.org/abs/2606.18847)): embodied, observation-grounded memory.
  - MeMento ([2608.01456](https://arxiv.org/abs/2608.01456)): embodied multimodal memory compression.
  - ECHO, AgentKV, GateSight.
- No robot-specific method with a clear novelty margin has been identified yet.

## 9. Delta-regularized GPTQ (pre-registered 2026-11-16 ~20:45 UTC, before data)

**Proposal (from the user).** RL changes PEAR through a few important directions, and NVFP4 rounding error, spread over
all weights, may erase them. GPTQ with
  H′ = XXᵀ + λ ΔᵀΔ,   Δ = W_PEAR − W_base,
penalizes rounding error along the input directions Δ acts on. Bit width and format are unchanged; only the Hessian
changes.

**Precondition check (CPU, `scripts/delta_rank.py`, `results/delta_rank.txt`).** Δ is partly low-rank:
- stable rank 7–61, against about 2,000 for the same entries shuffled;
- its top 16 directions hold 9–27% of ‖Δ‖², 15–40× more than chance;
- but 50% of ‖Δ‖² needs 256–682 directions and 90% needs about 3,000 of 5,120.

The NVFP4 error spectrum is flat, like the shuffled control. So ΔᵀΔ is far from isotropic, and the penalty is not just
damping.

**Test (`pod/gptq_delta.py`, `pod/build_arm.py --arm G`, one H200).**
- GPTQ runs on all 192 MLP matrices, on the released checkpoint's own NVFP4 scales. Every other tensor keeps the
  released values.
- Calibration: 343 rounds (about 389k tokens) of PEAR's own reasoning and actions from today's closed-loop rollouts,
  disjoint from the 14 scored episodes. Text only.
- λ = α·tr(H)/tr(ΔᵀΔ), with α ∈ {0 (plain GPTQ), 0.1, 1}.
- Scoring: teacher-forced dNLL vs. BF16 (A) on the 14 recorded episodes; the released checkpoint (B) is rebuilt and
  scored on the same pod.

**GO** needs both:
1. The best α > 0 arm beats G0 (plain GPTQ) on action-token dNLL, with an episode-bootstrap CI excluding 0.
2. Its thinking dNLL is no worse than G0's (point estimate).

**NO-GO** otherwise. If G0 alone beats B, that is a plain-GPTQ result, not support for the Δ penalty.

**Headroom, stated up front.** B costs +0.0027 nats/token on actions and +0.0093 on thinking (§7). Any weight method
recovers at most that.

**Result (2026-11-16 23:15 UTC; `results/q/`, `scripts/analyze_gptq.py`): NO-GO.**
- Arms run: A, G0 and G1 (α = 1). G0.1 and a rebuilt B were cut to save cost; G0.1's codes were computed but not
  scored.
- This run's A matches the pilot's A exactly, so the pilot's B and E are comparable.

| Arm (dNLL vs. BF16, nats/token) | Action | Thinking |
|---|---|---|
| B: released NVFP4 checkpoint | +0.0027 [0.0013, 0.0043] | **+0.0093** [0.0081, 0.0117] |
| E: RTN on the checkpoint's grid | +0.0042 [0.0026, 0.0053] | +0.0186 [0.0162, 0.0231] |
| G0: plain GPTQ on MLPs | **+0.0019** [0.0008, 0.0031] | +0.0130 [0.0115, 0.0161] |
| G1: GPTQ + ΔᵀΔ penalty (α = 1) | +0.0030 [0.0015, 0.0045] | +0.0121 [0.0105, 0.0151] |
| **G1 − G0 (paired)** | **+0.0010 [+0.0001, +0.0020]: worse** | −0.0010 [−0.0015, −0.0006]: better |

**Reading.**
- Criterion 1 fails in the wrong direction: protecting Δ's directions makes the **actions** measurably worse than plain
  GPTQ (+54% dNLL).
- Thinking improves slightly (−7%). This matches the matrix-level trade: α = 1 raised overall output error 25–37% to
  halve the error along Δ.
- Together with the pilot's oracle (restoring Δ exactly was 2.3× worse than RTN), this closes the "protect the RL
  update" line. PEAR's action fidelity tracks overall output error, not error along Δ.
- Side observations:
  - Plain GPTQ beats the released checkpoint on actions (0.0019 vs. 0.0027; CIs overlap) but is worse on thinking
    (0.0130 vs. 0.0093).
  - The released rounding looks activation-aware with broader calibration data, while ours used text-only robot
    reasoning.
  - Either way, all weight methods sit within about 0.003 nats/token of BF16 on actions.
