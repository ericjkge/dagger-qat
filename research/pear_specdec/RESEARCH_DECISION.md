# Research decision: structure-aware speculative decoding for PEAR (2026-11-15)

**Verdict: NO-GO on all three directions.**

Measured on 14 LIBERO episodes (365k generated tokens, 110,876 instrumented DFlash2 verify steps):
- **What it rules out:** PEAR's robot-reasoning structure leaves no headroom over DFlash2 large enough to carry a new,
  principled algorithm.
- **Best-case oracle upper bounds** over DFlash2's 3.34 tokens per step:
  - A (phase-aware): ≤ 2.9% via the action phase, ≤ 3.5% via adaptive draft length.
  - B (cross-round reuse): ≤ 1.5% temporal reuse, ≤ 4.4% for all retrieval.
  - C (quantization-aware drafting): not measured. It is published and needs Blackwell hardware.
- None of these bounds survives a realistic router, and each piece is already published.

Supporting documents:
- `LITERATURE_REVIEW.md`: prior work and novelty matrix.
- `BASELINE_REPORT.md`: baselines, latency and exactness.
- `STRUCTURE_PROFILING.md`: Experiments A, B and C.
- `PHASE0_AUDIT.md`: feasibility.

All numbers come from `results/` and were produced by the scripts in `scripts/`.

## Scorecard (decision criteria from the task)

| Criterion | A: phase-aware SD | B: cross-round reuse | C: quantization-aware drafting |
|---|---|---|---|
| 1. Distinct computational mechanism? | Partly. The answer phase is easy (6.2 vs. 3.25 tokens per step) and numeric reasoning is easier than prose (3.68 vs. 3.16), but both are per-step difficulty signals, not a new mechanism | **No.** The previous round predicts no better than an unrelated episode's round (+1.46% vs. +0.65% oracle gain) | Not measured |
| 2. Novel vs. the literature? | No: WhiFlash (entropy routing between drafter paradigms), SpecDec++, AdaEDL, BanditSpec, TraceSpec | No: SuffixDecoding, Graft (retrieval + DFlash), prompt lookup, SGLang NGRAM | No: QuantSpec, QSpec, ML-SpecQD |
| 3. Room for a principled algorithm? | Bounded by flat verify cost: 20.8 ms at K = 7 vs. 20.1 ms at K = 3 | An oracle hybrid bound exists, but there is no signal to learn the router from | — |
| 4. Expected speedup over DFlash2 (decode) | ≤ 2.9% (Amdahl, action phase) and ≤ 3.5% (adaptive K), both oracle | ≤ 1.5% (temporal) and ≤ 4.4% (all retrieval), both oracle | Unknown |
| 5. Implementation complexity | Medium (SGLang worker) | Medium | High (NVFP4 kernels) |
| 6. GPU requirements | 1× H200 sufficed | 1× H200 sufficed | B300 / GB300 |
| 7. Feasible in a course project? | Yes, but there is nothing worth building | Same | Doubtful |
| **Verdict** | **NO-GO** | **NO-GO** | **NO-GO** (deprioritized, unmeasured) |

## A. What we learned
1. **Baseline reproduced.**
   - DFlash2 on PEAR accepts 3.25 tokens per verify step on reasoning (paper: 3.2) and 3.34 on the replay set.
   - It decodes at 162 tok/s vs. 64 tok/s AR on H200/SGLang, a 2.53× (2.44–2.61) decode speedup.
   - The derived end-to-end round speedup is ≈ 2.09×.
2. **Reasoning is the whole decode workload.** It is 97.1% of generated tokens. The answer (text + action JSON) is
   2.9% and already gets 6.16 tokens per step, with 55% of its steps accepting the full block.
3. **Verification is flat in draft length at batch size 1.** A verify step costs 20.8 ms at K = 7 and 20.1 ms at K = 3,
   against 15.7 ms per AR token. Draft-length adaptation therefore has < 3.5% to gain.
4. **Cross-round temporal structure is absent at the token level.**
   - Previous-round retrieval alone gives 1.24 tokens per step, vs. 1.09 for the shuffled-episode control and 1.55 for
     in-context lookup.
   - Even as an oracle switch with DFlash2 it adds +1.46%, vs. +0.65% for the control.
   - Robot "re-planning" text is not verbatim reuse: rounds paraphrase rather than copy.
5. **Drafter acceptance degrades with context length:** 3.52 tokens per step at ≤ 8k context, 3.04 at 32–64k (−14%).
   The drafter's attention covers only a 2,048-token window.
6. **Long episodes are where the time goes, and where things break.**
   - The goal / libero_10 episodes ran to 26–184k tokens.
   - One round degenerated into 45k tokens of `~?~?` at about 180k context. Its cause is undetermined.
7. **The simulator and an uncached prefill each take about 14% of round wall time.** This is a stack artifact, because
   the prefix cache did not hit. It further dilutes any decode-only gain.

## B. What is already known (and why it invalidates each idea)
| Idea | Closest prior work | Invalidates? |
|---|---|---|
| Different drafting for the action vs. reasoning | ToolSpec, grammar/FSM drafting; WhiFlash routes AR vs. parallel drafters by entropy for "reasoning vs. structured" output | Yes, on novelty. And Amdahl caps the gain at 2.9% |
| Sub-phase-aware draft length | SpecDec++ (MDP stopping rule), AdaEDL (entropy early stop), BanditSpec | Yes. And at batch size 1 the cost curve makes it ≤ 3.5% |
| Reusing earlier rounds' tokens | SuffixDecoding (agentic traces), prompt lookup, SGLang NGRAM, Graft (retrieval grafted into DFlash-style blocks) | Yes, for verbatim reuse. Non-verbatim reuse showed no measurable structure |
| Long-context drafter degradation | DLite (draft models lose up to 80% acceptance in long multi-turn agent contexts; history-free drafter) | Yes, the effect is published, and the remedy is drafter (re)training |
| Quantized drafting / self-speculation | QuantSpec, QSpec, ML-SpecQD | Yes, on novelty |
| Robot-specific speculative decoding | Spec-VLA, KERV, WA-SpecDec, Zora (all on action-token VLAs, not reasoning LLMs) | Different setting, but they show "robot-specific SD" is itself not new |

## C. What is genuinely new
**Nothing we can honestly claim as a new algorithm.** The only new *empirical* facts:
- **Paraphrase, not copying.** PEAR's per-round reasoning paraphrases rather than copies across rounds, so cross-round
  reuse has no token-level signal (prev ≈ shuffled control).
- **Phase share.** The action phase is 2.9% of decode in a reasoning robot LLM, so action-specialized SD (the focus of
  Spec-VLA, KERV and the rest) is irrelevant for this model class.
- **Acceptance vs. context.** DFlash2's acceptance on PEAR falls 14% with context length.

These are characterization results. By the course's standard ("analysis-type project"), they are not a contribution.

## D. Why it should work: cost model (and why it does not here)
Per-round decode time with block drafting:

  T(K) = (N / τ(K)) · t_step(K),  t_step(K) = t_draft + t_verify(K + 1),  speedup vs. AR = τ(K) · t_AR / t_step(K)

- **Model vs. measurement:** with the measured t_AR = 15.7 ms, t_step(7) = 20.8 ms and τ(7) = 3.34, the model predicts
  2.52×. Measured: 2.53×.
- **What a new method can move:**
  - **τ**, through a better drafter or a hybrid. A per-step oracle that picks the better of DFlash2 and the best
    retrieval corpus gives τ_oracle/τ − 1 ≤ 4.35% on reasoning. A learned router achieves strictly less, and adds
    overhead of order 0.05–0.5 ms per step (2.4% of a 20.8 ms step per 0.5 ms).
  - **t_step(K)**: t_step(3)/t_step(7) = 0.965, so adaptive K can save at most 3.5%.
  - **Which tokens are slow.** Amdahl over phases: making the answer free gives 1/(1 − 0.029) = 1.029×.
- **Conclusion:** the combined oracle ceiling is about 5–8% decode, ≈ 4–6% end to end. That is below what would justify
  a new algorithm, and every component of it is a published technique.

## E. What could invalidate this NO-GO
The NO-GO would be overturned by any of:
1. **Longer episodes than ours.**
   - **Observation:** the paper's evaluation averages 94k tokens per episode, against our 29k mean.
   - **Hypothesis:** cross-round reuse might grow there.
   - **Falsification test:** prev-vs-shuffled oracle gap > 3 points on episodes ≥ 80k tokens.
2. **A different cost curve on Blackwell.**
   - **Observation:** PEAR's megakernels (B300) might make t_verify(K) steep in K.
   - **Hypothesis:** adaptive or phase-aware K would then matter.
   - **Falsification test:** t_step(3)/t_step(7) < 0.8 on B300.
3. **xhigh vs. low effort.** The low-effort setting (0.86k thinking tokens per round in the paper) raises the action
   share. A new verdict would need an answer share > 15%.
4. **Our serving stack under-measures DFlash2.** Ruled out for acceptance (it matches the paper). Not ruled out for
   absolute latency.
5. **Sampling variance.** CIs are tight on step-level statistics (n = 110k steps). Episode-level claims rest on
   14 episodes from LIBERO only, not the paper's other environments.

## F. What HT Kung is likely to question
- **"Is this just an analysis project?"** Yes, and we say so. We tested whether a structural opportunity exists before
  building anything, and it does not. We recommend not proceeding with A, B or C rather than shipping a bag of tricks.
- **"Did you compare against DFlash2, not AR?"** Yes. Every opportunity is expressed as an oracle upper bound *over
  measured DFlash2 steps* (110,876 steps), and the paired benchmark puts DFlash2 at 2.53× over AR.
- **"Is your retrieval estimate rigorous?"**
  - For a deterministic drafter under lossless verification, P(m drafts accepted) = Π p(dⱼ | prefix) equals the
    probability of the realized sample matching for m tokens. Replaying along recorded samples is therefore an unbiased
    estimator.
  - It is unit-tested against Monte-Carlo rejection sampling.
  - The oracle switch is an upper bound for any router.
- **"Systems overhead?"** We measured the verify-cost curve directly (20.8 vs. 20.1 ms) instead of assuming it.
- **"Is the baseline lossless?"** Greedy divergences all happen at bf16 ties: top-2 margin 0.00 in 4 of 5 cases and
  0.25 nats in 1. The rejection-sampling algorithm is unit-tested.
- **"Your hardware isn't PEAR's."** Agreed, for absolute latency. Acceptance is kernel-independent, and the conclusions
  rest on acceptance plus the shape of the cost curve. Item E2 is the open hardware question.

## G. Next experiments (only if pursuing this line further; ranked by information per GPU-hour)
1. **(≈ 1 B300-hour, highest value) Verify-cost curve on PEAR's own megakernels.** If t_step(K) is steep on Blackwell,
   an *adaptive-K* policy for block drafters becomes worth studying. That would still have to beat SpecDec++/AdaEDL to
   be new. This is the only measurement that could reopen A.
2. **(≈ 3 H200-hours) 10 more long-horizon episodes** (libero_goal / libero_10, all tasks). These would pin down the
   context-degradation curve beyond 64k and retest the cross-round gap on ≥ 80k-token episodes (E1).
3. **(≈ 1 H200-hour) Root-cause the degenerate long-context round.** Determine whether it is model behavior or this
   stack's numerics (replay the 180k context under vLLM). This matters to anyone deploying PEAR, not for novelty.
4. **Not recommended:** building a DFlash2 + retrieval hybrid or a phase router. The ≤ 4.4% oracle ceiling and the
   published prior art (Graft, WhiFlash) mean it would be a bag of tricks.

**Recommendation.** Stop this direction. Robot-reasoning LLMs spend 97% of decode on free-form reasoning that is
statistically ordinary text for a drafter. A novel *robotics-specific* speculative-decoding contribution would need a
different lever than token-level structure. One candidate is changing *what is generated*, i.e. how much reasoning a
round needs, which is no longer lossless decoding and falls outside this task's constraints.

## Budget and provenance
- **GPU:** 1× H200 community pod `d74dh65zxcfpzs`, 2.46 h at $3.59/h ≈ **$8.80** (Runpod billing read so far: $5.56,
  which lags). The pod is **stopped**: GPU released. Its 200 GB volume disk still bills at the stopped-disk rate until the pod is terminated. All results are synced locally, so terminating loses only re-downloadable checkpoints and environments.
- **Seeds:** server `--random-seed 0`, LIBERO seed 7 (PEAR's), trial 0, shuffled-control RNG seed 0, bootstrap seed 0.
- **Exact versions and flags:** `BASELINE_REPORT.md` and `configs/pod/`.
