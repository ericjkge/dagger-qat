# Feedback-tolerant early exit for robot-use reasoning agents (2026-11-16)

**Status:** pre-registered before any data. Results are in §5.

## 1. Problem

PEAR reasons for 3.2k tokens before every move (4.3 s per round on a GB300; single rounds up to 32k tokens). Its
limitations section leaves "shortening the reasoning without giving up success" to future work. Simply lowering
effort costs most of the success (74.4% → 21.9% on LIBERO).

In our 14 recorded episodes:
- 35% of rounds (those over 2k tokens) hold 82% of the tokens;
- the most expensive rounds are long re-derivations of where objects are, with repeated "wait, let me double-check"
  passes.

## 2. Hypothesis and method

**Hypothesis.** In expensive rounds, the action PEAR would output settles to within what the closed loop can absorb
long before its thinking ends. The remaining thinking only polishes the action to a precision the next observation
makes unnecessary.

**Method: feedback-tolerant early exit.**
- Generate the thinking in segments.
- At paragraph boundaries, at least G tokens apart, fork the context, force `</think>` and greedily decode the
  provisional action.
- Stop thinking once two consecutive provisional actions agree within the **feedback tolerance**: translation within
  max(τ_abs, τ_rel·|move|), rotation within τ_rot, and the same gripper command class (open, hold, close).
- Then sample the action after the forced `</think>`, exactly as PEAR would.
- The model sees its own truncated reasoning in later rounds.

**Why the tolerance is the point, and not a knob.** Existing answer-convergence exits (Dynasor-CoT, Answer Convergence,
DEER) wait for a discrete answer to repeat. A continuous robot action jitters at the 0.5–1 cm level from paragraph to
paragraph, and even between two samples at full thinking. So exact agreement either never fires or fires only on
trivial rounds. A robot agent, unlike a math solver, gets the world's response before its next decision: a residual
error inside the tolerance is visible in the next frame and is corrected by a cheap round. Measured corrective rounds
after "reached" outcomes cost 100–500 tokens.

## 3. Pre-registered checks and GO criteria

**E1, offline (recorded rounds, forced-exit probes every ≥150 tokens).** GO-eligible only if both hold:
- under τ = (1 cm, 15%, 0.1 rad), at least 30% of thinking tokens come after the provisional action has converged
  (oracle convergence);
- the replayed tolerant rule saves at least 1.5× the tokens of the exact-agreement rule.

**E2, closed loop.**
- Design:
  - LIBERO spatial, object, goal and long, tasks 0–9, trial 0: 40 paired episodes per arm.
  - Same harness and segmentation for all arms.
  - Arms: `full` (no exit; probes logged only), `exact` (Dynasor-style), `tol` (ours).
  - Token budget 160k, at most 30 rounds.
- GO requires all of:
  - `tol` uses at most 0.70× the tokens of `full`, counting probe decode tokens;
  - `tol`'s success is no more than 10 points below `full` (point estimate), with the paired CI reported;
  - `tol` saves more tokens than `exact`, or matches its savings at higher success.

**NO-GO** if E1 fails, or if `tol` loses more than 10 points of success, or saves less than 30% of tokens.

## 4. Result of E2 for early exit (pod 1, spatial and object, paired): NO-GO

| Arm vs `full` | Paired episodes | Success | Tokens incl. probes | Rounds |
|---|---|---|---|---|
| `exact` | 12 | 11/12 vs 12/12 | 0.95× | 0.97× |
| `tol` | 10 | 6/10 vs 10/10 | 1.15× | 1.38× |

- **Why it fails.** PEAR's action converges late: only about 15% of tokens come after the provisional action stays
  within 1 cm of the final action (on-policy probes).
- **What truncation costs.** Truncating trades thinking for imprecise moves. The closed loop pays them back with
  correction rounds, which exhaust the LIBERO step budget: all 4 `tol` failures are step-limit failures after 16–21
  rounds.
- **Lesson.** Per-round precision, not per-round tokens, sets task cost. Both exit arms were stopped.

## 5. Second candidate (pre-registered 2026-11-16 ~10:50 UTC, before closed-loop data): consensus control

**E3 measurements** (46 recorded rounds, 4 full-thinking samples each, `results/sample_k.jsonl`):
- in about half of rounds, samples disagree by more than 3 cm, and a typical sample sits about 6 cm from the
  consensus of the others;
- pairwise distance: p75 5.1 cm, p90 9.4 cm;
- disagreements are mostly magnitude along an agreed direction, plus occasional decision outliers (descend 25 cm while
  the others adjust; lift while the others descend);
- within a round, the shortest chain is as accurate as the longest (2.5 vs. 2.6 cm from the leave-one-out consensus);
- the fastest of 4 chains uses 40% fewer tokens than the average chain.

**Method (`consensus`).**
- Run k = 4 chains in parallel. A robot is a batch-1 deployment and decode is memory-bound, so this is almost free in
  latency.
- When m = 3 have finished, cancel the rest.
- Execute the medoid of the finished actions within the majority gripper-command class, and keep that chain's
  reasoning in the context.

**Pre-registered GO** (paired episodes vs. `full`):
- success no lower than `full` − 5 points (point estimate);
- critical-path decode tokens per episode at most 0.80× `full` (latency proxy: the m-th finisher's length per round);
- rounds at most 0.95× `full`;
- compute (all chains) is reported, not gated.

**NO-GO** if success drops by more than 5 points, or the latency ratio exceeds 0.9.

**Result (6 paired spatial episodes): NO-GO.** Success was 6/6 in both arms and rounds 59 vs. 60. Critical-path latency
was 1.24× `full` (waiting for the 3rd of 4 chains is slower than one chain), and compute was 3.3×. Stopped.

**Restart schedules (offline, `scripts/restart_offline.py`): NO-GO.**
- Each context's chain-length distribution is not heavy-tailed (median max/min 2.0 over 5 samples).
- Even an oracle per-context cutoff saves 3%.
- Luby and fixed cutoffs cost 1.3–40× more, because length scales vary across rounds by over 10×.

## 6. Third candidate (pre-registered ~11:55 UTC, before data): irreversibility-gated parallel reasoning

**Principle.** Spend latency on precision only where an error cannot be undone by the next round.
- Run k = 4 chains in parallel (batch 1, almost free).
- Act on the **first finisher**, unless its action is irreversible. Irreversible means a gripper command, or a descent
  toward contact (z ≤ −3 cm with less than 3 cm lateral motion).
- For irreversible actions, wait for m = 3 finishers and execute the medoid.

**Arms** (LIBERO-Long, tasks 0–9, trial 0, where failures occur):
- `full`: single chain;
- `first`: k = 4, m = 1;
- `gated`: k = 4, m = 3, gate on.
- `first` also runs on spatial and object.

**GO**, both required:
- `gated` success ≥ `full` success (point estimate) and ≥ `first` success;
- `gated` critical-path latency ≤ 0.80× `full`.

**Partial (latency-only) result** if `first` alone matches `full` success at ≤ 0.70× latency. That is short-1@k
transferred to robots, which is not novel by itself.

**Amendments (13:10 UTC, before any trial-1 data).**
- The baseline on LIBERO-Long is `single` (k = 1, unsegmented). Segmented `full` overloaded SGLang's tokenizer.
- Trial 1 of each LIBERO-Long task is added to every arm, for power.
- Early read on trial 0: `gated` 4/4 successes; `first` 4/6, both failures at the 30-round cap; `single` 2/2.

**Interim at 13:31 UTC (LIBERO-Long, 7 paired episodes vs. `single`).**
- `gated` succeeded 7/7 and `single` 5/7, at 0.42× critical-path latency (CI 0.33–0.61); compute was 1.6×.
- `first` matched `single` (4/6 vs. 4/6) at 0.60× latency.
- `single` and `first` failures all hit the 30-round cap.
- The gate fired on 37% of rounds.

**Added control, pre-registered before its data: `randgate`.** It waits for the 3-of-4 medoid on a random 37% of
rounds instead of the irreversible ones. If `randgate` ≈ `gated` in success, the benefit comes from occasional
consensus, not from irreversibility, and the principle claim fails (only the latency result remains).

**Interim at 15:30 UTC (LIBERO-Long, 12 paired episodes vs. `single`).**

`gated` meets the pre-registered criteria:
- success 10/12 vs. 7/12;
- 0.50× critical-path latency (CI 0.41–0.59); compute 1.75×;
- it also beats `first` (4/9).

The mechanism checks do not support the irreversibility principle:
- **Consensus rarely changes the action.** On the 187 of 412 `gated` rounds that waited for 3 chains, the executed
  medoid equals the first finisher's action at the median. It differs by more than 3 cm in 9% of those rounds and
  changes the gripper command in 4 of them. `randgate` is about the same (11% over 3 cm).
- **The success gap is mostly a parse artifact.** 2 of `single`'s 5 failures are `parse` ends (no valid action from
  its one chain). Any k = 4 arm avoids these automatically. Leaving those pairs out, `single` is 7/10 and `gated` 8/10.
- **`randgate` is 3/3** at 0.54× latency so far.

**Reading.** What survives is that a 4-chain first finisher roughly halves critical-path latency with no detectable
success loss. That is short-1@k ([2505.17813](https://arxiv.org/abs/2505.17813)) moved to a robot loop, with a valid
action fallback. The principle claim is pending `randgate`, which is likely to fail.

## 7. Final result (17:05 UTC): no validated GO

LIBERO-Long; `single` = one chain (k = 1); others k = 4.

| Arm | Episodes | Success | Critical-path latency vs. `single` (paired) | Compute |
|---|---|---|---|---|
| `single` | 15 | 7/15 (2 of 8 failures are `parse`) | 1.00 | 1.0× |
| `gated` | 17 | 11/17; paired 10/15 | **0.54 [0.43, 0.66]** | 1.9× |
| `randgate` | 10 | 5/10; paired 4/10 vs. 5/10 | 0.64 [0.54, 0.77] | 2.4× |
| `first` | 9 | 4/9; paired 4/7 vs. 4/7 | 0.58 [0.31, 1.08] | 2.3× |

On trial 0, all three arms ran every task:
- `gated` 7/10, `randgate` 5/10, `single` 4/10;
- `gated` vs. `randgate` discordant pairs: 2–0 (exact p = 0.5).

**Verdict.**
- `gated` formally meets its pre-registered GO: success ≥ `single` and ≥ `first`, latency ≤ 0.80×.
- The principle claim is not validated:
  - the random-gate control is not separable at this n;
  - when the gate fires, the 3-of-4 medoid equals the first finisher's action at the median (over 3 cm in 9% of
    those rounds; gripper command changed in 4 of 187);
  - part of the success gap over `single` is the parse failures that any k > 1 arm avoids.
- The robust finding is latency: four parallel chains with first-valid-finisher control roughly halve critical-path
  decode tokens at no detectable success cost. That is short-1@k plus a validity fallback, which is not a novel
  method.
- Separating `gated` from `randgate` would take about 60 or more paired episodes per arm (8 to 10 hours of 2×H200).
  That was outside this session's budget.

**Wall clock** (`pod/batch_latency.py`, idle H200, 1024-token decode):
- k = 4 costs 1.23× (short context) and 1.39× (long context) per decoded token, compared with k = 1.
- So `gated`'s 0.54× critical-path tokens is about **0.67–0.75× wall-clock latency**.

## 8. Correction (post-hoc): our 30-round cap drove the LIBERO-Long failures

PEAR's own evaluation has no round cap. It ends an episode on success, on the suite step limit (520 for LIBERO-Long),
on the 256k context budget, or on an invalid action (paper, App. A).

Our harness added `--max-rounds 30`. Of the 19 LIBERO-Long failures logged in `single`, `gated` and `randgate`:
- 15 ended at that cap;
- most with 46–248 of the 520 steps still unused (failed episodes used 272–520 steps; successes 209–318).

So the success comparisons in §6–7 measure "success within 30 rounds", which is not PEAR's metric. They are biased
toward whichever arm needs fewer rounds. The latency-token ratios do not depend on the cap.

The controller's `stopped` outcome (blocked or stalled; 61% of rounds in failures vs. 26% in successes) mostly
reflects grasps and contact, so it is not a clean lever.

## 9. CPU-only screen for a new angle (2026-11-16 evening; recorded logs only, $0)

Corpus: 121–163 deduplicated episodes, about 1.3–2.1k rounds from all arms. Scripts: `scripts/round_structure.py`,
`undershoot.py`, `gain_counterfactual.py`, `redundancy.py`.

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Systematic action gain bias: PEAR undershoots, so calibrate gain online | After reached lateral moves, follow-ups go the same way 84% of the time (median 0.48× the move, stable across axes and sizes). But 34% of moves are accepted as is, and the ratio is bimodal. The best online gain still needs a follow-up on 74% of moves vs. 65% unscaled; an oracle per-episode gain gives 64%. | NO-GO: coarse-to-fine servoing on new wrist views, not a bias |
| Cross-round reasoning redundancy (reuse earlier derivations) | 1.9% of thinking words are in sentences repeated from earlier rounds | NO-GO |
| Missed grasps waste rounds | 34 in the whole corpus; PEAR noticed 31 in the very next round | NO-GO |
| Unexplained `stopped` outcomes trigger long diagnosis | Next-round mean tokens: 2010 after a stopped arm move vs. 1008 after a reached one (medians 576 vs. 472, so a heavy tail). The thinking diagnoses: "hit the book's top?", "joint limit or collision?" | Real effect, but the fix is richer textual feedback, as in Inner Monologue (2022) and REFLECT (2023). Not novel enough. |
| What old context is used for (context exhaustion) | 42% of later rounds refer back. Topics: error/feedback (602 sentences), position memory (352), self-calibration (137), task progress (25) | Informative for a future memory method; no method yet |

## 10. Grounded-claim retention (pre-registered 2026-11-16 evening, before data)

**Problem.** Context exhaustion (68.5% of RoboDojo episodes in the paper). Dropping old thinking shifts actions by
+0.016 nats/token (§8 of pear_rl_delta_quant). Old reasoning is referenced mainly for:
- feedback on earlier moves;
- remembered object positions;
- PEAR's own pixel-to-cm estimates (§9).

**Method.** Keep only the *grounded-claim* sentences of earlier rounds' thinking: metric quantities, outcomes of
earlier moves, and object positions or occlusion. Drop the rest. This is training-free and needs no summarizer call.
On the 14 recorded episodes it keeps 23% of old-thinking characters.

**Test (offline, teacher-forced, `pod/score_keep.py`; one H200, cap $15).** For every third round r ≥ 2, score the
recorded round under five conditions:
- `full`;
- `strip`;
- `grounded`;
- `random`: the same characters, as random whole sentences;
- `last`: only the previous round's thinking.

**GO** for a closed-loop test needs all of:
1. Gap recovery: (strip − grounded) / (strip − full) ≥ 0.6 on action-token NLL, with the bootstrap lower bound
   above 0.3.
2. `grounded` beats `random` on action NLL, paired over rounds, with a CI excluding 0.
3. `grounded` is no worse than `last` (reported; `last` keeps less).

**NO-GO** if (1) or (2) fails.

**Novelty risk, stated up front.** ObsMem ([2606.18847](https://arxiv.org/pdf/2606.18847)) already uses
provenance-gated typed memory, for household QA and planning. The margin is retention selection inside a
continuous-control reasoning trace.

**Result (20:45 UTC, 61 rounds from 14 episodes; `results/keep/keep.jsonl`, `scripts/analyze_keep.py`): NO-GO.**

| Condition | Action NLL | Thinking NLL | Action top-1 | Prompt tokens |
|---|---|---|---|---|
| full | 0.0027 | 0.178 | 99.89% | 31.8k |
| strip | 0.0176 | 0.505 | 99.39% | 9.2k |
| grounded | 0.0119 | 0.355 | 99.56% | 15.6k |
| random (same characters) | 0.0106 | 0.342 | 99.61% | 14.8k |
| last round only | 0.0122 | 0.400 | 99.48% | 11.1k |

- `grounded` recovers 0.38 [0.09, 0.64] of the strip → full action-NLL gap, below the 0.6 bar.
- It does **not** beat random sentences of the same size (random − grounded = −0.0013 [−0.0048, +0.0019]).
- What old reasoning PEAR needs is spread across its text and not captured by the grounded-claim selector. At a matched
  budget, what is kept matters less than how much is kept.
