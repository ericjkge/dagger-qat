# Context exhaustion in robot-use reasoning agents: focused ideation (2026-11-16)

**Status:** one candidate method, with a mechanism grounded in data. It is not yet functionally verified.
A decisive teacher-forced check costs about $3 and needs approval before renting a GPU.

## 1. The problem is real and not solved by the trivial fix

From `../pear_rl_delta_quant/DIRECTION.md` §8:
- The paper reports that the 256k context budget ends 19% of LIBERO-Goal, 33% of LIBERO-Long and 68.5% of RoboDojo
  episodes.
- Thinking is 78.5% of what each round adds to the context.
- Dropping earlier rounds' thinking (the `preserve_thinking=false` template path) shifts PEAR's next actions by
  **+0.016 nats/token**. That is 2.6× the whole NVFP4 deployment cost.
- 26 of 67 rounds then flip at least one action token, mostly magnitudes and signs of displacements.

## 2. Mechanism: the history is mostly in-context geometric calibration

Local analyses, no GPU (`scripts/`, `results/`):

**What becomes surprising when old thinking is removed** (`strip_attribution.py`):
- The thinking tokens that lose most are camera-to-robot frame and scale reasoning: "up in wrist image = +x",
  "left = -y", "x offset = -dv·z/f", "previously up = +x".
- Action tokens that flip are displacement digits and signs.

**How much reasoning is frame/scale derivation** (`frame_reasoning.py`, 171 rounds, 400k thinking tokens). By a
paragraph classifier (spot-checked; a mild undercount):
- 34.9% of all thinking.
- 60% of round 0.
- 31% of later rounds. 82% of later rounds re-derive frame or scale at least once.
- Frame-heavy rounds have 45% more hedges per token ("wait", "hmm", "re-verify": 14.2 vs. 9.8 per 1k tokens).
- The 182k-token runaway episode (libero_goal_t0) is dominated by repeated frame re-derivation.

**What the prompt already gives.** `environment_prompt.md` states the axis conventions for agentview, and for the
wrist camera at yaw 0. It does not give:
- **metric scale** (px per cm, which depends on depth for the wrist camera);
- how the wrist mapping changes after the gripper rotates.

So the agent does in-context **system identification**: from its own commands and the images that follow, it works
out how a command moves things in the image. This information is real, episode-specific, and needed for actions. It
is stored as tens of thousands of tokens of hedged free text, which is exactly what exhausts the context.

## 3. Candidate method: estimator-state memory ("servo memory")

**Principle.** The part of the history that matters for control is a sufficient statistic of the closed-loop geometry,
namely the image Jacobian J. J maps a commanded end-effector change to a pixel change, per camera.
- Classical uncalibrated visual servoing estimates J online from (Δcommand, Δpixel) pairs, for example with Broyden
  rank-1 updates plus a covariance.
- Replace the free-text history with that estimator's state, and drop old thinking:
  - J per camera, with uncertainty;
  - the current task-progress line;
  - the action/state ledger (already in the context).
- Context then grows by O(1) per round instead of O(thinking).

**Inputs.**
- Δcommand is known exactly: actions and robot states are already in every turn.
- Δpixel needs one point per camera (gripper or target), from one of:
  - (a) PEAR's own stated pixel coordinates, which it already writes in its thinking;
  - (b) a light keypoint tracker.
- The estimator is training-free and runs in the harness.

**Why this is a method and not a bag of tricks.** One principle (history → physically grounded sufficient statistic)
produces both the compression and its correctness argument. It exploits two things only robot agents have:
- known, exact action and state logs;
- a fixed observation-to-action geometry that is identifiable from the closed loop.

**Closest prior work and the claimed gap:**

| Work | What it does | Gap |
|---|---|---|
| ABBEL ([2512.20111](https://arxiv.org/abs/2512.20111)) | RL-trained natural-language belief state as a sufficient statistic | Generic, learned, unstructured; no physics; needs RL training |
| MEM1, Context Folding ([2510.11967](https://arxiv.org/abs/2510.11967)), StateComp ([2609.27298](https://arxiv.org/abs/2609.27298)), ECHO | Learned or heuristic summarization and selective memory | Generic; does not know which content is geometric |
| ObsMem/WorldLines ([2606.18847](https://arxiv.org/abs/2606.18847)), MeMento ([2608.01456](https://arxiv.org/abs/2608.01456)), MemCtrl ([2601.20831](https://arxiv.org/abs/2601.20831)) | Embodied memory compression | Object and world state, not the action-to-image calibration |
| AxisGuide ([2606.06761](https://arxiv.org/abs/2606.06761)), CamVLA ([2607.05396](https://arxiv.org/abs/2607.05396)), robot pointmaps ([2607.11498](https://arxiv.org/abs/2607.11498)) | Ground the robot frame in images for VLA policies | Need calibrated cameras or training; not about agent context |
| Uncalibrated visual servoing (classical) | Online image-Jacobian estimation | Never used as the memory of an LLM agent |

**Risks:**
1. Reviewers may read it as "give the LLM a calibration tool" (Code-as-Policies and VoxPoser use calibrated
   perception). The defense is that it is uncalibrated and online, and is justified and evaluated as context
   compression.
2. Pixel measurements: either PEAR's own coordinates, which are noisy, or an added tracker.
3. Part of the old thinking may carry non-geometric content: task progress, failed attempts, object identity. The
   oracle check below bounds this.
4. Real robots: does the same share hold on YAM or RoboDojo? Unknown.

## 4. Decisive cheap check (teacher-forced, about $3, needs approval)

Same 67 rounds and pipeline as `strip_A.jsonl`. Score round r's recorded tokens under:
- S0: full history. Already scored.
- S1: thinking stripped. Already scored: action +0.016.
- S2: stripped + an **oracle calibration card** derived from the simulator: px-per-cm per camera at the current
  depth, and the wrist mapping at the current yaw. This is an upper bound for servo memory.
- S3: stripped + **only the geometry paragraphs** of old thinking (classifier-tagged; about 31–35% of the tokens). This
  tests whether the geometric content is what matters.
- S4: stripped + a **generic summary** of old thinking of the same length as S3 (the StateComp/ABBEL-style baseline).

**Stopping rules:**
- **GO:** S2 or S3 recovers at least 50% of the S1 action shift while keeping at most 35% of old-thinking tokens, and
  beats S4.
- **NO-GO:** neither recovers the shift. The needed content is not geometric.
- **PIVOT:** S4 ≈ S3. Generic summarization suffices, and the area is already published.

## 5. Check result (2026-11-16): NO-GO for servo memory

**Run.**
- One Secure H200, about 2.1 h, roughly $11. About half of that went to my own pipeline bugs:
  - replay crashes;
  - a wrist sign error, caught by the independent check before scoring;
  - a missing completion marker that blocked scoring for 30 minutes.
- Data: 67 rounds from 14 episodes, BF16 PEAR, teacher-forced. Results: `results/pilot/`, `results/ctx_summary.txt`.
- Oracle cards come from the simulator's camera model:
  - state units are confirmed as cm (99.6–104.4 per metre);
  - the image-axis map is chosen only from the prompt's agentview convention (167/171 cards);
  - it is independently confirmed on the wrist camera (159/159 at yaw ≈ 0);
  - example card in `results/ctx_summary.txt`.

"Recovered" is the share of the stripped-history shift that a condition removes. Old thinking kept is the share of
old-thinking tokens left in the context.

| History condition | Action dNLL vs full | Recovered (action) [95% CI] | Recovered (thinking) | Mean context | Old thinking kept |
|---|---|---|---|---|---|
| S0 full | 0 | — | — | 29.7k | 100% |
| S1 stripped | +0.0156 | 0 | 0 | 8.8k | 0% |
| S2 stripped + oracle calibration card | +0.0151 | **+0.03** [−0.08, 0.14] | −0.17 | 9.0k | 0% |
| S3 geometry paragraphs only | +0.0113 | +0.27 [−0.01, 0.53] | +0.61 | 17.2k | 40% |
| S4 PEAR's own summary (≤ S3 budget) | +0.0112 | +0.28 [0.14, 0.44] | +0.11 | 11.9k | 15% |
| S5 geometry + card | +0.0110 | +0.29 [0.03, 0.55] | +0.53 | 17.3k | 40% |

**Reading:**
1. **Servo memory fails at its own upper bound.** Exact simulator calibration recovers 3% of the action shift (CI
   includes 0). The action-relevant content of old thinking is not the action-to-image geometry.
2. Geometry paragraphs recover 61% of the *thinking* shift but only 27% of the *action* shift. The thinking recovery
   is mostly verbatim reuse of earlier phrasing: the recorded thinking copies its own earlier derivations. That is why
   the §2 surprisal analysis pointed at geometry. It was the wrong functional signal.
3. **A generic summary at less than half the geometry budget matches geometry retention on actions** (0.28 vs. 0.27).
   This is the PIVOT outcome in §4: what matters for actions is diffuse and generic, and that is published territory
   (MEM1, Context Folding, StateComp, ABBEL).
4. Nothing reaches the 50% GO bar. The pre-registered verdict is **NO-GO** for servo memory, and generic summarization
   is published.

**Lesson.** Token surprisal under ablation is dominated by copy effects in the thinking. Score actions, not thinking,
when attributing what history is for.
