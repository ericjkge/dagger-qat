# Robot-use LLM efficiency ideas from the team: assessment (2026-11-15)

**Standard.** An idea qualifies only if it has all three of:
- an existing, unsolved problem;
- a novel methodology that can be proven, not analysis and not a bag of tricks;
- a feasible path.

**Evidence.**
- Local measurements come from the 14 recorded PEAR + LIBERO episodes (171 rounds) and the replay benchmark, which has
  9 independent samples for each of 12 contexts.
- Script: `scripts/idea_checks.py`. Output: `results/idea_checks.txt`.
- No GPU was used.

| Idea | Verdict |
|---|---|
| 1. Phase-aware adaptive reasoning effort | Not promising |
| 2. Quantization of robot-use LLMs | **Promising, only as the RL-delta direction** (`../pear_rl_delta_quant/DIRECTION.md`) |
| 3. Thinking while the arm moves (async) | Not promising |
| 4. Pruning context across rounds | Not promising as a new method |

## 1. Phase-aware adaptive reasoning effort

**Measured:**
- PEAR already adapts its reasoning length by phase. Median thinking tokens, by the previous action:
  - descend: 407
  - translate: 666
  - gripper: 2,089
  - rotate: 2,999
  - first round: 3,346
- Observable phase (previous action type plus outcome) explains only **R² = 0.20** of log reasoning length.
- Re-sampling the same context accounts for **25%** of the variance. That part is sampling noise that no
  context-based effort controller can remove.
- PEAR's effort setting is just a system-prompt sentence (`chat_template.jinja`), and RL ran only at xhigh.
- The paper reports that low effort cuts LIBERO success from 74.4% to 21.9%.

**Prior work.** Choosing effort per step is published for LLM agents and for robots:
- ARES ([2603.07915](https://arxiv.org/abs/2603.07915)): a router picks low, mid or high effort at each step.
- TAB ([2604.05164](https://arxiv.org/abs/2604.05164)): per-turn thinking budgets trained with GRPO.
- RARRL, "When Should a Robot Think?" ([2603.16673](https://arxiv.org/abs/2603.16673)): learns when a robot should invoke
  reasoning and how much compute to spend.

**Verdict: not promising.** A phase-aware effort controller would be ARES/TAB/RARRL applied to PEAR, a bag of tricks.
It would also only nudge a length adaptation the model already performs, through a prompt knob the model was never
trained to vary.

## 2. Quantization of robot-use LLMs

**Generic variants are published:**
- KV-cache quantization.
- Decoding-time fixes for overthinking: [2606.00206](https://arxiv.org/abs/2606.00206) and ReSET
  ([2606.13233](https://arxiv.org/abs/2606.13233)).
- Quantizing the recurrent state of hybrid models: LeapQuant, StepQuant,
  [2609.04098](https://arxiv.org/abs/2609.04098).
- Agent distillation: QAD ([2601.20088](https://arxiv.org/abs/2601.20088)) and SteerDistill
  ([2611.13410](https://arxiv.org/abs/2611.13410)).

**The open, robot-specific gap** is the RL-skill-delta mechanism we measured.
- The RL update is 0.15% of the weight norm, against 8.5% NVFP4 error.
- The quantized update has cosine 0.09 with the true update.
- The full plan is in `../pear_rl_delta_quant/DIRECTION.md`.

**Verdict: promising, only in that form.**

## 3. Thinking while the arm moves (asynchronous inference)

**Measured:**
- Physical motion per round is **0.71 s** on average (LIBERO env steps at 20 Hz).
- Reasoning per round takes **2.8 s** at PEAR's GB300 speed (747 tok/s) and **9.5 s** on an RTX 5090 (221 tok/s).
- So perfectly overlapping reasoning with motion saves at most **20%** of round time on a GB300 and **7%** on an
  RTX 5090.
- Round t+1's reasoning reads the frames captured after the motion. Overlap therefore requires guessing the next
  observation.

**Prior work.**
- VLA work hides latency during motion: real-time chunking, Asynchronous Robot Policies
  ([2611.05155](https://arxiv.org/abs/2611.05155)), Streaming-WAM ([2609.28927](https://arxiv.org/abs/2609.28927)) and
  FutureRTC ([2607.24008](https://arxiv.org/abs/2607.24008)), which predicts the next observation.
- Second Thought ([2608.13667](https://arxiv.org/abs/2608.13667)) reasons while an LLM agent acts.
- "Democratizing Robot-Use Agents" ([2611.01560](https://arxiv.org/abs/2611.01560)) already compares pipelined and
  serial execution.

**Caveat.** Real arms may move slower than LIBERO. The paper reports no per-move timing for its YAM robot.

**Verdict: not promising.** The motion window is too short to hide multi-second reasoning: at most 7–20%, and only if
the next observation can be predicted. The overlap and observation-prediction techniques already exist in the
asynchronous-VLA and agent literature.

## 4. Pruning context across rounds

**Measured:**
- Thinking is **78.5%** of what each round adds to the context. Images are 19.2%, and the answer plus action is 2.3%.
- The two long episodes accumulated 68k and 137k thinking tokens against about 16k of everything else.
- The paper reports that the 256k budget ends 19% of Goal and 33% of Long episodes, and 68.5% of RoboDojo episodes.
- So the lever is large.

**Prior work and methodology.**
- Dropping old thinking needs no method: it is one chat-template flag. PEAR deliberately sets `preserve_thinking`.
- Selective pruning or memory is crowded:
  - ECHO ([2606.31650](https://arxiv.org/abs/2606.31650)): selective turn memory.
  - SideQuest ([2602.22603](https://arxiv.org/abs/2602.22603)): model-driven KV-cache management.
  - AgentKV ([2609.14872](https://arxiv.org/abs/2609.14872)): phase-aware KV eviction.
  - CommitKV ([2608.07855](https://arxiv.org/abs/2608.07855)): KV compression at turn boundaries.
  - IntentKV ([2606.09916](https://arxiv.org/abs/2606.09916)): cross-turn KV pruning.
  - GateSight ([2610.14754](https://arxiv.org/abs/2610.14754)): KV eviction in hybrid models like PEAR's.
- Whether keeping reasoning helps is itself contested in the literature: [2606.00135](https://arxiv.org/abs/2606.00135)
  finds +2–5% on BFCL from keeping it, while others find dropping it helps.

**Verdict: not promising as a new method.** It is a large practical lever, since thinking is 78.5% of context growth
and the budget ends up to a third of hard episodes. But the trivial fix is a template flag and every selective variant
is published, so the contribution would be an ablation or an applied trick. It remains a useful control arm for the
RL-delta pilot (context exhaustion).
