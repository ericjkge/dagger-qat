# Phase 0: feasibility audit (2026-11-14)

The π0.5 work is untouched. This project lives in `experiments/pear_specdec/`, with its own `pear_specdec/` package,
`tests/`, `configs/`, `scripts/` and `results/`.

## Answers to the audit questions

| # | Question | Answer (source) |
|---|---|---|
| 1 | Is PEAR cloned or installed here? | No. The local machine is a Mac with no NVIDIA GPU and 11 GB of free disk. The repo is public (Apache-2.0, 1 commit, 837 files) |
| 2 | Are the checkpoints accessible? | Yes, public on HF: `mit-han-lab/Qwen3.8-27B-PEAR` (BF16, ≈54 GB), `…-LIBERO_only`, `…-NVFP4`. Drafter `incoai/Qwen3.8-27B-DFlash2` (≈2B, BF16, ≈4 GB, Apache-2.0) |
| 3 | Architecture, tokenizer, template, markers, action format | Hybrid Qwen3.5-class model: 48 Gated-DeltaNet layers plus 16 full-attention layers (4 KV heads × 256), vocabulary 248,320, 27-layer ViT, no MTP. Template: `<think>\n…\n</think>\n\n{content}`, with `reasoning_effort` ∈ {low, medium, xhigh} and `preserve_thinking` defaulting to true, so earlier rounds' thinking stays in context. Action: a fenced JSON `{"arm":{x,y,z,roll,pitch,yaw},"gripper":{width}}` of **69 tokens**. LIBERO observation: 2 frames (agentview, wrist) at 512 px plus JSON state |
| 4 | Inference backend and spec-decoding support | PEAR uses vLLM with a Hydra server (`pear/deploy/server`), `method: dflash`, 7 speculative tokens, probabilistic draft sampling, temperature 1.0, top-p 0.95, one session per GPU, prefix caching (`mamba_cache_mode: align`). PEAR's decode and verify **megakernels** are built for compute capability 10.3 (GB300 / B300) and the RTX 5090 (NVFP4 only). `kernels=null` falls back to vLLM's own kernels on other GPUs. SGLang supports `DFLASH` upstream |
| 5 | How the DFlash2 drafter is loaded | `draft=<dir>` plus `speculative={method: dflash, num_speculative_tokens: 7, draft_sample_method: probabilistic}`. The megakernel draft path (`deploy/blackwell/bf16/draft/`) uses block QB=8, top-k 16 candidates per position and a learned selector |
| 6 | Are target logits, draft logits, accept counts and timings accessible? | Not through the OpenAI API by default. vLLM exposes `logprobs` (processed, up to 32) and spec-decode metrics (accepted per position) in its Prometheus counters. Per-iteration accepted lengths need a small hook on the spec-decode worker. **Planned instrumentation:** (a) server-side hook logging every verify step (drafts, accepted count, draft and verify ms); (b) offline teacher-forced scoring of recorded traces (one prefill pass returns target log-probs for any deterministic drafter's proposals; see `pear_specdec/acceptance.py`) |
| 7 | Can LIBERO run? | Yes, CPU MuJoCo/robosuite (`pear/envs/libero/env.py`, LIBERO commit `8f1084e`, seed 7, 10 stabilization steps). PEAR's eval launcher assumes a 32-node Ray cluster, so a single-node episode loop (env plus OpenAI client, same prompts) has to be written |
| 8 | GPUs, VRAM, disk, dependencies | Need ≥80 GB VRAM for the BF16 target (54 GB) plus the drafter (4 GB) plus state. KV is small: 16 full layers × 4 × 256 × 2 × 2 B ≈ 64 KB/token, ≈6 GB for 94k tokens. Disk ≈ 70 GB for checkpoints plus envs. CUDA 13 builds (torch cu130) |

## Feasibility conclusions
- **Acceptance statistics** (Experiments A and B) do not depend on the kernels, so an H200 with vLLM's or SGLang's
  DFlash path measures them faithfully. The DFlash2 card's own numbers were measured on an H200 with SGLang.
- **Latency faithful to PEAR's deployment** needs their megakernels, i.e. a B300 (compute 10.3) in BF16. Latency on
  H200 / vLLM kernels is a *different system*; we report it as such and never mix the two.
- **Experiment C:** a BF16 target with an NVFP4 drafter needs Blackwell. PEAR's NVFP4 *target* is not lossless
  w.r.t. BF16 (the paper says so). Given the literature review (QuantSpec, QSpec, ML-SpecQD), C is deprioritized.
- **The Amdahl bound kills Direction A1 before any GPU time:** 69 action tokens vs. ≈3,230 thinking tokens per round.
  Making the action free saves ≤ 3,299/3,230 − 1 = **2.1%** of decode tokens.

## Staged GPU plan (needs the user's approval)
Stage 1, "smallest reproducible experiment", on 1× H200:
1. Install, download the checkpoints and smoke-test, ≈1 h.
2. Collect DFlash2 traces with per-verify-step logs on **6 LIBERO episodes** (3 suites × 2 init states, xhigh,
   seed 0), about 6 × 20 rounds × 3.2k ≈ 0.4M generated tokens, ≈1 h at concurrency 1.
3. Run the AR baseline on a few rounds, replayed from the same contexts.
4. Teacher-forced scoring of the recorded traces for the offline drafters (context lookup, previous round only,
   shuffled-round control) and for DFlash2 acceptance per phase.

Total ≈ 3–4 GPU-hours. Expand only if Experiment B shows non-trivial cross-round acceptance beyond the in-context
lookup baseline.

## Outcome of Stage 1 (2026-11-15)
- **Ran on:** 1× H200 (not B300), so acceptance is faithful and latency is a different system, as planned.
- **Environment fixes needed:**
  - CUDA 13 forward-compat driver (`cuda-compat-13-0`) for the cu130 torch on driver 575.
  - CUDA 13.0 toolkit for SGLang / FlashInfer JIT.
  - DeepGEMM JIT off.
  - LIBERO installed editable in compat mode.
  - NVIDIA EGL for rendering (the Mesa EGL device platform was unavailable).
- **Collected:** 14 episodes instead of 6, because episodes were cheaper than estimated.
- **Benchmarked:** AR, NGRAM, DFlash2 K = 3 and DFlash2 K = 7, in paired replay.
- **GPU time:** 2.46 h, ≈ $8.80 of the $100 budget.
