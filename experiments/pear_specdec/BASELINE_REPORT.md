# Phase 2: baseline report, PEAR + DFlash2 on 1× H200 (2026-11-15)

**Not PEAR's deployment latency.** These are measurements of PEAR's model and drafter in a *different serving stack*:
SGLang on an H200, instead of PEAR's vLLM with Blackwell megakernels. Acceptance statistics transfer across stacks;
absolute latencies do not. PEAR reports 747 tok/s on a GB300.

## Configuration
| Item | Value |
|---|---|
| Hardware | 1× H200 SXM 141 GB (Runpod community pod `d74dh65zxcfpzs`), driver 575.57.08 (CUDA 12.9) + `cuda-compat-13-0` 580.196.07 so the cu130 torch can run |
| Software | SGLang `0.5.6.post3.dev14779+g747677224` (git `7476772`), torch `2.14.1+cu130`, flashinfer `0.7.4`, CUDA toolkit 13.0 for JIT. DeepGEMM JIT off (`SGLANG_ENABLE_JIT_DEEPGEMM=0`; BF16 model, unused). PEAR repo `f32aa1f` |
| Server flags | `--max-running-requests 1 --context-length 262144 --mem-fraction-static 0.85 --random-seed 0 --enable-metrics` + spec flags below (`configs/pod/serve.sh`) |
| Target / drafter | `mit-han-lab/Qwen3.8-27B-PEAR` BF16 / `incoai/Qwen3.8-27B-DFlash2` BF16 |
| Sampling | temperature 1.0, top-p 0.95 (exactness checks: temperature 0) |
| Instrumentation patch | 4 lines in `sglang/srt/managers/schedule_batch.py` (`configs/pod/patch_sgl.py`), active only when `PEAR_SPEC_STEP_LOG` is set. No effect on decoding |

**Deviations from PEAR's setup.**
1. **Engine and kernels:** SGLang and its kernels instead of vLLM + megakernels.
2. **Rendering:** LIBERO frames come from NVIDIA EGL instead of Mesa llvmpipe.
3. **Prefix cache:** SGLang's hybrid Gated-DeltaNet prefix cache never hit on these multimodal prompts
   (`cached_tokens = 0` on every request). Every round re-prefills the whole context, whereas PEAR's vLLM deployment
   caches it (`mamba_cache_mode: align`). Our prefill shares are therefore **upper bounds**.

## Baselines
| Baseline | What it is | Status |
|---|---|---|
| Autoregressive (AR) | No speculation | Measured |
| **PEAR's configuration** | DFlash2, block 8 = 7 drafts, probabilistic sampling path (SGLang `--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`) | Measured. Acceptance matches the paper (3.25 vs. 3.2) |
| Strong generic speculative decoder | DFlash2 *is* the strongest generic drafter available for this target: it was not trained on robot data. No EAGLE-3 / MTP head exists for Qwen3.8-27B-PEAR (the target has no MTP) | Same as above |
| Training-free generic | SGLang `NGRAM` (lookahead-style n-gram tree, 12 draft tokens) | Measured |
| Draft-length policy | DFlash2 with K = 3 (block 4) vs. K = 7, which measures verify cost vs. K. A per-step adaptive policy (SpecDec++/AdaEDL-style) is **bounded** by that curve, not implemented (see below) | Measured (bound) |

## Paired replay benchmark (decoding only)
- **Contexts:** 12 recorded contexts from the 14 LIBERO episodes, from 1.0k to 122k prompt tokens, each generated twice
  (`scripts/replay_bench.py`; the reconstructed prompts match the recorded ones character for character).
- **Measured:** the reasoning segment of each round, which is 97% of decode tokens.
- **Metric:** decode tokens/s = completion tokens / (server e2e − prefill).
- **Speedup:** ratio paired by context, with bootstrap 95% CIs over the 24 pairs.

| Config | Decode tok/s | ≤2k ctx | ≥40k ctx | Tokens / verify step | ms / verify step | ms / token | **Decode speedup vs. AR** |
|---|---|---|---|---|---|---|---|
| AR | 64.0 | 68.5 | 62.9 | 1 | 15.7 | 15.7 | 1.00× |
| NGRAM (SGLang) | 93.2 | 102.5 | 89.1 | 1.87 | 20.2 | 10.9 | 1.46× (1.42–1.49) |
| DFlash2, K = 3 | 139.0 | 158.5 | 132.9 | 2.77 | 20.1 | 7.3 | 2.17× (2.09–2.24) |
| **DFlash2, K = 7 (PEAR)** | **162.1** | 189.0 | 152.8 | **3.34** | **20.8** | **6.3** | **2.53× (2.44–2.61)** |

Raw rows: `results/bench/*.jsonl`. Summary: `results/bench/bench_summary.json`.

**Cost of a verify step vs. draft length.**
- A DFlash2 step (drafting + verifying 8 tokens) costs 20.8 ms, against 15.7 ms for one AR token (1.33×).
- With 4 tokens (K = 3) it costs 20.1 ms. Cutting the drafts from 7 to 3 saves only **3.5%** of step time but loses 17% of
  accepted tokens.
- So at batch size 1 on this hardware, verification is memory-bound and almost flat in K. **Any adaptive draft-length
  policy** (SpecDec++, AdaEDL, or a phase-aware one) can save at most ≈ 3.5% per step on steps where it shortens K, and
  it pays acceptance loss whenever the step would have accepted more. Its upper bound over DFlash2 is therefore **< 3.5%**.
  K > 7 is not available: the drafter's block size is fixed at training.

## End-to-end, in the real episodes (14 episodes, 170 rounds, DFlash2)
| Component | Share of round wall time | Mean per round |
|---|---|---|
| Reasoning decode | 67.4% | 12.70 s |
| Answer request (≈ 61 tokens, incl. its prefill) | 4.0% | 0.75 s |
| Prefill, incl. vision (**uncached**, upper bound) | 14.2% | 2.67 s |
| Simulator (`LiberoEnv.rollout`, CPU) | 14.1% | 2.66 s |
| Client/HTTP overhead | 0.2% | 0.05 s |

- **Separating vision encoding from prefill:** not possible with the request-level counters, so they are reported
  together.
- **Episode latency:** mean episode = 12.2 rounds ≈ 230 s of wall time with DFlash2.
- **End-to-end speedup:** normalize a DFlash2 round to 1, with 0.286 non-decode and 0.714 decode. Since AR decode is
  2.53× slower, an AR round takes 0.286 + 0.714 × 2.53 ≈ 2.09. DFlash2's **end-to-end round speedup is ≈ 2.09×**,
  vs. 2.53× decode-only, under our uncached prefill. This is a derived figure: we did not run AR episodes end to end. With a working prefix cache the
  decoding share grows and the end-to-end speedup approaches the decode speedup.
- **Amdahl for new work:** any further *decoding* improvement over DFlash2 is diluted by the ≈ 29% non-decode time.
  A 5% decode gain is ≈ 3.5% end to end.

## Exactness
1. **Algorithm unit tests** (`tests/test_specdec_core.py`, 8 passing):
   - Monte-Carlo check that one speculative step reproduces the target distribution.
   - E[accepted] for a deterministic drafter equals Σ Π p(dⱼ), which justifies the offline retrieval estimates.
   - n-gram index ≡ reference suffix matcher.
2. **Greedy agreement** (temperature 0, 512 tokens, 3 contexts):
   - DFlash2 K = 7 diverges from AR at token 135, 296 and 107.
   - K = 3 diverges at 299, is identical on one context, and diverges at 107.
   - NGRAM diverges at 135, is identical on one context, and diverges at 321.
3. **Divergence diagnosis** (`results/bench/tie_check.json`): at every divergence the target's next-token
   distribution, recomputed by plain prefill, has **both divergent tokens as its top 2**, with a log-probability margin
   of **0.00** (4 of 5) or **0.25** nats (1 of 5, a single bf16 rounding step). These are exact or near ties broken
   differently by different kernel shapes (8-token verify vs. 1-token decode). That is the expected bf16 behavior and
   shows no sign of a verification error.

**Conclusion.** The PEAR + DFlash2 baseline is reproduced: acceptance 3.25–3.34 tokens per step (paper: 3.2) and a
2.53× decode speedup over AR on H200/SGLang. Any proposed method must beat **DFlash2 at 162 tok/s and 3.34 tokens per
step** on these same contexts. Beating AR (64 tok/s) is not the bar.
