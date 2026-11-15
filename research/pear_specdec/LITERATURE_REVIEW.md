# Literature review and novelty assessment: structure-aware speculative decoding for PEAR (2026-11-14)

**How sources were checked.** Each entry below is marked:
- **[P]**: read in the primary source (paper text, repo or model card).
- **[A]**: abstract or arXiv page only.
- **[S]**: a secondary summary only.

Nothing marked [S] should be cited without re-checking. Finding nothing in a search is not evidence of novelty.

## 0. The system being accelerated (all [P])

- **PEAR** (Tang, Zhang, Guo, …, Liu, Han; MIT/Berkeley/UCSD/NVIDIA/Inco AI; preprint dated 12 Nov 2026).
  - Links: [blog](https://mit-han-lab.github.io/pear-blog/), [paper](https://mit-han-lab.github.io/pear-blog/paper.pdf),
    [code](https://github.com/mit-han-lab/pear) (Apache-2.0).
  - Checkpoints: `mit-han-lab/Qwen3.8-27B-PEAR` (BF16), `…-LIBERO_only`, and `…-NVFP4`.
  - **Architecture:** hybrid `Qwen3_5ForConditionalGeneration`.
    - 64 layers, repeating 3 Gated-DeltaNet linear-attention layers then 1 full-attention layer: 48 linear, 16 full.
    - Full attention: 24 query heads, 4 KV heads, head dim 256. Vocabulary 248,320. Context up to 262k.
    - 27-layer ViT, patch 16, 2×2 merge.
    - No MTP head (`mtp_num_hidden_layers: 0`).
  - **Workload per round:**
    - Input: the arm and gripper state as JSON, plus 2 LIBERO camera frames (agentview, wrist; 512 px).
    - Output: `<think>…</think>`, then visible text ending in a fenced JSON action.
    - The action is `{"arm":{x,y,z,roll,pitch,yaw},"gripper":{width}}`, **69 tokens** under the model's tokenizer
      (our count on a representative action).
    - Thinking is about **3.23k tokens per round** at xhigh and 0.86k at low (paper Fig. 9).
    - RL episodes average 19.7 rounds and 94k tokens.
    - **All earlier rounds, including their thinking, stay in context:** the chat template sets
      `preserve_thinking`, which defaults to true.
  - **Serving:**
    - vLLM, one session per GPU (`max_num_seqs: 1`), prefix caching with Gated-DeltaNet state kept at block
      boundaries (`mamba_cache_mode: align`).
    - Sampling: temperature 1.0, top-p 0.95.
    - Speculation: `method: dflash`, `num_speculative_tokens: 7`, `draft_sample_method: probabilistic`.
    - Custom decode and verify **megakernels**, for Blackwell only (GB300 in BF16 or NVFP4; RTX 5090 in NVFP4).
    - Reported speeds: 747 tok/s on a GB300, 221 tok/s on an RTX 5090 (both NVFP4), and 52 tok/s on an M5 Max.
    - Average DFlash2 acceptance length is 3.2. The authors say a drafter tuned for PEAR "may have a longer acceptance
      length".
  - **Quantization:** "NVFP4 quantization reduces our model's accuracy and lengthens its chain of thought, so all
    evaluations use BF16 weights."
- **DFlash** (Chen, Liang, Liu; [arXiv 2602.06036](https://arxiv.org/abs/2602.06036), ICML 2026) [A].
  - A block-diffusion drafter that proposes a whole block in one pass, conditioned on target hidden states.
  - Claims >6× lossless speedup, and up to 2.5× more than EAGLE-3.
- **DFlash2 drafter** `incoai/Qwen3.8-27B-DFlash2` [P, model card]:
  - About 2B parameters, block size 8 (7 drafts), top-k candidates per position plus a learned selector path,
    "two-tap dynamic convolutions".
  - Lossless. Served by SGLang (`--speculative-algorithm DFLASH`) and by a vLLM PR (#52816).
  - Acceptance length on the **base** Qwen3.8-27B (H200, temperature 1.0, xhigh): GSM8K 5.46, MATH 5.28, HumanEval
    4.39, MBPP 4.79, MT-Bench 4.10.
  - Concurrency-1 speedup over AR: 2.67–3.43×. At concurrency 32 it falls to 1.01–1.45×.
  - **PEAR's 3.2 is well below all of these:** the drafter was trained for the base model, not the RL-tuned PEAR
    policy, on general text, not robot-use traces.
- **SGLang Spec V2 / DFlash blog** ([LMSYS, 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)) [P]:
  engine-level overlap scheduling only. **No adaptive draft length or adaptive verification.**

## 1. Prior work by area

### 1.1 Generic speculative decoding
| Work | Mechanism | Lossless | Code | Overlap with us |
|---|---|---|---|---|
| Leviathan et al. 2023; Chen et al. 2023 | Draft-and-verify with rejection sampling | Yes | Yes | The basis; our `speculative_step` reimplements it |
| EAGLE-1/2/3 | Feature-level drafter, dynamic trees | Yes | Yes | Strong generic baseline |
| DFlash / DFlash2 | Block-diffusion drafter | Yes | Yes | PEAR's baseline |
| SpecDec++ ([2405.19715](https://arxiv.org/abs/2405.19715)) [A] | Draft length as an MDP; optimal stop is a threshold on rejection probability; trained head | Yes | Yes | **Already supplies the "optimal block length" theory that Candidate A would re-derive** |
| AdaEDL [S] | Training-free entropy-based stop rule | Yes | ? | Adaptive length baseline |
| BanditSpec ([2505.15141](https://arxiv.org/abs/2505.15141)) [A] | Bandit over speculative configurations, training-free | Yes | ? | **Online selection among drafting strategies, which is Candidate A's "switching"** |
| DISCO [S] | Trained stop classifier | Yes | ? | Adaptive length |
| SuffixDecoding ([2411.04975](https://arxiv.org/abs/2411.04975), NeurIPS 2025) [A] | Model-free suffix trees over prompt and prior outputs; built for repetitive agentic workloads; combined with model drafters at a threshold | Yes | Yes (Arctic Inference / vLLM) | **Direct prior art for Direction B** |
| Graft, "Draft Less, Retrieve More" ([2605.20104](https://arxiv.org/abs/2605.20104), NeurIPS 2026) [A] | Prunes low-confidence model-draft branches and grafts retrieved continuations into the freed tree budget; includes a DFlash-style exploration | Yes | Not found | **Hybrid model plus retrieval drafting, including with DFlash** |
| WhiFlash ([2606.07710](https://arxiv.org/abs/2606.07710), EMNLP 2026) [A, read 2026-11-14] | Token-level routing between an AR drafter (EAGLE-3) and a diffusion/parallel drafter (DFlash) using an entropy rule or a learned policy; Lazy Catch-up and KV-only Prefill keep switching overhead below 7% | Not stated in the abstract | Not found | **Phase/regime-aware drafter routing on agentic workloads ("reasoning-style" vs. "structured" output). Directly pre-empts Direction A as "route per phase"**. Reports +69.6% over EAGLE-3 and +37.3% over DFlash |
| ReTrace ([2608.29748](https://arxiv.org/abs/2608.29748)) [S] | Conditions drafts on the rejected suffix (within a request) | Yes | ? | Minor |
| QuantSpec ([2502.10424](https://arxiv.org/abs/2502.10424), ICML 2025) [A] | Self-speculation with 4-bit weights and a hierarchical 4-bit KV cache; >90% acceptance | Yes | Yes | **Direction C** |
| QSpec ([2410.11305](https://arxiv.org/abs/2410.11305)) [A] | One W4 model: W4A4 drafting, W4A16 verification, shared KV | Yes w.r.t. W4A16 | Yes | **Direction C** |
| ML-SpecQD [S]; "Speculative Decoding Meets Quantization" ([2505.22179](https://arxiv.org/abs/2505.22179)) [A] | Multi-level quantized drafts; finds 4-bit memory savings eaten by speculation compute | Yes | ? | **Direction C** |

### 1.2 Robotics-specific
All of these target **action-token VLAs** (OpenVLA family), not reasoning LLM agents. None models CoT.

| Work | Mechanism | Lossless | Overlap |
|---|---|---|---|
| Spec-VLA ([2507.22424](https://arxiv.org/abs/2507.22424), EMNLP 2025) [A] | EAGLE plus **relaxed** acceptance by action-bin distance | **No** | Action tokens only |
| KERV ([2603.01581](https://arxiv.org/abs/2603.01581), DAC 2026) [A] | Kalman-filter correction of rejected action tokens; kinematics-adaptive threshold | **No** | Action tokens only |
| WA-SpecDec ([2608.08725](https://arxiv.org/abs/2608.08725)) [A] | World-model scene features at prefill; acceptance rule unchanged | Yes | Action tokens only |
| Zora ([2611.11098](https://arxiv.org/abs/2611.11098), 11 Nov 2026) [P, partial] | **Retrieval** of future action sequences keyed by observation/state embeddings (HNSW), batch verification with a discrepancy threshold, plus a hardware accelerator | **No** (thresholded) | **Cross-episode retrieval for robots; does not cover CoT** |
| HeiSD ([2603.17573](https://arxiv.org/abs/2603.17573)) [S] | Hybrid speculative decoding with kinematic awareness | ? | Action tokens |

### 1.3 Reasoning-specific
| Work | Mechanism | Lossless | Overlap |
|---|---|---|---|
| SpecReason ([2504.07891](https://arxiv.org/abs/2504.07891)) [A] | Small model drafts reasoning *steps*; base model judges utility | **No** (semantic) | Step-level, not token-level |
| Lookahead Reasoning [S] | Step-level parallelism on top of token-level SD; semantic verifier | **No** | Same |
| ConfSpec ([2602.18447](https://arxiv.org/abs/2602.18447)) [S] | Confidence-gated cascaded step verification | **No** | Same |
| Speculative Thinking [S] | Large model guides a small one at the reasoning level | **No** | Same |
| SSR, "Self-Speculation for Faster Reasoning Models" ([2608.20359](https://arxiv.org/abs/2608.20359)) [S] | Drafts from a partial CoT's answer distribution | ? | Reasoning-to-answer shortcut |

### 1.4 Structured output, tool use and agents
| Work | Mechanism | Lossless | Overlap |
|---|---|---|---|
| ToolSpec ([2604.13519](https://arxiv.org/abs/2604.13519)) [A] | FSM compiled from tool schemas supplies scaffolding tokens; retrieved past calls supply values; up to 4.2× | Yes | **Phase-aware drafting for the structured segment, i.e. Direction A's action half** |
| SchemaFill ([2610.07086](https://arxiv.org/abs/2610.07086)), OoO-Spec ([2608.00814](https://arxiv.org/abs/2608.00814)) [S] | Slot-parallel / out-of-order speculation for tool calls | ? | Same |
| TraceSpec ([2610.41177](https://arxiv.org/abs/2610.41177), Oct 2026) [A] | SD for LLM agents; **phase- and state-conditioned** acceptance gaps; reports that aggregate acceptance hides per-phase trends | Yes | **Characterizes phase-dependent acceptance in agent trajectories** |
| DLite ([2610.23787](https://arxiv.org/abs/2610.23787), Oct 2026) [A] | History-free drafter: draft models lose up to 80% acceptance length on long multi-turn agent contexts | Yes | **Long-history drafter degradation, which directly bears on PEAR's 94k-token episodes** |

## 2. Novelty matrix

Legend: **PUB** already published; **EXT** obvious extension; **UNV** potentially novel but unverified;
**NEW** differentiated by a new mechanism.

| Our direction | Closest prior work | Verdict | Why |
|---|---|---|---|
| A1. Different drafter or schedule for reasoning vs. action tokens | ToolSpec (schema FSM), SpecDec++ / AdaEDL (adaptive length), BanditSpec (strategy selection), TraceSpec (phase-conditioned) | **PUB / EXT** | Each piece exists. **Amdahl also caps it:** an action is 69 of ~3,300 generated tokens, so making action decoding free saves ≤2.1% |
| A2. Phase-aware schedule *within* reasoning (sub-phases of the CoT) | SpecDec++, AdaEDL, BanditSpec, TraceSpec | **EXT** unless profiling shows sub-phase structure that a per-token confidence signal cannot already see | Optimizing A(z,k,b)/C(z,k,b) per regime is the SpecDec++ MDP with a regime variable |
| B. Cross-round reasoning reuse as drafts | SuffixDecoding (prior outputs, agentic), prompt lookup / n-gram (vLLM and SGLang built-ins), Graft (retrieval plus model tree, DFlash variant), Zora (cross-episode retrieval for robots) | **PUB** for verbatim reuse | Earlier rounds' thinking is **already in context**, so in-context lookup sees it. Only *non-verbatim, observation-conditioned* reuse could be new, and nobody has shown such structure exists. To be measured |
| C1. Low-precision drafter, BF16 target | QuantSpec, QSpec, ML-SpecQD, 2505.22179 | **PUB** | Also limited headroom: the drafter is ~2B against a 27B target, and drafting is a small share of step time |
| C2. Quantized target | — | **Not lossless w.r.t. BF16** | PEAR itself reports NVFP4 lengthens CoT and lowers accuracy |

## 3. What would count as new (still to be tested, not claimed)

1. **A cost model with the *robot round* as the unit.** End-to-end latency per round is decode plus prefill of the
   new observation plus sim/IK. Speculation that lengthens the CoT, or shifts work to prefill, is not a win. This is
   a measurement framing, not by itself an algorithm.
2. **Exploiting RL-policy drift from the drafter's base model.** The 3.2 vs. 4.1–5.5 gap suggests the gains are in
   *where* the RL-tuned policy departs from the base model. Retraining the drafter is explicitly disallowed as a
   contribution. A training-free mechanism that uses the base/RL divergence structure would be unusual, but it is
   only a hypothesis.
3. **Non-verbatim temporal structure.** If successive rounds' reasoning is a *template with changed numbers*
   (coordinates, distances), a drafter that copies the template and fills slots from the new observation
   (proprioception is given as JSON in the prompt) would go beyond suffix retrieval.
   - **Overlap:** ToolSpec does FSM plus slot filling for tool calls; doing it for CoT conditioned on the new state
     is the possible difference.
   - **Must be measured first.**

## 4. Unread or unverified items to resolve before any GO
- WhiFlash: abstract read 2026-11-14. Routes per token between an AR drafter and a DFlash-style drafter with an entropy or learned policy. This covers "phase-aware drafter choice" generically (A). Full text still unread.
- Graft: abstract read 2026-11-14. Lossless and training-free; retrieved tokens fill pruned tree budget; a preliminary DFlash-style variant. The retrieval source is not specified in the abstract. Full text still unread.
- TraceSpec full text (whether it already reports reasoning vs. structured-output acceptance).
- DLite full text (long-history degradation numbers, relevant to PEAR's 94k-token episodes).
- ML-SpecQD primary source; AdaEDL, DISCO and HeiSD primary sources.
- Exact vLLM DFlash support status (the drafter card points to PR #52816; PEAR uses vLLM with its own Blackwell
  kernels).

## 5. Update after measurement (2026-11-15)
The novelty matrix above was written before any GPU run. Measured results (`STRUCTURE_PROFILING.md`,
`BASELINE_REPORT.md`) confirm the PUB / EXT classifications and add the missing magnitudes:
- **A1:** the action is 2.9% of decode, with Amdahl ≤ 1.029×.
- **A2:** verify cost is flat in K (20.8 vs. 20.1 ms), so adaptive length can gain < 3.5%.
- **B:** previous-round retrieval gives a +1.46% oracle gain, vs. +0.65% for the shuffled-episode control. The
  "non-verbatim, observation-conditioned reuse" hypothesis found no token-level signal.
- **Long-context degradation:** −14% acceptance from ≤ 8k to 32–64k, consistent with DLite.

Final verdicts: `RESEARCH_DECISION.md` (NO-GO on A, B and C).
