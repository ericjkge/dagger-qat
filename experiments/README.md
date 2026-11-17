# Experiments (itsjenniepan)

Every study here ended **NO-GO**, or was left unfinished, against the bar we set ourselves: a novel, provable
method, not analysis and not a bag of tricks. They are kept on this branch for the record and are deliberately
**not on `main`**. Each folder is self-contained, with its write-up, code and results.

| Folder | Date | Question | Verdict | Write-up |
|---|---|---|---|---|
| `openvla_oft_offline_calibration/` | 2026-11-09 | Does LIBERO-Plus–perturbed calibration data make W4 GPTQ on OpenVLA-OFT more robust? (offline layer-streaming harness) | **Unfinished:** code and tests only, never run to results | `scripts/offline_pass.py` docstring |
| `pi05_contact_guard/` | 2026-11-14 | Do 4-bit activation errors near contact flip π0.5's contact outcomes, and can contact-aware rounding prevent it? | **NO-GO:** A4 damage is no worse than random error of the same size; the real failure (a `down_proj`-outlier speed bias) is already fixed by Hadamard rotation | `results/2026-11-14-pi05-contact/SUMMARY.md` |
| `pear_specdec/` | 2026-11-15 | Can PEAR's robot-reasoning structure (phases, cross-round reuse) beat DFlash2 speculative decoding? | **NO-GO:** oracle headroom ≤ 4.4%, and each piece is already published | `RESEARCH_DECISION.md` |
| `pear_efficiency_ideas/` | 2026-11-15 | Screen of the team's robot-LLM efficiency ideas | **NO-GO** on all of them (CPU-only checks) | `IDEAS_ASSESSMENT.md` |
| `pear_anytime_act/` | 2026-11-16 | Can PEAR act before finishing its reasoning (early exit, consensus, restarts, irreversibility gating, grounded-claim retention)? | **NO-GO.** Gated k=4 halves latency, but its principle failed validation; grounded retention is no better than random | `DIRECTION.md` (§7–§10) |
| `pear_context_exhaustion/` | 2026-11-16 | Can compact "servo memory" replace PEAR's growing reasoning history? | **NO-GO** | `IDEATION.md` §5 |
| `pear_rl_delta_quant/` | 2026-11-16 | Does protecting the RL update Δ = W_PEAR − W_base help NVFP4 quantization (exact-Δ oracle; Δ-regularized GPTQ H + λΔᵀΔ)? | **NO-GO.** The oracle is worse than RTN; Δ-GPTQ makes actions worse than plain GPTQ (paired CI excludes 0) | `DIRECTION.md` §9 |

## Takeaways that carry over
- **Weights are not where PEAR loses accuracy.** Every weight-quantization method lands within about 0.003 nats/token
  of BF16 on action tokens.
- **Our LIBERO harness capped episodes at 30 rounds; PEAR's paper has no cap.** Success rates measured under
  `--max-rounds 30` are biased low (`pear_anytime_act/DIRECTION.md` §8).
- **Teacher-forced dNLL on the 14 recorded `dflash2_main` episodes is a cheap, sensitive screen.** It costs about
  $2–4 per arm, and paired episode-bootstrap CIs separate effects of about 0.001 nats/token.

## Running the code
- **Paths:** scripts expect to run from inside their own folder. `pod/` scripts were run on Runpod H200/A100 pods
  under `/workspace`, so adjust those paths.
- **`pi05_contact_guard/`:** set `PYTHONPATH=src` and use openpi PyTorch `pi05_libero`.
- **`openvla_oft_offline_calibration/`:** imports `dqat.gptq`. That module is not ours; it lives on `main`'s history
  (commit `a90c67c`), so check out that module alongside before running.
