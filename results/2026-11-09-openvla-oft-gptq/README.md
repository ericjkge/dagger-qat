# OpenVLA-OFT / LIBERO-Long: BF16, AWQ and GPTQ

The completed matched evaluation gives **93.4% for unquantized BF16, 93.4% for
AWQ, and 93.0% for GPTQ**. GPTQ succeeds on two fewer episodes than either
baseline, a difference of **−0.4 percentage points**.

| Method | Successes | Episodes | Success rate | Difference from BF16 |
| --- | ---: | ---: | ---: | ---: |
| Unquantized BF16 | 467 | 500 | 93.4% | — |
| AWQ W4/g128, simulated in BF16 | 467 | 500 | 93.4% | 0.0 pp |
| GPTQ W4/g128, simulated in BF16 | 465 | 500 | 93.0% | −0.4 pp |

GPTQ completed at 12:02 p.m. EST on November 9, 2026, in 1h29m37s, with exit
code 0:0, no restarts and zero recorded Python exceptions. All 1,500 records
across the three methods were validated for complete task/state/seed pairing,
unique identities, task names, success flags, action/call counts and summary
agreement. A shared simulator warning is handled separately below.

## Paired comparison

Differences are candidate minus reference on the same episodes. Intervals are
descriptive percentile bootstrap intervals from 10,000 matched-episode resamples
with seed 7, following the original BF16/AWQ report.

| Candidate − reference | Difference | 95% interval | Candidate-only successes | Reference-only successes |
| --- | ---: | ---: | ---: | ---: |
| AWQ − BF16 | 0.0 pp | [−2.2, +2.2] pp | 15 | 15 |
| GPTQ − BF16 | −0.4 pp | [−2.6, +1.8] pp | 15 | 17 |
| GPTQ − AWQ | −0.4 pp | [−2.6, +1.8] pp | 15 | 17 |

GPTQ and each baseline share 450 successes and 18 failures. The equal pairwise
counts against BF16 and AWQ do not mean those comparisons involve identical
episode sets. Across all three methods, 439 episodes succeed in every arm and
14 fail in every arm; the [aggregate JSON](comparison.json) includes the complete
eight-pattern outcome counts.

The observed GPTQ drop is small relative to the interval. This sample does not
establish that GPTQ is worse, better or equivalent to either baseline. These
intervals describe the sampled episodes under one base seed; they do not capture
variation across calibration seeds, model training seeds or new tasks.

## Per-task successes

Every row uses the same 50 initial states in each arm. Task differences are
descriptive observations, without claims of statistical significance.

| ID | Task | BF16 | AWQ | GPTQ | GPTQ − BF16 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Alphabet soup and tomato sauce into basket | 45/50 | 47/50 | 47/50 | +4.0 pp |
| 1 | Cream cheese and butter into basket | 50/50 | 50/50 | 49/50 | −2.0 pp |
| 2 | Turn on stove and put moka pot on it | 49/50 | 49/50 | 50/50 | +2.0 pp |
| 3 | Black bowl into bottom drawer, then close | 50/50 | 50/50 | 46/50 | −8.0 pp |
| 4 | White mug on left plate; yellow/white mug on right | 47/50 | 45/50 | 46/50 | −2.0 pp |
| 5 | Book into back compartment of caddy | 46/50 | 46/50 | 46/50 | 0.0 pp |
| 6 | White mug on plate; chocolate pudding to its right | 45/50 | 45/50 | 43/50 | −4.0 pp |
| 7 | Alphabet soup and cream cheese into basket | 49/50 | 48/50 | 49/50 | 0.0 pp |
| 8 | Both moka pots on stove | 40/50 | 42/50 | 42/50 | +4.0 pp |
| 9 | Yellow/white mug into microwave, then close | 46/50 | 45/50 | 47/50 | +2.0 pp |

The largest observed GPTQ loss is on the bowl/drawer task: 46/50 versus 50/50 for
both baselines. Other tasks improve or remain close, so the aggregate difference
does not describe a uniform change in behavior.

## Simulator-warning sensitivity

Each method emitted one MuJoCo acceleration-instability warning in the same
matched episode. All three recorded it as an unsuccessful 520-action rollout.
The warning did not raise a Python exception. Preserve the raw 500-episode
scores above, and separately exclude that shared pair from all three arms:

| Method | Successes / retained episodes | Success rate |
| --- | ---: | ---: |
| Unquantized BF16 | 467/499 | 93.6% |
| AWQ | 467/499 | 93.6% |
| GPTQ | 465/499 | 93.2% |

GPTQ remains about 0.4 points lower than either baseline (exact difference
−0.4008 pp); both paired intervals remain [−2.6, +1.8] pp after rounding. This
sensitivity check leaves the interpretation unchanged. The excluded episode is
not silently removed from the primary score or attributed to quantization.

## Protocol, provenance and limits

All arms use the same checkpoint, tasks and initial states, per-episode policy
seeds, environment seed, L1 head, two images, proprioception, 90% center crop,
256px simulator observations, eight-action chunks, 10 settling steps and
520-action horizon. The policy loader, demonstration loader, rollout code and
AWQ implementation are unchanged between the original baseline and GPTQ sources.

AWQ and GPTQ use the exact same 32 task-balanced training-demo observations and
normalization statistics, with the same seven Llama projections per block at
W4/group 128. Vision, projectors, embeddings, LM output and action head remain
unquantized. GPTQ uses natural column order, asymmetric groups, 128-column
blocks, 1% Hessian damping and true-sequential calibration, following original
reference revision `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`. Downstream decoder
inputs include earlier quantization errors. No environment rollouts or evaluation
outcomes were used to fit the quantizer.

- GPTQ source: clean `a90c67c9eac5ddf9ba44a9c03ac2b4240c39cd93`.
- BF16/AWQ source: clean `ef1fa1608b8299b44307f1ae6c1ca2167decd246`.
- Checkpoint: `moojink/openvla-7b-oft-finetuned-libero-10`, revision
  `95220f9a3421a7ff12d4218e73d09ade830fa9a3`.
- Calibration dataset: `yifengzhu-hf/LIBERO-datasets`, revision
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- GPTQ cache SHA256, independently verified after completion:
  `d97fba21c6b297b248b6997477e811dafc5cc82dab07cd5dd7030db818ecef42`.
- All three full runs used A100-SXM4-80GB GPUs on Harvard Cannon, Python 3.10.14
  and Torch 2.2.0+cu121. The pinned submodules and Transformers revision match the
  [original baseline report](../2026-11-09-openvla-oft-libero-long/README.md#reproducibility).
- GPTQ calibration plus a successful smoke rollout took 9m20s. Cache reload
  reproduced that outcome. All 28 tests passed locally and on Cannon's exact
  runtime before GPTQ ran. The old BF16/AWQ aggregate comparison is unchanged.

The full records, logs and scheduler audit remain in the ignored local archive
and on Cannon. Public files contain aggregate measurements and protocol details.
For new runs, use the [setup and evaluation commands](../../README.md#cannon),
then compare complete outputs:

```bash
PYTHONPATH=src python -m dqat.summarize --fp /path/to/fp-run --gptq /path/to/gptq-run
```

Both quantizers reconstruct BF16 weights for inference. These results measure
task accuracy under simulated quantization, without packed INT4 speed or memory
claims. Job wall times include loading and simulator work, so they are not
inference-speed measurements. The per-episode reseeding schedule differs from
upstream, and calibration uses original 128px HDF5 demonstrations. This is a
matched nominal comparison with one base seed, not an exact paper reproduction
or evidence of OOD robustness.

The baseline stage is complete. A controlled on-policy GPTQ calibration refresh
is the next proposed comparison, holding the quantizer and calibration budget
fixed and keeping calibration rollouts separate from evaluation. No DAgger-style
update or new inner quantization method has been evaluated yet.
