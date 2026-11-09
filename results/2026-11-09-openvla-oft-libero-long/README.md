# OpenVLA-OFT / LIBERO-Long — 2026-11-09

The completed paired evaluation gives the same aggregate success rate for the
unquantized model and decoder-only AWQ:

| Method | Successes | Episodes | Success rate |
| --- | ---: | ---: | ---: |
| Unquantized BF16 | 467 | 500 | 93.4% |
| AWQ W4, group 128, simulated in BF16 | 467 | 500 | 93.4% |

AWQ minus FP is **0.0 percentage points**. The descriptive paired bootstrap 95%
interval is **[-2.2, +2.2] pp** (10,000 resamples, bootstrap seed 7). There are
452 shared successes, 15 FP-only successes, 15 AWQ-only successes, and 18 shared
failures. Equal totals do not imply identical behavior or establish equivalence.

Both jobs completed with exit code 0:0, zero recorded Python exceptions and zero
restarts. All 1,000 episode records were checked for complete task/state/seed
pairing, unique identities, valid success flags, action/call counts, and agreement
with the summaries. Timeouts at 520 actions are recorded as unsuccessful episodes.

**Follow-up simulator audit:** while preparing GPTQ, the simulator log revealed a
MuJoCo acceleration-instability warning in one shared episode, which both methods
counted as unsuccessful. This warning did not raise a Python exception and was
not annotated in the original per-episode records. The raw 500-episode scores
above are retained; excluding that single flagged pair gives **467/499 (93.6%)
for each method**, with the same 0.0-point difference. Warning-affected episodes
should be examined separately from ordinary policy failures in subsequent runs.

## Per-task successes

Each row contains the same 50 initial states in each arm.

| ID | Task | BF16 | AWQ | AWQ − FP (pp) |
| ---: | --- | ---: | ---: | ---: |
| 0 | Alphabet soup and tomato sauce into basket | 45/50 | 47/50 | +4.0 |
| 1 | Cream cheese and butter into basket | 50/50 | 50/50 | 0.0 |
| 2 | Turn on stove and put moka pot on it | 49/50 | 49/50 | 0.0 |
| 3 | Black bowl into bottom drawer, then close | 50/50 | 50/50 | 0.0 |
| 4 | White mug on left plate; yellow/white mug on right | 47/50 | 45/50 | −4.0 |
| 5 | Book into back compartment of caddy | 46/50 | 46/50 | 0.0 |
| 6 | White mug on plate; chocolate pudding to its right | 45/50 | 45/50 | 0.0 |
| 7 | Alphabet soup and cream cheese into basket | 49/50 | 48/50 | −2.0 |
| 8 | Both moka pots on stove | 40/50 | 42/50 | +4.0 |
| 9 | Yellow/white mug into microwave, then close | 46/50 | 45/50 | −2.0 |

## Protocol and interpretation

This is one base-seed protocol on nominal `libero_10`, with 10 tasks × 50 initial
states. Both arms use the same official OFT checkpoint, L1 action head, two images,
proprioception, 90% center crop, 256px simulator observations, eight-action chunks,
10 settling steps and a 520-action horizon. See the [baseline protocol](../../README.md#baseline-protocol).

Before every episode, global policy RNGs are seeded with
`7 + 1000 * task_id + initial_state_id`, then `env.seed(0)` resets the process-wide
NumPy RNG to 0. Python/Torch retain the per-episode seed. This reseeding schedule
differs from upstream's once-per-run global seed and once-per-environment seed.
These results compare the two arms under our paired protocol; they are not an
exact reproduction of upstream's three-seed paper result.

AWQ quantizes the seven linear projections in each of the 32 Llama decoder blocks
using asymmetric integer zero points, activation-aware scale search and clipping.
The vision backbone, multimodal/proprio projectors, embeddings, LM output and
action head stay unquantized. Calibration uses 32 task-balanced frames from the
pinned original 128px LIBERO training HDF5s, rather than OFT's regenerated RLDS
data. Exact selections are retained in the local run archive.

Weights are quantized/dequantized and stored in BF16, so these are task-accuracy
results, not packed INT4 memory or kernel-speed measurements. The bootstrap
interval describes this episode sample, without accounting for variation across
training/calibration seeds or tasks. No OOD conclusion follows from nominal
LIBERO success. GPTQ, a DAgger-style update and perturbed evaluations remain next
experiments; the [research notes](../../docs/research-notes.md) describe controls.

## Reproducibility

Both full evaluations used NVIDIA A100-SXM4-80GB GPUs on Harvard Cannon. Source
checkout was clean at `ef1fa1608b8299b44307f1ae6c1ca2167decd246`; later commits add
documentation and these results. The pinned runtime uses Python 3.10.14 and
Torch 2.2.0+cu121. BF16 took about 1h32m and AWQ about 1h28m, plus about nine
minutes for AWQ calibration and a smoke rollout. These are total job wall times,
including loading and environment work, rather than inference-speed measurements.

- Checkpoint: `moojink/openvla-7b-oft-finetuned-libero-10`, revision
  `95220f9a3421a7ff12d4218e73d09ade830fa9a3`.
- Calibration data: `yifengzhu-hf/LIBERO-datasets`, revision
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- OpenVLA-OFT submodule: `e4287e94541f459edc4feabc4e181f537cd569a8`;
  LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Transformers fork: `bc339d9ad707454c0c115970db43c260067c61ab`.
- Calibration artifact SHA256:
  `af08ac980fc3276d820baa93ad054509b51c831469ef4fee9a71a4388a10e7e6`.
  The full run reused the validated artifact, and its hash was independently
  rechecked after completion.

The public [comparison](comparison.json) contains aggregate counts and the paired
bootstrap interval. Full episode records, calibration selections, environment
inventory and scheduler records are retained in the local workspace and on
Cannon. Use the pinned setup and evaluation commands in the root README to
generate a new pair of runs, then compare their output directories:

```bash
PYTHONPATH=src python -m dqat.summarize --fp /path/to/fp-run --awq /path/to/awq-run
```

Implementation validation: 20 local CPU tests passed using the pinned Transformers
fork; 18 numerical/integration tests passed on Cannon's exact runtime before the
two final comparator checks were added. Ruff and shell/diff checks passed. Both
GPU smoke episodes succeeded, and replaying the saved AWQ artifact reproduced its
smoke rollout before the full benchmark.
