# LIBERO-Plus Long pilot v1

This is an exploratory 70-case pilot, not the full 2,519-variant Plus Long suite
or the full 10,030-task, four-suite leaderboard. It tests whether calibration
learned on nominal LIBERO transfers to perturbations. It does not fit on Plus.

## Fixed cases and methods

`configs/libero-plus-long-pilot-v1.json` selects one official variant for every
combination of the 10 original Long tasks and seven perturbation categories.
Selection uses public metadata before any pilot outcome is observed: rotating
1–5 difficulty targets, nearest available difficulty, and a SHA256 tie-break
seeded with 7. Sensor noise covers all five official operators twice, including
glass blur. Author-provided difficulty labels need not be equally available in
each task/category. The resulting difficulty counts are 10, 15, 18, 16 and 11.
This deliberately balanced pilot is not a probability sample of the full suite.

All five arms use the same cases, initial-state index 0 and per-case seed
`7 + official zero-based task ID`: BF16, AWQ, offline GPTQ, round-2 learner GPTQ,
and round-2 BF16-rollout-calibrated GPTQ. Quantizer artifacts must match hashes
and signatures from their completed nominal evaluations. No cache is fitted or
changed here. Quantized execution remains simulated W4/g128 in BF16.

The upstream Plus implementation is pinned to
`4976dc30028e805ff8094b55501d532c48fec182`; the official asset dataset to
`dd2bd61b7d9a6fef1abc52d606e983b41886a149`. The downloaded 6.4 GB ZIP is SHA256
verified. Its 457,675 entries have a nested `assets/` prefix. Setup extracts shared
assets and uses a CPU environment-construction pass to materialize the exact
scene/object files needed by the fixed pilot, recording their hashes. File bytes
are unchanged; the extraction hooks are used only during setup. Plus dependencies
and configuration are isolated from nominal LIBERO.

## Evaluation details

Retain the same checkpoint, normalization, 256-pixel simulator images, two image
inputs, proprioception, center crop, eight-action chunks, ten settling actions
and 520-action horizon as nominal Long. Use the official Plus environment and
initial-state loaders with one trial per variant. Reseed before environment
construction and again before reset, making shard order irrelevant.

Read the actual instruction from the constructed environment's BDDL. Plus's
filename-derived `task.language` includes perturbation suffixes for some
categories; passing those suffixes would accidentally change the language input
when testing visual changes. Intentional language-category instructions still
come from their perturbed BDDL. This explicit instruction handling and per-case
seeding mean this is our paired protocol, not an exact leaderboard reproduction.

Track finite simulator state and MuJoCo warning counters throughout each episode.
Infrastructure exceptions abort a shard and are never counted as task failures.
Keep warning-flagged episodes in primary raw scores, and separately exclude the
union of flagged cases from every paired arm for a sensitivity comparison.

## Parallelism and reporting

Each arm has four deterministic case shards (18, 18, 17, 17 cases): 20 GPU jobs,
with a requested concurrency cap of 20 A100 80 GB GPUs after the user requested a
30-minute turnaround. Each shard gets a
private copy of mutable checkpoint config/Python files and its own module cache;
large model tensors remain shared read-only files. This removes model-loading
serialization without changing model weights. A seven-category smoke includes
slow glass blur and uses eight action steps after settling before the array starts.
Scored cases retain the full 520-step horizon. Pipeline smoke outcomes do not select
cases, artifacts or hyperparameters.

Target less than an hour of GPU runtime after setup, subject to measured smoke
throughput. Queue wait, repository checkout and the first asset download are
additional; no strict wall-clock guarantee is possible on shared Cannon.

Report complete-case totals, per-category counts and paired differences:
learner versus offline GPTQ, learner versus BF16-rollout control, and quantized
arms versus BF16. Reject missing/duplicate cases or mismatched protocol, artifact,
seed, instruction or initial-state identity. With only ten cases per category,
these results are directional; confidence intervals are descriptive, not evidence
of an across-seed or across-task population effect. Keep complete raw records and
cluster metadata in private archives; publish only aggregates and public protocol.

Sources: [official repository](https://github.com/sylvestf/LIBERO-plus),
[paper](https://arxiv.org/abs/2510.13626).

## Urgent 14-case subset

Before any Plus outcomes were observed, a 4:30 p.m. deadline prompted a smaller
parallel attempt on available full Ampere GPUs (A100, A40 or RTX A6000). For category index `c` in the frozen manifest order,
select base tasks `(2*c)%10` and `(2*c+1)%10`. This gives two cases per category,
covering all ten base tasks, and reuses their exact seeds and original 520-action
horizons. The subset has only zoom-blur and fog cases for sensor noise; it does
not represent all five noise operators or the full Plus task distribution.

Each of the five methods gets one 14-case job. GPU models may differ across arms
in this urgent subset and are reported per arm; it is an exploratory check, with
the larger A100-only pilot providing the planned hardware control. Its protocol
is `libero-plus-long-rapid14-v1`, and its outputs/aggregate are separate from the
70-case A100 pilot. Fourteen trials are too few for a precise success-rate estimate;
report exact counts and broad uncertainty, and do not substitute these scores for
a full benchmark or combine them with the 70-case run.

## Separate software-rendered MIG pilot

To use Cannon's test-GPU queue while full GPUs remain queued, a separate
`libero-plus-long-rapid14-osmesa-v1` protocol reuses the same 14 cases and frozen
artifacts. Every arm uses an A100 MIG 3g.20gb and OSMesa/llvmpipe rendering with
four CPU rendering threads. `scripts/setup_osmesa.sh` extracts the signature- and
SHA256-verified Rocky Mesa 23.1.4 library into the project; the existing Python
environment is unchanged. Evaluation checks the library hash and renderer settings.

Run a seven-category model/rendering smoke before submitting
`scripts/cannon_plus_mig.sbatch` as array `0-4%2`, with AWQ, offline GPTQ and
learner GPTQ ahead of the reference arms. The comparator requires
`--rapid --software-rendering` and matching GPU models. Never pool software-rendered
episodes with EGL episodes or interpret a cross-renderer difference as a quantizer
effect. These are exploratory software-rendered results, not the official benchmark.
