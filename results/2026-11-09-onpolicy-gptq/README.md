# Controlled on-policy GPTQ calibration: nominal LIBERO-Long

The pre-set two-round learner-calibration experiment did **not improve** over
fixed-demonstration GPTQ in this run. This is standard LIBERO-Long, not LIBERO-Plus.

| Method / calibration source | Successes | Success rate |
| --- | ---: | ---: |
| Unquantized BF16 reference | 467/500 | 93.4% |
| AWQ, original demonstrations | 467/500 | 93.4% |
| GPTQ, original demonstrations | 465/500 | 93.0% |
| GPTQ, learner rollouts + demo anchors, round 2 | 464/500 | 92.8% |
| GPTQ, BF16-policy rollouts + demo anchors, round 2 | 474/500 | 94.8% |

| Paired comparison | Difference | Descriptive bootstrap 95% interval |
| --- | ---: | ---: |
| Learner calibration minus offline GPTQ | −0.2 pp | [−2.4, +2.0] pp |
| BF16-rollout control minus offline GPTQ | +1.8 pp | [−0.2, +4.0] pp |
| Learner calibration minus BF16-rollout control | −2.0 pp | [−4.0, −0.2] pp |

The learner arm wins 16 episodes and loses 17 against offline GPTQ. The control
wins 19 and loses 10 against offline GPTQ. These are exploratory comparisons from
one calibration seed and one fixed set of tasks/states; the intervals resample
matched episodes, not independent training seeds or task populations. They do not
establish a general benefit for the control or rule out other on-policy methods.

## Protocol and validation

See [the predeclared protocol](../../docs/onpolicy-gptq-protocol.md). Each of two
rounds collects two fresh episodes per task, excludes evaluation initial physical
configurations, and recalibrates unchanged GPTQ from the original BF16 checkpoint.
The 32-observation budget is 16 fixed demo anchors plus 16 rollout observations;
round 2 uses eight observations from each collection round. The learner controls
its own rollouts; the matched control uses BF16 rollouts. No teacher action labels
or Plus observations enter calibration. This is DAgger-style data aggregation,
not action-supervised QAT.

Both full evaluations completed with zero recorded Python errors. All 1,000 new
episode identities, seeds, task names, action counts, summaries and log records
were checked against the original paired 500-episode protocol. The strict
comparator reproduced the archived aggregate exactly. Source was clean at
`e09670c071d4a3638e6b9e5eb92aa59860f0c313`; all 37 tests also passed on the actual
Cannon runtime. All four collection/calibration stages completed successfully.

One MuJoCo instability warning occurred in the same failed episode in all five
arms. Retaining it gives the primary scores above. Excluding that shared pair
gives learner GPTQ 464/499 (93.0%), offline GPTQ 465/499 (93.2%), and control
474/499 (95.0%); the interpretation is unchanged. Full sensitivity calculations,
per-task counts and artifact hashes are in [comparison.json](comparison.json).

The checkpoint, GPTQ settings and protected modules match the
[offline experiment](../2026-11-09-openvla-oft-gptq/README.md). These are decoder-only
W4/g128 simulated-quantization accuracy measurements with BF16 execution, not
packed INT4 speed or memory results. Raw records, calibration manifests and
cluster metadata remain in the private local/Cannon archives.
