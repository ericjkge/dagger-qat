# Baselines

Every comparison in every direction is made against the baselines below,
under the protocol below. A comparison against any other configuration, budget
or metric is not a result.

<!-- TODO(human): fill in once scripts/train.py and configs/ exist. -->

| name | role | overrides |
|---|---|---|
| `flowmap-clean` | standard flow map trained on clean data | TODO |
| `flowmap-corrupt` | standard flow map trained on corrupted data, no robustness mechanism | TODO |

## Protocol

- TODO: the command, e.g.
  `python scripts/train.py --config configs/<data>.yaml <overrides> seed=<seed>`,
  with seeds 42, 43 and 44 and a fixed number of updates.
- Primary metrics: TODO (e.g. FID or energy distance from `scripts/eval.py`,
  sample count, number of sampling steps). Report mean ± SE over seeds.
- Match compute by training FLOPs or forward passes, not by updates or steps;
  state both counts and wall-clock time for the candidate and the baseline.

## Reference results

None recorded yet. Until they are, run the baseline you compare against under
this protocol and report its numbers beside yours, with job IDs.
