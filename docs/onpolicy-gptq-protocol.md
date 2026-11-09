# First GPTQ data-aggregation experiment

This protocol is fixed before running the new evaluations. The reference is the
completed offline GPTQ result: **465/500 (93.0%)** on nominal LIBERO-Long.

The question is whether updating calibration inputs toward states visited by the
quantized policy improves the same GPTQ inner method. This is a DAgger-style
**data-aggregation calibration loop**, not action-supervised DAgger or QAT:
GPTQ uses reconstruction statistics, and no teacher action labels enter its
update. Rebuild from the original BF16 checkpoint every round; retain W4/group
128, natural column order, asymmetric grids, 128-column blocks, 1% damping,
true-sequential calibration and the protected non-decoder modules.

## Arms and pre-set rounds

| Arm | Controller during round 1 | Controller during round 2 | Final comparison |
| --- | --- | --- | --- |
| Offline GPTQ | No new collection | No new collection | Existing 93.0% result |
| Student collection | Existing offline GPTQ policy | Round-1 recalibrated GPTQ policy | Round-2 GPTQ |
| Expert collection control | Original BF16 policy | Original BF16 policy | Round-2 GPTQ |

Use two rounds per new arm, with 20 collection episodes per round (two per task).
There is no expert/learner action mixing. Both arms share collection reset seeds,
episode caps, preprocessing, reservoir size, demo anchors and selection rules.
Actual action/teacher-call counts can differ with trajectory lengths; retain
these counts and total wall times rather than claiming identical compute costs.

Collection uses fresh simulator resets seeded by
`100000 + 10000 * round + 100 * task_id + episode_id`. Before taking actions,
compare the simulator's physical initial configuration (`qpos`, ignoring time
and velocity) against every stored evaluation initial state, using zero relative
tolerance and absolute tolerance `1e-8`. Reject overlap, nonfinite states or any
MuJoCo warning. Also reject repeated collection configurations within an arm,
including the prior round. The stored evaluation states are used only for this
exclusion check; no evaluation images, actions, successes or failures enter the
calibration selection.

Collect observations immediately before each eight-action chunk. Use the same
256px rendering and upstream observation preparation as evaluation. Keep 10
settling steps and a 520-action cap. Valid unsuccessful trajectories contribute
observations under the same sampling rule as successful trajectories. Instability
and Python errors abort collection instead of contaminating the dataset.

## Fixed observation budget and aggregation

Each task has a uniform reservoir of eight candidate observations per round,
sampled across its two trajectories using an independent RNG. Save all 80
reservoir observations per round; aggregate both pools for round 2. Only 32
observations enter each GPTQ calibration:

- **16 fixed demo anchors:** a deterministic subset of the original 32 baseline
  demo observations, retained unchanged in every arm and round.
- **16 rollout observations:** all from round 1 for the first update; eight from
  round 1 and eight from round 2 for the second update.
- Preserve the baseline's total task counts: `[4, 4, 3, 3, 3, 3, 3, 3, 3, 3]`.
  Demo quotas are `[2, 2, 2, 2, 2, 2, 1, 1, 1, 1]`; rollout quotas are their
  complements. Alternate source rounds while traversing the rollout quota,
  then select observations without replacement within each task/source pool.

Hashes bind the observation arrays, manifest, actor checkpoint, source policy
cache and round lineage. NPZ storage preserves different original-demo and
rollout image resolutions. Both new arms undergo the same preprocessing; the
expert-collection control helps account for fresh data and image-source changes
relative to the original HDF5 calibration.

## Evaluation and interpretation

Each preparation stage calibrates a fresh BF16 model and runs one pipeline-check
episode. Pipeline checks do not select mixtures, rounds or hyperparameters.
Evaluate **only the pre-set final round** on the original 500 paired episodes in
each new arm. Thus this experiment adds 1,000 full evaluation episodes and 80
collection episodes across both arms. The original evaluator's seeds, initial
states, preprocessing, action handling and horizon remain unchanged.

Report student minus offline GPTQ, expert-control minus offline GPTQ, and student
minus expert-control, with paired success/failure counts and descriptive 95%
bootstrap intervals (10,000 resamples, seed 7). A larger point estimate alone
does not establish a reliable gain. The expert-control comparison is needed to
attribute improvement specifically to learner-induced observations. Audit
simulator warnings separately and retain the raw 500-episode scores alongside
any matched exclusion sensitivity. This remains one calibration experiment and
one base-seed nominal evaluation with simulated BF16 inference.

## Running on Cannon

Use a new scratch experiment directory and the existing pinned environment. Set
`DQAT_BASELINE_GPTQ_RUN` to the completed baseline's `run.json`; the corresponding
cache is `$DQAT_DATA_DIR/calibration/gptq-w4g128-32frames-seed7.pt`.

```bash
export DQAT_BASELINE_GPTQ_RUN=/path/to/completed/offline-gptq/run.json
experiment="$DQAT_DATA_DIR/experiments/onpolicy-gptq-v1"
test_job=$(sbatch --parsable scripts/cannon_test.sbatch)
for arm in student expert_control; do
  first=$(sbatch --parsable --dependency="afterok:$test_job" \
    scripts/cannon_loop.sbatch prepare "$arm" 1 "$experiment")
  second=$(sbatch --parsable --dependency="afterok:$first" \
    scripts/cannon_loop.sbatch prepare "$arm" 2 "$experiment")
  sbatch --dependency="afterok:$second" --time=03:00:00 \
    scripts/cannon_loop.sbatch evaluate "$arm" 2 "$experiment"
done
# After both full evaluations complete:
sbatch scripts/cannon_compare_loop.sbatch "$experiment"
```

Select an A100-SXM4-80GB explicitly for the matched experiment. Record job IDs,
scratch paths, exact source commit and logs in ignored local data. Public reports
contain aggregate results and protocol details. Failed or preempted stages retain
their files and block downstream jobs; never report a partial run as a score.

The outer-loop motivation comes from [DAgger](https://proceedings.mlr.press/v15/ross11a.html);
the unchanged reconstruction update follows [GPTQ](https://arxiv.org/html/2210.17323v2).
DAgger's imitation-learning guarantees do not automatically apply to this update.
