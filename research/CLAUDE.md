# Noise-Robust Flow Maps

## Research direction

`research/README.md` holds the project goals, the criteria for choosing
experiments and the standards of evidence; they bind every direction. This file
holds the operating rules.

You hold one direction, named by the part of your branch after `claude/`. The
human's brief for it is `research/directions/<slug>.md`, and the baselines every
direction is judged against are in `research/baseline.md`. At the start of each
working turn, read the current version of both with `git fetch origin` and
`git show origin/eddy:<path>`, since your branch's copies may be stale, and
treat changes to either as direct instruction from the human. Every claimed
result compares against those baselines under their protocol; a comparison
against anything else is not a result. You write
`research/experiments/<slug>/` and `research/runs/<slug>.jsonl`; the rest of the
repository is read-only to you. A change needed in shared model, data or
evaluation code is proposed under `## Blocked on` in your note and not made.
Do not edit `research/README.md`, `research/baseline.md`, any brief under
`research/directions/`, another direction's note or another direction's runs
file.

## Implementation

Keep the implementation small: use one direct path for each operation, retain
abstractions only where they remove real duplication, and do not add
compatibility layers, speculative checks, fallback input formats or unused
options. Extend existing model, data and evaluation modules in `src/` only for a
selected scientific need.

<!-- TODO(human): project conventions the agent must preserve, e.g. the time
convention (which end is data, which is noise), the flow-map
parameterization, the corruption schedule in src/corrupt.py, precision. -->

Commit each completed working turn with a plain message, before reporting back.
An uncommitted note is invisible: the dashboard reads these files out of the
pushed branch.

```sh
git add research/experiments/<slug> research/runs/<slug>.jsonl
git commit -m "<slug>: <what changed>"
git push origin HEAD
```

## Research workflow

The execution interfaces are `scripts/train.py`, `scripts/sample.py`,
`scripts/eval.py` and `scripts/make_data.py`, configured by `configs/*.yaml`.
<!-- TODO(human): how to run and inspect a config once these exist. -->
Set update counts, sample counts and evaluation milestones explicitly; script
defaults do not authorize a budget.

Every job goes through `research/bin/submit <slug> <script.sbatch>`, called
by that relative path. Submissions are approved automatically; do not wait
for the human. The script refuses a job without `--time`, a job over its
time or GPU limit, a job array, and a job beyond the limit on jobs queued or
running per direction. Those limits are the compute budget: fit the
experiment within them, prefer short pilot runs, and never split one run
into chained jobs to exceed them. When a refusal blocks a needed run, record
it under `## Blocked on` rather than working around it. Do not call `sbatch`
directly. Record each submission's `#SBATCH` header and job ID under current
state so the human can audit it afterwards. Nothing runs on a login node —
not a short test, not a sanity check. The environment is conda `vla`
(Python 3.11), activated inside the job script. Job output, checkpoints
and large artifacts go under `/n/scratch/users/e/edk005/vla/<slug>/`, never
into Git. You may cancel a job you submitted for your direction that has
failed or become pointless; never cancel any other job.

The notebook entrance is `research/README.md`; your own current account is
`research/experiments/<slug>/README.md`. Keep its frontmatter accurate,
particularly `status: running | promising | blocked | dead`, and keep
its sections in order: hypothesis, one falsifiable sentence; kill criterion,
fixed before the first run and never weakened afterward; current state;
results, whose numbers come from `runs/<slug>.jsonl`; next step, one concrete
action; blocked on, only when you need the human; dead ends, one line each.
Replace superseded conclusions rather than annotating them. It is a current
account, not a diary. Do not create another status manifest, launch, review or
comparison wrapper, registry, packaging chain or per-run freeze system; the
orchestration under `research/bin/` already covers this and is not yours to
extend.

Keep unique scientific material once, and verify a retained copy before
deleting a sole source or result. Preserve unrelated dirty work.

A change to your brief, or a message the human types into your session, is a
direct instruction and takes priority over your current plan; record the
redirection under current state so the change of course remains visible.
Report the metric you measured, not the one expected. If the kill criterion is
met, say so and set `status: dead` — a clean negative result is a completed
direction. Never fabricate a number, a citation or a run that did not happen,
and do not silently revise a mathematical claim while reporting on it.

Housekeeping authorizes no experiments: do not submit, retry, recover, cancel
or rerun jobs outside the direction you hold and the human's current request.
