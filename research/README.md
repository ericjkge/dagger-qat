# Noise-Robust Flow Maps

<!-- TODO(human): state the durable project goal in a paragraph: what a
noise-robust flow map is, what "robust" is measured against (the corruption
schedule in src/corrupt.py), and which data it must work on (toy
distributions in src/toy.py, CelebA, ImageNet, AFDB). -->

This note holds the durable project goals, the criteria for choosing
experiments and the standards of evidence. See [`README.md`](../README.md) for
implementation usage and [`CLAUDE.md`](CLAUDE.md) for the operating rules every
agent follows.

# Criteria for choosing experiments

<!-- TODO(human): what makes a method worth testing here. -->

Before training a new method, state the mechanism, its cost, one falsifiable
prediction and a bounded test. Start on the toy distributions before scaling
to images. Keep diagnostics distinct from candidate methods, and do not treat a
better number on one benchmark as sufficient progress.

# Standards of evidence

A claim of improvement requires more than one seed and a stated comparison
against the baselines in [`baseline.md`](baseline.md), under its protocol and
matched compute, not matched steps. A difference inside seed variance is not a
result; say so and run more seeds.

State the kill criterion of a direction before its first run and do not weaken
it after seeing data. Replace superseded conclusions in this note and in
subject notes rather than annotating them.

# Directions

Each direction is specified by the human in `directions/<slug>.md`: its
question, mechanism, first experiment and ideas to explore. That note is the
direction's brief. One agent holds each direction on branch `claude/<slug>`
and keeps its own account in `experiments/<slug>/README.md`. A brief marked
`status: active` is started by `bin/agent-host.sbatch`; `status: paused` is not.

# Retired directions

A direction that meets its kill criterion is recorded here; its branch stays
as the record. Retained negative results exist so the question is not reopened
in three months.

| idea | killed | why |
|---|---|---|

# Layout

`bin/` holds the orchestration scripts, documented in
[`bin/README.md`](bin/README.md). `directions/<slug>.md` holds the human's
brief for each direction. `baseline.md` holds the baselines and the protocol
every direction is compared under. `templates/` holds the experiment note
template.
`experiments/<slug>/` is agent-owned. `runs/<slug>.jsonl` is append-only and
written by training code, never by hand. `dashboard/` is generated locally and
is not tracked.
