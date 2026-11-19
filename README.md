# robot-distillation

On-policy distillation of PEAR's robot-use reasoning model into smaller students, to cut reasoning latency on robots.

- **Teacher:** [Qwen3.8-27B-PEAR-LIBERO_only](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only), from
  [Training Reasoning Models in Simulation to Natively Use Real Robots](https://arxiv.org/abs/2611.18562).
- **Students:** [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), then [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- **Infrastructure:** [PEAR](https://github.com/mit-han-lab/pear), a git submodule at `third_party/pear` pinned to a
  commit. We reuse its LIBERO environments, prompt format, rollout and trainer, and the verl version it pins, which
  already implements on-policy distillation. This repository adds configuration and three small plugins.

## Pipeline

| Step | What | Script | Entry point |
| --- | --- | --- | --- |
| 1 | Teacher data: run the teacher on LIBERO's 4 suites and keep the successful episodes' full multi-round contexts | `scripts/collect.sh` | `opd.collect` |
| 2 | SFT: train each student on the teacher's episodes, with the loss on every generated token and the inference context format | `scripts/sft.sh` | `opd.sft` |
| 3 | On-policy distillation from the SFT checkpoint: per-token reverse KL to the teacher, no task reward, evaluated every 10 steps | `scripts/distill.sh` | `opd.distill` |
| 4 | Evaluation of any model, teacher at either effort or a student, on LIBERO (2,000 episodes), LIBERO-Pro and LIBERO-Plus | `scripts/eval.sh` with `BENCH=...` | `opd.eval` |

The evaluation protocol is PEAR's: 2,000 LIBERO episodes (4 suites x 10 tasks x 50 initial states). Records go to
`$RUN/val_dumps/eval_<step>.jsonl`, and the success rate to `eval_<step>_summary.json`.

## Layout

```
configs/
  models/        one file per model: its Hub repo and pinned revision, plus PEAR's Qwen3.5-family arguments
  hardware/      o2.sh: HMS O2's A100 and L40S machines, which PEAR's H100/GB300 recipes do not cover
  eval.yaml      Hydra configs, one per step, composed on top of PEAR's and verl's
  collect.yaml
  sft.yaml
  distill.yaml
scripts/
  common.sh      paths, and the PEAR run arguments shared by every step
  setup.sh       one-time install of PEAR's environments and the three models
  slurm.sbatch   one step as a Slurm job on O2: a Ray cluster over the job's machines, then the step
  eval.sh  collect.sh  sft.sh  distill.sh
src/opd/
  eval.py  distill.py  sft.py   thin Hydra entry points onto PEAR's and verl's runners
  collect.py                    PEAR's evaluation trainer, which also saves successful episodes as token ids + images
  sft_data.py                   verl's SFT dataset contract over those episodes
tests/
third_party/pear                pinned submodule
```

## Setup

```bash
git clone --recursive <this repo> && cd robot-distillation
export WORK_DIR=<a directory every node sees>   # environments, models and caches go here
bash scripts/setup.sh                            # PEAR's verl, LIBERO, LIBERO-Plus and LIBERO-Pro environments, and the models
```

`setup.sh` puts the models in `$WORK_DIR/models/<name>`. Every run happens on the head of a Ray cluster that was
started with `third_party/pear/scripts/start_ray_node.sh`; see PEAR's
[quick start](third_party/pear/docs/getting_started/quickstart.md).

## Running

```bash
export WORK_DIR=... GPU=h100 RAY_ADDRESS=<head>:6379
M=$WORK_DIR/models

# 1. teacher episodes (xhigh effort)
ROLLOUT_NNODES=8 MODEL=$M/Qwen3.8-27B-PEAR-LIBERO_only RUN=$WORK_DIR/runs/teacher bash scripts/collect.sh
# 2. SFT
MODEL=$M/Qwen3.5-9B EPISODES_DIR=$WORK_DIR/runs/teacher/episodes RUN=$WORK_DIR/runs/sft-9b bash scripts/sft.sh
# 3. on-policy distillation
TRAINER_NNODES=1 ROLLOUT_NNODES=2 TEACHER_NNODES=2 MODEL_FILE=configs/models/qwen3_5_9b.sh \
  MODEL=<SFT checkpoint, exported to HF> TEACHER=$M/Qwen3.8-27B-PEAR-LIBERO_only RUN=$WORK_DIR/runs/opd-9b bash scripts/distill.sh
# 4. evaluation, e.g. LIBERO-Pro
BENCH=libero_pro ROLLOUT_NNODES=4 MODEL_FILE=configs/models/qwen3_5_9b.sh MODEL=<model> RUN=... bash scripts/eval.sh
```

On O2, a step runs as a Slurm job that brings up its own Ray cluster; the variables above go through `--export`:

```bash
sbatch --export=ALL,MODEL_FILE=configs/models/qwen3_5_9b.sh,MODEL=$M/Qwen3.5-9B,RUN=$WORK_DIR/runs/base-9b \
  scripts/slurm.sbatch scripts/eval.sh
# distillation on one 4-GPU machine: the GPUs split between the roles
sbatch --export=ALL,TRAINER_GPUS=2,ROLLOUT_GPUS=1,TEACHER_GPUS=1,TEACHER_TP=1,... scripts/slurm.sbatch scripts/distill.sh
```

`scripts/slurm.sbatch` explains how to size a job's request so that it does not cost fairshare priority.

Every script passes its extra arguments to Hydra as overrides. For example,
`+data.apply_chat_template_kwargs.reasoning_effort=low` evaluates the teacher at low effort.

## Not yet verified

Nothing has run end to end yet. These are the points a first pilot has to check:

- **Chat template:** the teacher and the students tokenize LIBERO text identically. They share the vocabulary, the
  merges and the normalizer; the teacher only adds audio/TTS tokens, and the pre-tokenizers differ only on Unicode
  combining marks. Their chat templates differ, though. The teacher's adds a reasoning-effort system instruction and
  keeps every round's reasoning in the context. Qwen3.5's adds no effort instruction and drops reasoning from earlier
  turns. The students probably need the teacher's `chat_template.jinja`.
- **Collected fields:** the transfer-queue field names `opd.collect` reads (`response_mask`, `multi_modal_data`).
- **Image processing:** whether `opd.sft_data`'s image processing matches what the rollout fed the model.
- **O2 hardware:** PEAR is tuned for H100 and GB300. `configs/hardware/o2.sh` turns off its Hopper/Blackwell kernels,
  FlashAttention-3 and LMCache. That configuration, and `scripts/slurm.sbatch`'s Ray cluster, have not run yet.
- **Distillation config:** that PEAR's separate-async trainer runs verl's distillation, with score centering and
  the DAPO filter turned off in `configs/distill.yaml`.
