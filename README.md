# robot-distillation

On-policy distillation of a robot-use reasoning model into smaller, faster students.

## Motivation

[PEAR](https://arxiv.org/abs/2611.18562) trains a 27B reasoning model to control robots, with reinforcement learning
in simulation alone. The model reasons in a chain of thought before every action. At its highest reasoning effort it
solves 74.4% of LIBERO, but each round of thought adds latency on a real robot. Lowering the effort shortens the thought
but costs most of the success: 21.9% at low effort.

This project asks whether a smaller student, trained to imitate the teacher on the states the student itself reaches,
can keep the teacher's success at a fraction of its time per episode. Training on the student's own states (on-policy
distillation) avoids the compounding errors of imitating teacher trajectories alone.

- **Teacher:** [Qwen3.8-27B-PEAR-LIBERO_only](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only)
- **Students:** [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- **Benchmarks:** LIBERO (2,000 episodes: 4 suites × 10 tasks × 50 initial states), LIBERO-Pro and LIBERO-Plus

## Pipeline

| Step | | Script |
| --- | --- | --- |
| 1. Teacher data | Run the teacher on LIBERO's four suites. Keep each successful episode's full multi-round context: images, reasoning and actions. | `scripts/collect.sh` |
| 2. SFT | Fine-tune each student on the teacher's episodes, with the loss on every generated token, in the same context format as at inference. | `scripts/sft.sh` |
| 3. On-policy distillation | Starting from the SFT checkpoint, at each step the student runs LIBERO episodes, the teacher scores every student token, and the student takes one policy-gradient step on the per-token reverse KL. There is no task reward. | `scripts/distill.sh` |
| 4. Evaluation | Measure the success rate of any model on LIBERO, LIBERO-Pro or LIBERO-Plus. | `scripts/eval.sh` |

The project is built on [PEAR](https://github.com/mit-han-lab/pear), which provides the LIBERO environments, the
prompt format, the rollout and the trainer. Distillation uses the on-policy distillation of the
[verl](https://github.com/volcengine/verl) version that PEAR pins.

## Layout

```
configs/
  models/       one file per model: its Hub repository and pinned revision
  hardware/     GPU profiles beyond PEAR's own H100 and GB300 recipes
  *.yaml        one Hydra config per step
scripts/        setup, one script per step, and the Ray launcher (run.sh, ray_node.sh)
src/opd/        entry points, teacher-episode collection and the SFT dataset
third_party/    PEAR, pinned as a submodule
```

## Setup

```bash
git clone --recursive https://github.com/ericjkge/robot-distillation.git && cd robot-distillation
export WORK_DIR=/shared/path       # a directory every machine sees: environments, models and runs
bash scripts/setup.sh              # PEAR's environments (verl, LIBERO, LIBERO-Pro, LIBERO-Plus) and the three models
```

## Running

Every step reads its settings from environment variables, and passes any extra arguments to Hydra as config
overrides. The steps that use Ray (`collect`, `distill`, `eval`) start through `scripts/run.sh`, which works with any
cluster:

- **One machine:** `bash scripts/run.sh <step>` starts Ray locally and gives the machine every role.
- **Slurm:** `sbatch [resources] scripts/run.sh <step>` starts Ray over the job's machines. `ROLES` names each machine's
  role, one per machine in node order: `trainer`, `rollout` or `teacher`.
- **An existing Ray cluster:** start `scripts/ray_node.sh <role> [head address]` on every machine, set `RAY_ADDRESS`
  and `ROLLOUT_NNODES`, `TRAINER_NNODES` and `TEACHER_NNODES`, then run the step.

```bash
M=$WORK_DIR/models R=$WORK_DIR/runs

# 1. teacher episodes, at xhigh effort
MODEL=$M/Qwen3.8-27B-PEAR-LIBERO_only RUN=$R/teacher bash scripts/run.sh scripts/collect.sh

# 2. SFT
MODEL=$M/Qwen3.5-9B EPISODES_DIR=$R/teacher/episodes RUN=$R/sft-9b bash scripts/sft.sh

# 3. on-policy distillation: here three machines, one per role
sbatch --nodes=3 --export=ALL,ROLES="trainer rollout teacher",MODEL_FILE=configs/models/qwen3_5_9b.sh,\
MODEL=<SFT checkpoint>,TEACHER=$M/Qwen3.8-27B-PEAR-LIBERO_only,RUN=$R/opd-9b scripts/run.sh scripts/distill.sh

# 4. evaluation: BENCH=libero (default), libero_pro or libero_plus
MODEL_FILE=configs/models/qwen3_5_9b.sh MODEL=<model> RUN=$R/eval-9b bash scripts/run.sh scripts/eval.sh
```

To evaluate the teacher at another reasoning effort, add the override
`+data.apply_chat_template_kwargs.reasoning_effort=low`. Evaluation writes per-episode records to
`$RUN/val_dumps/eval_<step>.jsonl` and the success rates to `eval_<step>_summary.json`.

**Hardware.** `GPU=h100` or `GPU=gb300` uses PEAR's tuned recipes for those GPUs. The default, `GPU=generic`,
runs on any CUDA GPU (`configs/hardware/generic.sh`) without PEAR's custom kernels. To add a profile, add a file to
`configs/hardware/`.

## Acknowledgments

Built on [PEAR](https://github.com/mit-han-lab/pear), [verl](https://github.com/volcengine/verl),
[vLLM](https://github.com/vllm-project/vllm) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
