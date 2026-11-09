# dagger-qat

DAgger-style quantization-aware training for VLAs: train the quantized policy on states it visits itself, labeled by the full-precision policy, to reduce error accumulation.

**Current step:** baselines — OpenVLA-OFT on LIBERO-Long, full precision vs. AWQ.

## Layout
- `src/dqat/policy.py` — load OFT checkpoint, predict actions
- `src/dqat/awq.py` — AWQ (simulated quantization)
- `src/dqat/eval_libero.py` — LIBERO-Long eval
- `src/dqat/calibration.py` — task-balanced training-demo calibration
- `src/dqat/summarize.py` — validate and compare paired runs
- `scripts/` — Linux setup and Cannon Slurm launchers
- `third_party/` — openvla-oft, LIBERO (submodules)
- [Research notes](docs/research-notes.md) — literature, on-policy/OOD hypotheses and proposed ablations

## Baseline protocol

Use `moojink/openvla-7b-oft-finetuned-libero-10` on `libero_10` (LIBERO-Long).
The **FP baseline is unquantized BF16**, matching upstream, rather than all-FP32.
Both arms use the L1 action head, two images, proprioception, 90% center crop,
eight actions per policy call, 10 settling steps, and a 520-action horizon.
The default evaluation covers all 10 tasks × 50 initial states, seed 7. Episodes
are paired across arms; the policy seed is `7 + 1000 * task_id + initial_state_id`
and the environment seed is fixed to 0, as upstream requires. This per-episode
seeding differs from upstream's single initial RNG seed and is recorded explicitly.

AWQ uses **W4, group size 128, asymmetric integer zero points**, with a 20-point
activation-aware scaling search and per-group clipping. It quantizes the seven
Llama linear projections per decoder block. The vision backbone, multimodal
projector, embeddings, LM output, proprio projector and action head remain
unquantized. We adapt the official MIT HAN Lab Llama algorithm, retain its
license in `licenses/llm-awq.txt`, and use float32 search/reduction arithmetic for
stability. Search handles variable-length multimodal inputs with the actual
attention masks. Quantize/dequantize weights are stored in BF16; this experiment
measures task success, **not packed INT4 memory or kernel speed**.

Calibration uses 32 frames, balanced across all 10 tasks, from the pinned original
LIBERO training HDF5 demonstrations. The exact task/demo/frame selections are
saved. These are the original 128px demonstrations, not OFT's regenerated RLDS
training data. Evaluation observations and outcomes are never used for calibration.
`--calibration-samples` can change the budget; use a different `--awq-cache` path
for each configuration. A cache mismatch is an error, never silently accepted.

The checkpoint, dataset, Transformers fork, OpenVLA-OFT and LIBERO revisions are
pinned. `requirements-runtime.txt` pins direct dependencies; setup saves the
resolved transitive versions in `$DQAT_DATA_DIR/runtime-freeze.txt`.

## Cannon

From a **new checkout** (keep the previous π0.5 environment separate):

```bash
git clone --recurse-submodules https://github.com/ericjkge/dagger-qat.git
cd dagger-qat
export DQAT_DATA_DIR=/path/to/your/scratch/dagger-qat
mkdir -p data/logs "$DQAT_DATA_DIR"
# Provide uv on PATH. The sbatch wrapper also checks the prior project's tools directory.
setup_job=$(sbatch --parsable scripts/cannon_setup.sbatch)
sbatch --dependency="afterok:$setup_job" --time=00:45:00 \
  scripts/cannon_eval.sbatch fp --tasks 0 --episodes 1 --videos
```

Setup runs on a CPU compute node, creates `.venv-runtime` with Python 3.10.14,
installs PyTorch 2.2.0/CUDA 12.1 and the custom Transformers fork, checks model
and simulator imports, and downloads weights and training demonstrations. It
sets a project-local LIBERO config without an interactive prompt. Use scratch
storage with sufficient room for weights, demonstrations and package caches.
No FlashAttention compilation is needed; upstream's SDPA path is supported.

Once the smoke episode is validated, submit the two baselines:

```bash
sbatch scripts/cannon_eval.sbatch fp
sbatch scripts/cannon_eval.sbatch awq
```

The AWQ job calibrates once and saves reusable scale/clipping parameters before
evaluation. Jobs request one full A100 (40GB or 80GB), eight CPUs and 64GB RAM; adjust the
partition/GPU to your allocation. The `a100` feature excludes `a100-mig` nodes.
Slurm owns `CUDA_VISIBLE_DEVICES`. GPU requeue
jobs may be preempted; an interrupted run is incomplete and must be rerun into
a fresh output directory before reporting its score.

On another Linux GPU machine, use `bash scripts/setup.sh` then
`bash scripts/run_baseline.sh fp` / `bash scripts/run_baseline.sh awq`.

## Results and validation

Each invocation creates a fresh `$DQAT_DATA_DIR/results/<run-id>-<method>/` with
`run.json`, `episodes.jsonl` and `summary.json`; `--videos` also writes MP4s.
Metadata includes Git and submodule versions, checkpoint revision, GPU, exact
settings, calibration selections and AWQ artifact hash. Environment/model errors
are `status: error, success: null` and fail the job. Completed horizon timeouts
are legitimate task failures. Partial summaries must not be treated as full scores.

```bash
source scripts/common.sh
"$DQAT_PYTHON" -m dqat.summarize --fp /path/to/fp-run --awq /path/to/awq-run
"$DQAT_PYTHON" -m pytest -q
```

The comparison checks complete episode pairing and matching protocol/checkpoint
provenance, then reports success rates, per-task counts, discordant successes and
a paired bootstrap interval. A single seed is an initial baseline, not a
reproduction of upstream's three-seed paper result. Local CPU tests cover the
quantization grid, scaling invariance, clipping/search error, artifact replay,
multimodal decoder capture, demo preprocessing and rollout failure semantics.

Session status and experiment handoffs belong in `CHANGELOG.md`.
