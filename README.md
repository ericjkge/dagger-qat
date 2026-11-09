# dagger-qat

DAgger-style quantization-aware training for VLAs: train the quantized policy on states it visits itself, labeled by the full-precision policy, to reduce error accumulation.

**Current step:** initial FP/AWQ baselines completed; matched GPTQ is being validated.

The [2026-11-09 paired LIBERO-Long run](results/2026-11-09-openvla-oft-libero-long/README.md)
finished at **93.4% (467/500) for both unquantized BF16 and AWQ W4/g128**.
Each method has 15 successes that the other misses. This is a single base-seed
nominal evaluation with simulated quantization; see the report for protocol,
per-task results, uncertainty and reproducibility details.

## Layout
- `src/dqat/policy.py` — load OFT checkpoint, predict actions
- `src/dqat/awq.py` — AWQ (simulated quantization)
- `src/dqat/gptq.py` — GPTQ with Hessian error compensation (simulated quantization)
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
are paired across arms. Before **every episode**, the policy seed is
`7 + 1000 * task_id + initial_state_id`, followed by `env.seed(0)`. In LIBERO,
the latter resets the process-wide NumPy RNG to 0; Python/Torch keep the
per-episode seed. Upstream seeds global RNGs once and the environment when it is
created, so this paired protocol uses a different reseeding schedule. Scores
should be compared within this protocol, not presented as an upstream paper
reproduction.

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

## GPTQ baseline

GPTQ uses the same checkpoint, 32 calibration observations, decoder-only W4/g128
scope and paired evaluation protocol. It adapts the original
[IST-DASLab implementation](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf),
retaining Apache-2.0 attribution in `licenses/gptq.txt`. Settings are asymmetric
per-output-channel groups, natural column order (activation ordering disabled),
128-column compensation blocks and damping of 1% of the Hessian diagonal mean.
Weights and Hessian calculations use float32; evaluation weights are BF16.

Calibration is true-sequential: Q/K/V, attention output, MLP up/gate, then MLP
down. Later groups and blocks see earlier quantization errors, as in the reference
GPTQ Llama implementation. This is an internal calibration path, not on-policy
environment data collection. AWQ's original-output propagation is preserved.
Both methods use their native grid/search rules at the same bit width and scope.

The GPTQ cache saves packed four-bit codes plus float32 group scales and integer
zero points. Loading reconstructs BF16 weights for the existing inference engine;
the artifact's compact storage does not constitute a packed-kernel benchmark.
Cache metadata includes the reference revision, exact calibration manifest and
settings. Raw records, runtime inventories and cluster details remain local;
public result reports contain aggregate measurements.

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

Validate GPTQ on one episode before running its full benchmark with the saved cache:

```bash
gptq_smoke=$(sbatch --parsable --time=02:00:00 scripts/cannon_eval.sbatch \
  gptq --tasks 0 --episodes 1 --videos)
sbatch --dependency="afterok:$gptq_smoke" --time=05:00:00 \
  scripts/cannon_eval.sbatch gptq
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
"$DQAT_PYTHON" -m dqat.summarize --fp /path/to/fp-run --gptq /path/to/gptq-run
"$DQAT_PYTHON" -m pytest -q
```

The comparison checks complete episode pairing and matching protocol/checkpoint
provenance, then reports success rates, per-task counts, discordant successes and
a paired bootstrap interval. A single seed is an initial baseline, not a
reproduction of upstream's three-seed paper result. Local CPU tests cover the
quantization grid, scaling invariance, clipping/search error, artifact replay,
multimodal decoder capture, demo preprocessing and rollout failure semantics.
GPTQ checks additionally compare against synthetic outputs generated by the
original implementation, cover variable-length Hessian accumulation, verify
quantized-block propagation, and replay packed artifacts through real Llama layers.

Session status and experiment handoffs belong in `CHANGELOG.md`.
