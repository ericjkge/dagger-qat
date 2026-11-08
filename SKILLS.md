# SKILLS.md

Setup and debugging lessons for agents working in this repo. Only verified facts go here.
Put each lesson as: **symptom / fact → what to do**. Add the date you verified it.

## Dev machine (Mac) constraints (verified 2026-10-28)

- **Apple M2, 8 GB RAM, no CUDA.** Do not try to load π0.5 here. Its weights are ~7 GB
  in bf16, and macOS already uses several GB, so the machine has no room for activations and
  would swap heavily. RAM is the hard blocker, not disk. Use the Mac only for unit tests on
  toy layers and for code reading.
- **Disk:** ~42 GB free after cleanup (was ~2.7 GB). Enough for envs, the checkpoint and
  LIBERO data. Still check `df -h ~` before big downloads.
- **`uv` is not installed**, and the default `python3` is 3.14 (Homebrew). The project needs
  3.11 (model env) and 3.8 (sim env). Install uv first, then let uv manage the Python versions.
  Don't use the system/Homebrew Python.
- `nvidia-smi` doesn't exist here, which is expected. Any GPU-dependent step must run on the
  GPU box.

## Colab (verified 2026-10-28)

- The Google Colab VS Code extension (`google.colab` 0.9.7) has **no CLI**. Its commands
  (connect to a runtime, open a terminal, upload files) only run from the VS Code UI, so a
  terminal agent can't execute code on Colab through it. Workflow: the agent writes code and a
  notebook into the repo, and a human runs it in VS Code on a Colab GPU kernel. Results come
  back as files (or saved notebook outputs) that the agent reads.

## π0.5 in PyTorch via openpi (verified 2026-10-28 by reading openpi @ 215abfb)

- **Official LIBERO checkpoint is JAX only** (`gs://openpi-assets/checkpoints/pi05_libero`,
  12.4 GB float32). `examples/convert_jax_model_to_pytorch.py` restores it in float32, so it
  needs >20 GB host RAM. Free Colab (~12.7 GB) can't do it.
- **Pre-converted copy:** `sunshk/pi05_libero_pytorch` @ `7f62302` (third party). Its
  `norm_stats.json` is byte-identical to the official one; 812 bf16 tensors, 3.617B params,
  openpi key names. Load only `model.safetensors` + `assets/`; never run its custom code.
- **openpi's `load_pytorch` builds the model in float32 first (~14.5 GB host RAM).** Workaround
  in `scripts/e1_activation_shift.py:load_policy`: build under `torch.device("cuda")` with
  default dtype bf16, **then reset every tensor's dtype to what openpi's loader gives**, then
  copy tensors one at a time with strict name/shape checks.
  Pitfall (hit 2026-10-28): openpi keeps `action_in_proj`, `action_out_proj` and `time_mlp_*`
  in **fp32**. Only `paligemma_with_expert` goes to bf16, with a few fp32 exceptions.
  Building everything in bf16 crashes in `embed_suffix` with "mat1 and mat2 must have the same
  dtype". The fix reads the target dtypes from a meta-device copy built openpi's way.
- Set `pytorch_compile_mode=None` in the model config for analysis. The default
  `max-autotune` compiles `sample_actions`, which is slow and gets in the way of hooks.
- **Environment:** openpi pins Python 3.11, `numpy<2`, `torch==2.7.1`,
  `transformers==4.53.2`. On Colab, use openpi's own `uv sync` venv and run scripts with
  `.venv/bin/python`. Don't pip-install into the Colab kernel (numpy 2 conflicts).
  **Colab gotcha (hit 2026-10-28):** Colab's Python is 3.13, and a plain `uv sync` there built
  the venv with 3.13 and then failed compiling `labmaze` (openpi → gym-aloha → dm-control →
  labmaze, which has no 3.13 wheel). Fix: `UV_PYTHON=3.11 UV_PYTHON_PREFERENCE=only-managed`,
  `uv python install 3.11`, delete any wrong-version `.venv`, then
  `uv sync --frozen --python 3.11`. Skip README's extra `uv pip install -e .`: `uv sync`
  already installs openpi, and `uv pip` resolves outside the lockfile.
  The transformers patch is required:
  `cp -r src/openpi/models_pytorch/transformers_replace/* <venv transformers dir>/`.
  Check it with `transformers.models.siglip.check`.
- **Inference structure** (matters for hooks and calibration): `sample_actions` runs SigLIP once
  per camera (3 slots), the PaliGemma LM once on the prefix, then the action expert once per
  denoising step (10 steps × 10 action tokens). VLM and expert never run in the same call at
  inference.
- **Padding runs through the network.** LIBERO feeds a zero-filled third camera
  (`right_wrist_0_rgb`, mask False), and prompts are padded to 200 tokens. These activations are
  computed but masked out. **Exclude them from calibration statistics.** See `Probe` in
  `scripts/e1_activation_shift.py` (uses `embed_prefix`'s pad mask).
- `Policy.infer(obs, noise=...)` accepts a fixed noise array of shape (10, 32) for
  pi05_libero. Use it for any quantized-vs-FP comparison. Input keys: `observation/image`,
  `observation/wrist_image` (uint8 HWC), `observation/state` (8,), `prompt`. Output:
  (10, 7) actions.

## LIBERO data (verified 2026-10-28)

- HF `physical-intelligence/libero` @ `a4336d5`: LeRobot v2.0, 1693 episodes, 40 tasks (29–50
  episodes each), 256×256 PNG images inside the parquet files (~23 MB per episode, 22.9 GB total).
  `meta/episodes.jsonl` maps episode → task, so you can download only the episodes you need.
- These images are what the model was trained on, so feed them as-is (no extra flip).

## Local tooling (verified 2026-10-28)

- `brew install uv` works. `uv run pytest` / `uv run ruff check .` pass. Tests use CPU torch.
- After editing `scripts/e1_activation_shift.py`, regenerate the notebook with
  `uv run python scripts/build_e1_notebook.py` (the notebook embeds the script).

## Quantization code (verified 2026-11-08 on a fake model; not yet on real π0.5)

- Calibration statistics must skip padding. π0.5 pads the prompt to 200 tokens and feeds a
  zero image for the masked third camera; including them distorts ranges and Hessians.
  `vlacal.quant.capture.TokenMask` handles this; reuse it, don't re-derive masks.
- openpi keeps `action_in/out_proj` and `time_mlp_*` in fp32 inside a bf16 model. `QLinear`
  does quantization math in fp32 and the matmul in the layer's own dtype, so an
  unquantized `QLinear` is bit-identical to the original.
- Compare against the official code with the vendored copies in `tests/reference/` (licenses
  kept, `# ruff: noqa`). Upstream GPTQ computes group scales from W *before* the
  in-block updates; don't "fix" that.
- The fake model plants large values (1e3) in padding so any leak into statistics is obvious.
  Its attention must mask padded keys, or the planted values leak through attention into
  `o_proj` inputs (and 1e4 overflowed to NaN).
- Comparing arms: same frames, same per-frame noise (`fixed_noise`), same calibration size.

## Experiment hygiene (project rules, not setup)

- Comparing quantized vs FP: use the **same input frames and the same flow-matching noise
  tensor**. Otherwise sampling noise gets mixed into the measured quantization error.
- Keep everything except the calibration data fixed across runs: bit width, granularity, FP
  layer list, calibration set size, eval frames.
