# Changelog

## Unreleased
- Repo skeleton.

## 2026-11-08 — Sophie / Codex: OpenVLA-OFT baseline setup

- Read current `main` (`4d1c289`) and the previous `PROPOSAL.md` from `a035129`.
  The overhaul removed that proposal and the previous π0.5 runtime; this session
  follows the new immediate scope: OpenVLA-OFT on LIBERO-Long, unquantized vs. AWQ.
  GPTQ, on-policy refresh and a new loop-aware quantizer remain later work.
- Implemented the OFT loader, paired LIBERO evaluation, per-episode JSONL and
  comparison tooling. Preserve the official preprocessing and action semantics;
  propagate infrastructure failures instead of counting them as policy failures.
- Implemented simulated Llama W4/g128 AWQ with activation scale search and clipping,
  adapted from MIT HAN Lab `llm-awq` at `d6e797a`. Sensitive non-decoder components
  remain unquantized. Calibration is 32 task-balanced original training-demo frames.
- Added a pinned Linux/CUDA runtime, asset download/provenance, Cannon setup and
  evaluation jobs. Checkpoint revision: `95220f9a3421a7ff12d4218e73d09ade830fa9a3`;
  dataset revision: `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- Local validation: CPU numerical/integration tests use the pinned Transformers
  fork with the installed macOS Torch 2.11; GPU runtime uses Torch 2.2 as upstream.
  Actual benchmark scores must come from Cannon, not these tests.
- Cannon: authenticated through the user's terminal; new checkout
  `/n/home13/sophiezeng/dagger-qat`, scratch data at
  `/n/netscratch/amin_lab/Everyone/sophiezeng/dagger-qat`.
  Initial CPU setup job `59049206` caught a Protobuf/TF Metadata version conflict;
  pinned Protobuf 3.20.3 for the TF 2.15/TFDS 4.9.3 stack. Setup retry: `59051313`.
  Dependent one-episode FP smoke job: `59051490`.
  Serialize upstream's checkpoint-file updates with a lock so concurrent model
  loads cannot race on the shared checkpoint directory.
  Local validation: 17 tests passed, Ruff and shell syntax/diff checks passed.
  GPU evaluation is pending setup validation.
  No OpenVLA-OFT success rate is claimed yet.
