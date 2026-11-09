# dagger-qat

DAgger-style quantization-aware training for VLAs: train the quantized policy on states it visits itself, labeled by the full-precision policy, to reduce error accumulation.

**Current step:** baselines — OpenVLA-OFT on LIBERO-Long, full precision vs. AWQ.

## Layout
- `src/dqat/policy.py` — load OFT checkpoint, predict actions
- `src/dqat/awq.py` — AWQ (simulated quantization)
- `src/dqat/eval_libero.py` — LIBERO-Long eval
- `scripts/` — RunPod launchers
- `third_party/` — openvla-oft, LIBERO (submodules)
