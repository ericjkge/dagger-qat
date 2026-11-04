# Progress

Short, high-level log of what has been done and why. Newest first.

## 2026-10-28 (later): E1 notebook ready, not yet run

- **Built:** `notebooks/e1_colab.ipynb` (3 cells, free Colab T4) running
  `scripts/e1_activation_shift.py`. Unit tests in `tests/test_e1_stats.py` (9 pass, incl. a
  fake-model test that padding never leaks into statistics).
- **What E1 measures** (FP π0.5, no quantized model yet): for every Linear input, compare
  clean held-out frames vs the *same* frames with lighting/noise perturbations, against ranges
  calibrated on clean frames from other episodes:
  - range overflow and clipped energy
  - INT8 error with static (calibrated) vs dynamic (calibration-free) scales
  - outlier-channel overlap and per-channel range shift (drives SmoothQuant factors)
  - split by vision tower / projector / VLM / action expert, with paired bootstrap CIs
  Also checks that the FP model's actions match the demos (vs a mean-action baseline) and how
  much they move under each perturbation.
- **Checkpoint:** official openpi weights are JAX; converting needs >20 GB RAM (free Colab has
  ~12.7 GB). Using a pre-converted third-party PyTorch copy (`sunshk/pi05_libero_pytorch`).
  Verified: identical normalization stats to the official ones, and the expected tensor names
  and sizes. The notebook adds a functional check (action error vs demos).
- **Limits:** image-space perturbations only approximate LIBERO-Plus; eval frames are from the
  training data (LIBERO demos), which is the in-distribution reference by design.

## 2026-10-28

**Where we are:** repo skeleton only. No code, deps, submodules, or checkpoints yet.

**Compute reality check**
- No dedicated GPU. Model runs go through free Colab (T4, 15 GB GPU, ~12.7 GB RAM).
- Dev machine: Apple M2, 8 GB RAM, ~42 GB free disk (after cleanup). π0.5 (~3B params,
  ~7 GB in bf16) cannot run in 8 GB RAM, so the Mac is only for code and unit tests.
- Decision: do not substitute a different/small model or synthetic data for π0.5. Results
  would not transfer, and the comparison would be apples-to-oranges.

**Research design (agreed direction)**
- Key framing: under shift, a quantized policy can fail for two separate reasons:
  1. the FP model itself is brittle (not our hypothesis), and
  2. the quantized model drifts further from FP (our hypothesis).
  We isolate (2) by comparing quantized vs FP outputs **on the same input with the same
  flow-matching noise**.
- Planned offline experiments for the first GPU session (no simulator needed):
  - **E1 Diagnostic:** per-layer activation shift, clean vs perturbed frames (clipping rate
    vs clean ranges, outlier-channel overlap, SmoothQuant factor drift), split by vision
    tower / Gemma / action expert.
  - **E2 Calibration matrix:** calibrate on {clean, OOD, mix} × test on {clean, seen
    perturbation, held-out perturbation}, metric = action error vs FP.
  - **E3 Control:** static (calibrated) vs dynamic (calibration-free) activation scales at the
    same bit width. The gap is the part of the OOD error that calibration causes.
- Deferred until rollouts are possible: LIBERO-Plus task success, on-policy calibration, QVLA.
