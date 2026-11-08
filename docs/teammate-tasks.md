# Teammate tasks (Cannon, A100)

Goal: run the oracle kill-check in `docs/oracle-experiment.md` as fast as possible. The
quantization code, recorder, quantized server and offline evaluator already exist and pass unit
tests on a fake model; nothing has been run on the real π0.5 yet. Tasks are in order; the
**bold** ones block everything after them.

Branch: `itsjiamoon/quant-core`. Unit tests: `uv run pytest` (Mac or Linux, no GPU).

## Frame layout (everything reads/writes here)

Each episode is one `.npz` written by `sim/recording.py` (`EpisodeRecorder`): every policy
query's images, state and step, the prompt, and a JSON `meta` (suite, task, init state, seed,
perturbation, policy metadata, success).

```
$VLACAL_DATA_DIR/frames/
  libero_fp/calib/        FP policy, clean LIBERO, init states 0-24      (calibration)
  libero_fp/heldout/      FP policy, clean LIBERO, init states 25-49     (clean eval)
  plus_camera/split_a_fp/       FP policy, LIBERO-Plus camera, split A   (O1 calibration)
  plus_camera/split_b_fp/       FP policy, camera split B                (FP success on B)
  plus_camera/split_b_qclean/   Q-clean policy, camera split B           (eval + O2 calibration)
```

Use all 4 suites (spatial, object, goal, long). Roughly 64+ episodes per directory is enough for
the offline stage (256 calibration frames at 4 per episode).

## Tasks

1. **Verify the quantized model on the real checkpoint (≈30 min, 1 GPU).**
   Start the server with `--quant-config configs/quant/rtn_w4.json` (needs no calibration
   data) and `--quant-report`. Check:
   - it reports 180 quantized layers (the code asserts this for `PI0Pytorch`)
   - GPU memory fits
   - weight relative errors in the report look sane (about 0.1–0.2 for 4-bit per-channel RTN;
     much larger means a bug)
   - one smoke episode runs end to end
   `sim/smoke_episode.py` currently rejects quantized servers (it requires
   `metadata["quantization"] is None`); relax that check behind a flag in the batch runner.
2. **Batch rollout runner** (`sim/run_batch.py`). It is the smoke client in a loop over a list
   of (suite, task, init state, seed):
   - per-episode JSONL (same record as the smoke client, plus the perturbation fields)
   - pass `on_query=recorder.add` to `run_episode`, then
     `recorder.save(<frames dir>/<suite>_t<task>_i<init>_s<seed>.npz, meta)`
   - resume support: skip episodes already in the JSONL
   - several sim clients per GPU server if throughput allows. **Measure episodes per GPU-hour**:
     it decides how many closed-loop arms we can afford.
3. **FP clean LIBERO recordings** → `libero_fp/calib` and `libero_fp/heldout`. This also
   gives the first FP success numbers beyond the single smoke episode. Compare them to the
   published rates.
4. **Install LIBERO-Plus** (github.com/sylvestf/LIBERO-plus, assets on HF `Sylvest/LIBERO-plus`)
   in a **separate** sim env (e.g. `sim-plus/.venv`). It replaces the `libero` package, so it
   must not touch `sim/.venv`, which is our in-distribution simulator. Check that it can
   reproduce a few camera-perturbation episodes with FP π0.5.
5. **Camera split** → `configs/splits/camera.json`: a deterministic, seeded 50/50 split of
   camera-perturbation episodes into A and B, stratified by suite and by camera-shift level.
   Commit it, so every arm uses identical episodes.
6. **Q-clean recordings on split B.** Serve
   `--quant-config configs/quant/sq_w4a8_static.json --calib "$VLACAL_DATA_DIR/frames/libero_fp/calib/*.npz"`
   (default per-episode 4, max 256, seed 0; these must match the `q_clean` arm). Record →
   `plus_camera/split_b_qclean`. Also record FP on A and B. Success on B for FP vs Q-clean is
   the closed-loop gap check.
7. **Offline kill-check** (≈1–2 GPU-hours, 9 arms):
   ```
   .venv-model/bin/python scripts/offline_gap.py \
     --checkpoint "$VLACAL_DATA_DIR/checkpoints/pi05_libero_pytorch" \
     --plan configs/killcheck/camera.json --out "$VLACAL_DATA_DIR/results/killcheck-camera-1"
   ```
   Run from the repo root. Send back `results.json` and the printed table.
8. **Then decide** with the rules in `docs/oracle-experiment.md`, and only then run closed
   loop: FP, `q_clean`, `o1`, `o2`, `q_dynamic` on split B, 200–300 paired episodes each.
   If there is no W4A8 gap, redo steps 6–7 with `sq_w4a4_static.json`.
9. Afterwards: language (paraphrase calibration) and lighting (negative control), same recipe.

## Things that will bite

- Calibration and eval always use fixed per-frame noise (`vlacal.calib.frames.fixed_noise`),
  so arm differences are not sampling noise. Don't bypass `calibration_runner` / `predict`.
- `offline_gap.py` checks that undoing quantization restores the FP model exactly. If it raises,
  something modified the model in place: stop and report it; don't loosen the check.
- Statistics always come from the FP model (non-sequential calibration). That's intended.
- Keep calibration size fixed across arms (4/episode, ≤256 frames). Changing it for one arm
  makes the comparison apples-to-oranges.
