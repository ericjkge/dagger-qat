# Contact-guard-preserving activation quantization for π0.5 on LIBERO: results (2026-11-14)

**Recommendation: NO-GO.** Contact-guard-coordinated rounding is not supported by the data. The failure mode we did
find (`down_proj` dead-zone shrinkage) is already handled by published rotation methods: HoloQ-VLA and FoldQuantVLA
(see §8 and [SUMMARY.md](SUMMARY.md)).

The working hypothesis was that "activation quantization errors near contact transitions can change whether contact
occurs, even when numerical action errors are small." The data do not support it in that form.

- **What happens near contact under A4.** When 4-bit activation quantization (A4) perturbs contact, it mostly
  *shifts the timing* by about 1 control step. It rarely *removes* a contact.
- **A matched control behaves the same.** A random action perturbation of the same norm produces contact changes at
  the same rate.
- **The closed-loop damage has another cause.** It comes from a systematic, phase-independent attenuation of the
  action chunk: A4 shrinks translational speed by about 23%. The cause is the dead zone of the per-token 4-bit grid
  on the extremely heavy-tailed `down_proj` inputs.
- **The fix is calibration-free.** A randomized Hadamard rotation before the grid (QuaRot-style) removes the
  attenuation and restores BF16 contact timing in closed loop.

---

## 1. Setup (what was actually run)

| Item | Value |
|---|---|
| Model | openpi PyTorch, official `pi05_libero` checkpoint (JAX→PyTorch converted, openpi commit `215abfb`), BF16 weights and compute, `torch.compile` off so hooks run |
| Quantized | **Activations only**: the inputs of all 7 linears in each of the 18 action-expert (Gemma-300M) layers, as 4 input groups per layer (`qkv`, `o`, `gateup`, `down`), across all 10 flow steps. 72 groups × 10 steps = 720 (group, step) sites. **Weights stay BF16.** The VLM prefix and its KV cache are untouched |
| Quantizer | Dynamic per-token asymmetric min-max integer grid with nearest rounding (fake-quant: quantize → dequantize to BF16). 4-bit (A4) and 8-bit (A8) |
| Configs | `a4_all`; `a8_all`; `a4_down` (only the `down_proj` input); `a4_nodown` (everything except `down_proj`); spin-off configs `a4_rotdown` and `a4_rot` (randomized-Hadamard rotation before the grid on `down` / on all groups) |
| Tasks | LIBERO-Spatial: *pick up the black bowl between the plate and the ramekin and place it on the plate* (`spatial_bowl`)<br>LIBERO-Object: *pick up the alphabet soup and place it in the basket* (`object_soup`)<br>LIBERO-Goal: *put the cream cheese in the bowl* (`goal_cheese`) |
| Episodes | 20 per task: pinned LIBERO initial states 0–19, `env.seed(7)`, 10 settle steps, max steps 220 / 280 / 300, `replan_steps = 5`, 224 px inputs (openpi preprocessing) |
| Pairing | Every comparison uses the same simulator snapshot, the same policy inputs and the same flow-matching noise. The noise seed is a deterministic function of (task, episode, call). BF16 inference is bit-deterministic at fixed noise: replay error 0.0 over all 313 Step-3 states |
| Contact guards (simulator, analysis only) | `d_grasp` (min finger↔object distance), `c_grasp` (finger contact), `grasped` (robosuite `_check_grasp`), `d_place` (object↔target distance), `c_place`. Active guard = `d_place` if grasped, else `d_grasp` |
| Phases (from the BF16 reference) | `transition` = any contact channel changes within the 5 executed steps; `pre_contact` = active guard < 3 cm; `holding`; `free`; `contact` |
| Hardware | 1× A100-SXM4-80GB (Runpod Community Cloud, $1.39/h), about 2.7 h of GPU time |

**Implementation changes** (all new; nothing in openpi or LIBERO was modified):

- `src/dqat/pi05/actquant.py`: hook-based fake quantizer. It also supports additive probes (gradients for all 720
  sites in one backward pass), recording, injection, a same-grid rounding callback, and the Hadamard rotation.
- `src/dqat/pi05/policy.py`: π0.5 wrapper with explicit noise and a differentiable action path.
- `src/dqat/pi05/sim.py`: snapshot/restore including the gripper's integrated state, deterministic branch rollouts,
  contact guards.
- `src/dqat/pi05/phases.py`: phase labels and contact-event comparison.
- Scripts in `scripts/pi05/`; unit tests in `tests/test_pi05_actquant.py` (6 passing).

## 2. Step 2: does A4 change contact events, beyond what its error magnitude explains?

For **1,397 one-step counterfactual states** (BF16-driven rollouts, every policy call), we compared the BF16 chunk
with the quantized chunk. The 5 executed actions were branched from the same snapshot. A **magnitude-matched random
control** (same normalized ‖Δa‖, random direction) was included.

**Contact-event flip rate** (a channel changes within the 5-step window), with episode-cluster bootstrap 95% CIs:

| Phase (n) | A4-all | random, matched | A4 `down` only | A4 without `down` | A8 |
|---|---|---|---|---|---|
| transition (204) | 0.176 [0.12, 0.23] | 0.162 [0.11, 0.21] | 0.172 | 0.083 | 0.034 |
| pre_contact (260) | 0.031 [0.01, 0.05] | 0.046 | 0.015 | — | — |
| free (518), holding (402), contact (13) | 0 | ≈ 0 | 0 | 0 | 0 |

- Median normalized ‖Δa‖ for A4-all is about 0.66–1.07. Relative activation MSE is about 5.3% for A4 and 0.05% for A8.
- **Logistic regression** of flips on phase, log ‖Δa‖, log distance-to-contact and an arm indicator gives these
  coefficients: random-arm indicator **+0.08 (≈ 0)**, log ‖Δa‖ +0.45, log distance −0.02.

**So:** once phase and perturbation magnitude are controlled for, quantization error is **no more contact-damaging
than a random error of the same size**. Contact sensitivity is a property of the *state*: only transition states,
and to a lesser degree pre-contact states, are affected. Nothing specific to the quantization *direction* showed up.

**Delay vs. miss** (10-step re-simulation: each arm's perturbed 5-step prefix, then the shared BF16 continuation;
`step2/rescore_summary.json`):

| Transition states (204) | miss | extra | shifted | delayed | advanced |
|---|---|---|---|---|---|
| A4-all | 0.074 | 0.034 | 0.387 | **0.270** | 0.142 |
| random, matched | 0.059 | 0.039 | 0.343 | 0.162 | 0.211 |
| A4 without `down` | 0.044 | 0.034 | 0.240 | 0.147 | 0.093 |
| A8 | 0.015 | 0.015 | 0.093 | 0.034 | 0.059 |

Contacts are mostly *shifted*, not removed. A4 misses about as often as random noise does. What sets A4 apart is a
**bias toward delay** (27% delayed vs. 14% advanced, compared with random noise's 16% vs. 21%). The next sections
trace this bias to `down_proj`.

**Closed loop** (the quantized policy drives; same initial states, seeds and noise schedule; 20 episodes per task):

| Task | BF16 | A4-all | A4 without `down` | A4 + rot(`down`) | A4 + rot(all) |
|---|---|---|---|---|---|
| bowl: success / mean steps / median grasp delay | 20 / 85.7 / 0 | 20 / 119.3 / **+21** | 20 / 90.0 / +1 | 20 / 87.1 / +1 | 20 / 85.0 / 0 |
| soup | 20 / 147.1 / 0 | **16** / 204.2 / +6.5 | 18 / 162.1 / −1 | 19 / 153.8 / −1 | 20 / 149.4 / 0 |
| cheese | 19 / 110.9 / 0 | 19 / 128.6 / **+16.5** | 20 / 91.0 / 0 | 19 / 102.3 / 0 | 18 / 112.8 / 0 |
| **total success** | **59/60** | **55/60** | 58/60 | 58/60 | 58/60 |

- All 4 A4-all soup failures are 280-step **timeouts with the object still grasped** (6–16 mm from the basket).
  This is a slow-execution failure, not a contact flip.
- The BF16 cheese failure (episode 17) is a timeout that recurs in the rotated arms.
- With n = 20 per task, the success differences alone are not statistically significant (soup 20 vs. 16: Fisher
  p ≈ 0.11). The **paired timing shifts** are the robust signal.

## 3. Step 3: does Δg ≈ M(ĥ − h), M = (∂g/∂a)(∂a/∂h), predict the guard changes?

**Method.**
- **States:** 313, stratified by phase (75 each for transition, pre_contact, free and holding; all 13 contact
  states) across the 3 tasks.
- **∂g/∂a:** central finite differences in the simulator (35 action coordinates, δ = 0.02 normalized). g itself is
  not differentiable.
- **∂a/∂h:** autograd through the BF16 action expert (35 backward passes, all 720 sites at once); the gradient path
  matches inference to ≤ 2.3e-7.
- **Δh:** the nearest-rounding error on the actual 4-bit grid, in the FP-input first-order approximation.
- **Readouts:** g_end (guard after 5 steps) and g_min (minimum over the window, so crossings anywhere in the window
  count).
- **Sign test:** margins |g_ref| < 2 mm are reported separately as ambiguous.

Results (`step3/mechanism_summary.json`, `step3/predicted_vs_observed.png`):

| A4-all, g_end | Simulator-only linearization ∇g·Δa_actual | Full chain M(ĥ − h) | Model side: cos(Δa_pred, Δa), ‖Δa_pred‖/‖Δa‖ |
|---|---|---|---|
| transition (75) | r 0.65, Spearman 0.56, 67% within tolerance | r 0.53, Spearman 0.42, sign 0.67, 71% within | 0.76, 0.59 |
| pre_contact (75) | r 0.61, Spearman 0.59, 61% within | r 0.55, Spearman 0.55, sign 0.69, 36% within | 0.83, 0.65 |
| near (\|g_ref\| < 5 cm, 219) | r 0.66, Spearman 0.65 | r 0.52, Spearman 0.62, sign 0.74 | 0.85, 0.62 |
| single group, flow step 9 (pre_contact) | Spearman 0.79 | Spearman 0.74, sign 0.88 | **0.98, 0.95** |
| single group, flow step 0 | — | sign ≈ 0.6 (noise floor) | 0.10, 0.05 |

"Within tolerance" means |error| ≤ max(0.5 mm, 30% of |observed Δg|).

**Sign preservation (contact crossings).**
- Clear-margin crossings are rare: 1 observed vs. 2 predicted in transition states, 0 vs. 0 in pre-contact states.
  Agreement over the 219 near-contact states is 0.97, but it is dominated by "no crossing" on both sides.
- Most real crossings happen **inside the ambiguous margin**: 10 of 59 ambiguous transition states cross. By
  construction, a linear predictor cannot call these.

**Why the predictor is only partially predictive.**

1. **The guard is non-smooth exactly at transitions.** Even linearizing only the simulator reaches just r ≈ 0.65.
   Contact onset, the gripper's `sign(a)` integration and switches in the minimum-distance geometry pair all produce
   jumps. The largest residuals are 6–12 cm guard jumps in free space, caused by min-distance pair switching; this
   is a measurement artifact, not physics.
2. **The model side is linear per site but not in aggregate.** For one late-step site, J·Δh reproduces Δa
   (cos 0.98, norm ratio 0.95). With all 720 sites quantized, the first-order sum misses the compounding: cos 0.76,
   and it **underestimates ‖Δa‖ by about 40%**.
3. **Early flow-step perturbations sit below the noise floor.** Δa ≈ 0.003, and the linear prediction is
   uninformative.

**Verdict for Step 3:** the chain is predictive in **rank and sign** for single sites and in pre-contact states.
It is **not** reliable in magnitude, and not at the transition states where contact actually changes. Step 4 was
therefore run only as a small oracle diagnostic.

## 4. Step 4 (oracle PoC): guard-coordinated rounding on the same grid

**Setup.**
- **States:** 75 (51 pre-contact, 24 transition).
- **Quantized site:** the single site with the largest observed |Δg_end|. In 33 of 50 states this is `L0.down` at
  flow step 9.
- **Rounding choice:** the K = 12 coordinates with the largest |M_i Δh_i| are re-rounded to floor or ceil. The grid
  (scale, zero point, 4 bits) is fixed.
- **Methods:** nearest; guard-greedy (linearized |g| error); action-greedy (generic output-aware ‖J Δh‖², normalized
  units); brute force over 2¹² (diagnostic).
- **Metric:** the **actual** simulated |Δg_end|. This is an oracle: it uses privileged simulator gradients and BF16
  activations at run time.

| Method | Only that site quantized: mean / median \|Δg\| (mm) | better / worse than nearest | Inside full A4: mean / median (mm) | better / worse |
|---|---|---|---|---|
| nearest | 0.68 / 0.43 | — | 4.24 / 2.21 | — |
| guard (greedy) | **0.55 / 0.28** | **0.88 / 0.08** | 4.31 / 2.22 | 0.47 / 0.52 |
| action-aware (generic) | 0.59 / 0.34 | 0.75 / 0.20 | 4.24 / 2.23 | 0.44 / 0.55 |
| brute force (guard objective) | 0.56 / 0.29 | 0.87 / 0.11 | 4.25 / 2.12 | 0.47 / 0.52 |

- **Isolated, the targeted rounding works.** Guard-coordinated rounding beats nearest and beats generic
  output-aware rounding (88% vs. 75% of states improved; median error −35% vs. −22%). Greedy is essentially optimal
  for its linear objective.
- **The absolute gain is tiny:** about 0.1 mm.
- **Inside full A4 it vanishes** (coin-flip better or worse). The error from the other 719 sites, mostly the
  systematic `down_proj` attenuation, is about 30× larger than what re-rounding 12 coordinates can buy.
- **Not deployable as tested:** it needs simulator gradients at run time.

## 5. Spin-off: the real failure mode is a `down_proj` dead-zone bias, fixed by Hadamard rotation

### Diagnosis

**Per-group statistics** of the 4-bit grid on real expert inputs (12 states; `raw/gain/gain_diagnostic.json`):

| Input group | Kurtosis | Entries rounded to 0 | Input gain ⟨ĥ,h⟩/⟨h,h⟩ | **Output gain** ⟨Wĥ,Wh⟩/⟨Wh,Wh⟩ |
|---|---|---|---|---|
| `down` (gelu(gate)·up, width 4096) | **130–264** | **89–92%** | 0.93–0.94 | **0.83–0.87** |
| `qkv` | 19–23 | 54–55% | 0.99 | 0.95–0.96 |
| `gateup` | 28–38 | 51–52% | 0.99 | 0.98 |
| `o` | 8–9 | 46–48% | 1.00 | 0.99 |

**Mechanism.**
- A few huge per-token outliers set the min-max step.
- Most informative `down_proj` inputs then fall into the zero bin, and the MLP output is systematically shrunk.
- Across 18 residual layers this compounds into a **shorter action chunk in the same direction**.

**One-step check** (90 states, paired; `raw/speed/speed_check.json`):

| Config | Translational speed ratio (median) | cos | Action MSE (normalized) | Relative activation MSE |
|---|---|---|---|---|
| A8 | 0.999 | 1.000 | 4.0e-6 | 0.05% |
| A4-all | **0.766** | 0.989 | 2.5e-2 | 5.3% |
| A4 `down` only | **0.711** | 0.993 | 2.6e-2 | 5.4% |
| A4 without `down` | 1.013 | 0.999 | 3.8e-3 | 4.7% |
| A4 + rot(`down`) | 1.010 | 0.999 | 3.8e-3 | 3.1% |
| **A4 + rot(all)** | **0.998** | **1.000** | **6.6e-5** (380× below A4-all) | 1.8% |

**Causal closed-loop test** (Section 2 table; `spinoff/spinoff_closed_loop_and_speed.png`):
- Removing `down_proj` quantization **or** rotating its input restores BF16 episode length and contact/grasp/place
  timing (median grasp delay 0 to ±1 step, vs. +6.5 to +21 for A4-all).
- Success is 58/60 for each fixed arm, vs. 55/60 for A4-all.

**Why the rotation works.**
- An orthonormal randomized Hadamard transform R spreads a per-token outlier over all 4096 coordinates. This shrinks
  the min-max range, so far fewer entries fall in the dead zone.
- Computation stays exact: W h = (W Rᵀ)(R h), so Rᵀ folds into the weight offline.
- It needs **no calibration data and no simulator information**, so it is deployable.
- This mirrors QuaRot and SpinQuant in LLMs. The new point here is that, for a VLA, the dominant symptom of the
  outliers is **a bias in action magnitude (slower motion → delayed contact → timeouts)**, not contact-specific
  sensitivity.

## 6. Runtime and memory overhead (A100, eager PyTorch, 20 calls; `raw/overhead/overhead.json`)

| Config | ms per policy call (median) | Peak GPU memory |
|---|---|---|
| BF16 | 310 | 7.64 GB |
| A8 / A4 fake-quant | 546 | 7.64 GB |
| A4 + rot(`down`) | 832 | 7.64 GB |
| A4 + rot(all) | 1579 | 7.64 GB |

These numbers measure **simulation** cost: Python hooks plus a Python-loop Walsh-Hadamard transform, 720 sites per
call. They are not the cost of a real int4 kernel. A fused FWHT costs O(n log n) per token, small next to the matmul.
QuaRot reports online Hadamard overhead in the single-digit percent.

Step 3 analysis cost about 46 s per state (35 backward passes + 70 finite-difference branches + 28 quantized arms).
The Step 4 oracle cost about 15–20 s per state.

## 7. Failure cases and limitations

- **Privileged information.**
  - The guards, phases, ∂g/∂a and the Step-4 rounding all use simulator state; they are analysis tools only.
  - The deployable result (Hadamard rotation) uses none of them.
- **Fake quantization.**
  - Activations are rounded to the integer grid and dequantized in BF16, so this is not integer arithmetic.
  - Weights are BF16 throughout. **No claim here is about weight quantization.**
- **Jacobians.** Computed through the **BF16** graph, not FP32 as originally planned. The noise floor (Δa about
  0.003–0.005 for tiny step-0 perturbations) limits small-signal predictions.
- **Guard measurement.**
  - `d_grasp` and `d_place` are minimum mesh distances. Switches in the closest geometry pair cause cm-scale jumps
    (outliers in Step 3; robust metrics and a near-contact subset are reported).
  - Invalid negative mesh distances with no confirming contact are dropped and counted.
- **Gripper discontinuity.** The Panda gripper integrates `sign(a_grip)`, so a sign flip produces a step change in g
  that no linear model captures.
- **Window boundary.** Step 2 flip rates use the 5 executed steps. The 10-step rescore shows that many "flips" are
  ±1-step shifts across the boundary.
- **Scope.** 3 tasks × 20 episodes and a single env seed, so success-rate differences are underpowered. Placement
  contact on soup is mostly unobserved, because episodes end on success.
- **Single-group oracle.** Step 4 re-rounds only K = 12 coordinates of one site; a joint multi-site oracle was not
  attempted.
- **One rotation seed.** Hadamard sign seed 0, not tuned. No hyperparameters were grid-searched on task success.

## 8. Recommendation: NO-GO

- **Contact-guard-coordinated rounding: NO-GO.** Under 4-bit activation quantization, contact changes in π0.5 are
  (i) confined to transition states, (ii) no more frequent than for a random perturbation of equal norm, (iii) mostly
  ±1-step timing shifts, and (iv) only partly predictable by M(ĥ − h) exactly where it matters. Even as an oracle,
  guard-aware rounding buys about 0.1 mm, which disappears inside full A4.
- **The `down_proj` / Hadamard spin-off: also NO-GO as a method contribution**, after a quick novelty check.
  - Rotation-based W4A4 for π0.5 already exists: HoloQ-VLA (arXiv 2605.28803) and FoldQuantVLA (arXiv 2609.24433,
    which also keeps `down_proj` at INT8).
  - `down_proj` massive outliers are well known in LLMs (DuQuant, QuaRot).
  - QuantVLA corrects related "residual energy drift".
  - On our tasks, rotated A4 already matches BF16 (58/60 vs. 59/60), so there is no headroom.
- **Follow-up idea (error feedback / sigma-delta across denoising and control steps): not novel enough.** Prior work
  covers it: Sigma-Delta Quantized Networks (ICLR 2017), MoDiff, Timestep-Aware Correction / PTQD, and a 2026
  noise-shaping quantizer for neural-network controllers (SICE JCMSI).
- **If continuing:** first locate a regime where rotation-based 4-bit still fails in closed loop (A3, W4A4,
  LIBERO-Long), then design a VLA-specific method around that failure.

## 9. Reproduction

All on the pod: `source /workspace/env.sh` (cwd `/workspace/openpi`, always `.venv/bin/python`, never `uv run`).
Results root: `R=/workspace/results/2026-11-14-pi05-contact`. The npz snapshots are kept only on the pod volume.

```bash
bash /workspace/dqat/scripts/pi05/setup_pod.sh                   # env, openpi@215abfb, LIBERO, mujoco 3.2.3, ckpt conversion
.venv/bin/python /workspace/dqat/scripts/pi05/smoke.py ...       # guard / determinism / grad-path checks
bash /workspace/dqat/scripts/pi05/launch_step2.sh                # sweep (1,397 states) + closed_a4_all
CFGS="a4_nodown a4_rotdown a4_rot" bash /workspace/dqat/scripts/pi05/launch_closed.sh
bash /workspace/dqat/scripts/pi05/launch_step3.sh                # mechanism.py (per-phase 25) + rescore_events.py
.venv/bin/python /workspace/dqat/scripts/pi05/rounding_poc.py --checkpoint /workspace/ckpt/pi05_libero_pytorch \
    --sweep $R/sweep/<task> --mechanism $R/mechanism/<task> --out $R/rounding/<task> --max-states 25
.venv/bin/python /workspace/dqat/scripts/pi05/gain_diagnostic.py --checkpoint ... --sweep $R/sweep/spatial_bowl --out $R/gain
.venv/bin/python /workspace/dqat/scripts/pi05/speed_check.py --checkpoint ... --sweeps $R/sweep/* --states 30 --out $R/speed
.venv/bin/python /workspace/dqat/scripts/pi05/overhead.py --checkpoint ... --sweep $R/sweep/goal_cheese --out $R/overhead
```

Local analysis (raw data in `results/2026-11-14-pi05-contact-raw/`, outputs here):

```bash
python scripts/pi05/analyze_sweep.py ...            # step2/summary.json, flip_by_phase.png, flip_vs_magnitude.png
python scripts/pi05/analyze_closed.py --root results/2026-11-14-pi05-contact-raw --out results/2026-11-14-pi05-contact/step2
python scripts/pi05/analyze_rescore.py results/2026-11-14-pi05-contact-raw/rescore/* --out results/2026-11-14-pi05-contact/step2
uv run --no-project --with numpy --with scipy --with matplotlib python scripts/pi05/analyze_mechanism.py \
    results/2026-11-14-pi05-contact-raw/mechanism/* --out results/2026-11-14-pi05-contact/step3
python3 scripts/pi05/analyze_rounding.py results/2026-11-14-pi05-contact-raw/rounding/* --out results/2026-11-14-pi05-contact/step4
uv run --no-project --with numpy --with matplotlib python scripts/pi05/plot_spinoff.py --raw results/2026-11-14-pi05-contact-raw \
    --closed results/2026-11-14-pi05-contact/step2/closed_loop.json --out results/2026-11-14-pi05-contact/spinoff
```

Logs: `results/2026-11-14-pi05-contact-raw/logs/`. Configs: every run's `meta.json` (`config_defs`, prompt, GPU,
seeds).
