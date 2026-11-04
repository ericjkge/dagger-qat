"""E1: do image shifts push π0.5's activations outside ranges calibrated on clean frames?

Runs on one CUDA GPU inside openpi's uv environment (see notebooks/e1_colab.ipynb).
Full-precision model only. No model is quantized here: we measure the input of every
nn.Linear (what activation quantization sees) and simulate INT8 quantization of those inputs
on the side.

Comparison design (everything is paired, only the image condition changes):
- calib: clean frames from one episode per task -> calibration ranges (per-tensor and
  per-channel abs-max, as static PTQ would use).
- eval: frames from a *different* episode of the same 40 tasks.
  "clean" = in-distribution baseline. Each perturbation = the same eval frames, perturbed.
  Same frames and same flow-matching noise in every condition.
- Padding is excluded: the zero-filled, masked third camera slot and padded prompt tokens
  still run through the network but never reach the policy, so they are not counted.

Usage (from the openpi repo root):  .venv/bin/python e1_activation_shift.py --out DIR --ckpt DIR
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_REPO = "physical-intelligence/libero"
DATA_REV = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
HORIZON = 10  # pi05_libero action horizon
ACTION_DIM_MODEL = 32  # model-internal (padded) action dim, used for the noise tensor
CONDITIONS = ["clean", "dark", "bright", "warm", "noise10", "noise25"]

# Per-frame, per-layer statistics (columns of the stats array).
STAT_COLS = [
    "n",  # number of activation elements
    "sumsq",  # sum of x^2
    "n_over_T",  # elements with |x| > calibrated per-tensor abs-max
    "excess_sq_T",  # sum of (|x| - T)_+^2, i.e. energy that a static clip at T removes
    "n_over_C",  # elements with |x| > calibrated per-channel abs-max
    "err_static",  # INT8 error, per-tensor scale from clean calibration (static)
    "err_dyn_tensor",  # INT8 error, per-tensor scale from this forward call (dynamic)
    "err_dyn_token",  # INT8 error, per-token scale (dynamic)
]
S = {c: i for i, c in enumerate(STAT_COLS)}


# ---------------------------------------------------------------- pure helpers (unit-tested)


def group_of(name: str) -> str | None:
    """Which part of π0.5 a Linear belongs to. None = never called at inference (lm heads)."""
    if "lm_head" in name:
        return None
    if ".vision_tower." in name:
        return "vision"
    if ".multi_modal_projector" in name:
        return "projector"
    if ".paligemma.model.language_model." in name:
        return "vlm"
    if ".gemma_expert." in name:
        return "expert"
    return "action_io"  # action_in/out_proj, time_mlp_*: kept at FP in VLA quant papers


def perturb(cond: str, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Image-space shifts on uint8 HWC. Approximate lighting / sensor noise; not LIBERO-Plus."""
    x = img.astype(np.float32)
    if cond == "clean":
        return img
    if cond == "dark":
        x = x * 0.5
    elif cond == "bright":
        x = x * 1.5
    elif cond == "warm":  # warmer light: more red, less blue
        x = x * np.array([1.15, 1.0, 0.8], dtype=np.float32)
    elif cond.startswith("noise"):
        x = x + rng.standard_normal(x.shape).astype(np.float32) * float(cond[len("noise") :])
    else:
        raise ValueError(cond)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def int8_sq_error(x, scale):
    """Sum of squared error of symmetric INT8 fake-quant (127 levels per side). Torch tensors."""
    import torch

    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device).clamp_min(1e-12)
    q = torch.clamp(torch.round(x / scale), -127, 127) * scale
    return ((x - q) ** 2).sum(dtype=torch.float64)


def layer_stats(x, T=None, C=None):
    """Stats of one Linear input x [tokens, channels], float32. Returns (stats, chan_absmax)."""
    import torch

    a = x.abs()
    out = torch.zeros(len(STAT_COLS), dtype=torch.float64, device=x.device)
    out[S["n"]] = x.numel()
    out[S["sumsq"]] = (x * x).sum(dtype=torch.float64)
    if T is not None:
        over = (a - T).clamp_min(0)
        out[S["n_over_T"]] = (over > 0).sum()
        out[S["excess_sq_T"]] = (over * over).sum(dtype=torch.float64)
        out[S["n_over_C"]] = (a > C).sum()
        out[S["err_static"]] = int8_sq_error(x, T / 127)
        out[S["err_dyn_tensor"]] = int8_sq_error(x, a.max() / 127)
        out[S["err_dyn_token"]] = int8_sq_error(x, a.amax(dim=1, keepdim=True) / 127)
    return out, a.amax(dim=0)


def topk_jaccard(a: np.ndarray, b: np.ndarray, frac: float = 0.01) -> float:
    """Overlap of the top-`frac` channels by abs-max (outlier channels) between two sets."""
    k = max(1, int(round(frac * len(a))))
    sa, sb = set(np.argsort(-a)[:k].tolist()), set(np.argsort(-b)[:k].tolist())
    return len(sa & sb) / len(sa | sb)


def chan_log2_shift(a: np.ndarray, ref: np.ndarray) -> float:
    """Median |log2| ratio of per-channel abs-max vs calib (SmoothQuant s_j moves alpha x this)."""
    eps = 1e-6 * max(float(ref.max()), 1e-12)
    return float(np.median(np.abs(np.log2((a + eps) / (ref + eps)))))


def layer_metrics(tot: np.ndarray) -> dict:
    """Ratios from summed stats (over frames) of one layer."""
    n, ss = tot[S["n"]], max(tot[S["sumsq"]], 1e-30)
    return {
        "exceed_T": tot[S["n_over_T"]] / n,
        "exceed_C": tot[S["n_over_C"]] / n,
        "clip_energy_T": tot[S["excess_sq_T"]] / ss,
        "relerr_static": tot[S["err_static"]] / ss,
        "relerr_dyn_tensor": tot[S["err_dyn_tensor"]] / ss,
        "relerr_dyn_token": tot[S["err_dyn_token"]] / ss,
    }


def paired_bootstrap_ci(
    d: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% CI of per-frame paired differences."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- data


@dataclasses.dataclass
class Frame:
    uid: int
    split: str
    task: str
    episode: int
    t: int
    image: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    gt: np.ndarray  # (HORIZON, 7) demo actions a[t : t + HORIZON]


def _decode(v) -> np.ndarray:
    from PIL import Image

    if isinstance(v, dict):
        v = v["bytes"]
    if isinstance(v, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"))
    return np.asarray(v, dtype=np.uint8)


def load_frames(cache: Path, n_tasks: int, cal_fracs, eval_fracs, seed: int) -> list[Frame]:
    """One calib episode and one (different) eval episode per task; frames at fixed fractions."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def get(path):
        return hf_hub_download(
            DATA_REPO, path, repo_type="dataset", revision=DATA_REV, cache_dir=cache
        )

    eps = [json.loads(line) for line in open(get("meta/episodes.jsonl"))]
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["tasks"][0]].append(e["episode_index"])
    rng = random.Random(seed)
    frames, uid = [], 0
    for task in sorted(by_task)[:n_tasks]:
        cal_ep, eval_ep = rng.sample(sorted(by_task[task]), 2)
        for split, ep, fracs in (("calib", cal_ep, cal_fracs), ("eval", eval_ep, eval_fracs)):
            tab = pq.read_table(
                get(f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
            ).to_pydict()
            acts = np.asarray(tab["actions"], dtype=np.float32)
            for f in fracs:
                t = int(f * (len(acts) - HORIZON))
                frames.append(
                    Frame(
                        uid,
                        split,
                        task,
                        ep,
                        t,
                        _decode(tab["image"][t]),
                        _decode(tab["wrist_image"][t]),
                        np.asarray(tab["state"][t], dtype=np.float32),
                        acts[t : t + HORIZON],
                    )
                )
                uid += 1
    return frames


# ---------------------------------------------------------------- model


def load_policy(ckpt_dir: str):
    """openpi pi05_libero policy, built directly on the GPU in bf16.

    openpi's own loader builds the model in float32 first (~14.5 GB host RAM), which does not fit
    on free Colab. We build with bf16 as the default dtype, then set every parameter/buffer to the
    dtype openpi's loader would give it (read off a zero-memory meta-device copy built openpi's
    way: backbone bf16 with a few fp32 params, action/time projections fp32), then copy the
    checkpoint in tensor by tensor.
    """
    import torch
    from openpi.models import model as _model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies import policy_config
    from openpi.training import config as _config
    from safetensors import safe_open

    def _match_openpi_dtypes(model, model_config):
        from collections import Counter

        try:
            with torch.device("meta"):  # openpi's construction (fp32 default), no memory used
                ref = PI0Pytorch(config=model_config)
        except Exception as e:  # fallback: openpi's rule, stated directly
            print(f"[load] meta build failed ({e!r}); using explicit dtype rule")
            for name, child in model.named_children():
                if name != "paligemma_with_expert":
                    child.float()  # action_in/out_proj, time_mlp_*: fp32 in openpi
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
            return
        ref.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")  # as policy_config
        want = {n: t.dtype for n, t in [*ref.named_parameters(), *ref.named_buffers()]}
        changed = 0
        for n, p in model.named_parameters():
            if p.dtype != want[n]:
                p.data, changed = p.data.to(want[n]), changed + 1
        for n, b in model.named_buffers():
            if b.dtype != want[n]:
                mod_name, _, buf = n.rpartition(".")
                model.get_submodule(mod_name)._buffers[buf] = b.to(want[n])
                changed += 1
        got = {n: t.dtype for n, t in [*model.named_parameters(), *model.named_buffers()]}
        if got != want:
            bad = [n for n in want if got.get(n) != want[n]][:5]
            raise RuntimeError(f"dtype mismatch vs openpi loader: {bad}")
        print(
            f"[load] dtypes match openpi's loader ({changed} tensors recast): "
            f"{dict(Counter(str(d) for d in got.values()))}"
        )

    def load_pytorch_lowmem(self, train_config, weight_path):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                model = PI0Pytorch(config=train_config.model)
        finally:
            torch.set_default_dtype(prev)
        _match_openpi_dtypes(model, train_config.model)
        sd = model.state_dict()
        with safe_open(weight_path, framework="pt", device="cuda") as f:
            keys = list(f.keys())
            unexpected = [k for k in keys if k not in sd]
            if unexpected:
                raise RuntimeError(f"checkpoint has keys the model lacks: {unexpected[:10]}")
            for k in keys:
                t = f.get_tensor(k)
                if t.shape != sd[k].shape:
                    raise RuntimeError(
                        f"shape mismatch {k}: ckpt {tuple(t.shape)} vs model {tuple(sd[k].shape)}"
                    )
                sd[k].copy_(t)
        ptrs = {sd[k].data_ptr() for k in keys}
        missing = [
            k for k in sd if k not in set(keys) and sd[k].data_ptr() not in ptrs
        ]  # tied weights are ok
        if missing:
            raise RuntimeError(f"model params not in checkpoint: {missing[:10]}")
        print(f"[load] {len(keys)} tensors loaded strictly; tied aliases: {len(sd) - len(keys)}")
        return model

    _model.BaseModelConfig.load_pytorch = load_pytorch_lowmem
    cfg = _config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    return policy_config.create_trained_policy(cfg, ckpt_dir, pytorch_device="cuda")


class Probe:
    """Forward pre-hooks on every Linear; padding-aware statistics per frame."""

    def __init__(self, model, device="cuda"):
        import torch

        self.device = device
        self.torch = torch
        self.names, self.groups, self.mods = [], [], []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and group_of(name) is not None:
                self.names.append(name)
                self.groups.append(group_of(name))
                self.mods.append(mod)
        stray = [
            n
            for n, g in zip(self.names, self.groups, strict=True)
            if g == "action_io"
            and n not in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
        ]
        if stray:
            raise RuntimeError(f"unexpected Linear layers (module grouping is wrong): {stray[:5]}")
        self.calib_T = None  # [L] per-tensor abs-max from clean calibration
        self.calib_C = None  # list of [D] per-channel abs-max from clean calibration
        self.ctx = {"skip_image": False, "img_idx": 0, "img_masks": None, "prefix_valid": None}
        self.calls = np.zeros(len(self.names), dtype=np.int64)

        pwe = model.paligemma_with_expert
        orig_prefix, orig_image = model.embed_prefix, pwe.embed_image

        def embed_prefix(images, img_masks, lang_tokens, lang_masks):
            self.ctx.update(img_masks=img_masks, img_idx=0)
            out = orig_prefix(images, img_masks, lang_tokens, lang_masks)
            self.ctx["prefix_valid"] = out[
                1
            ]  # pad mask [B, L]: False for masked camera + prompt padding
            return out

        def embed_image(img):
            i = self.ctx["img_idx"]
            self.ctx["skip_image"] = not bool(self.ctx["img_masks"][i].all())
            self.ctx["img_idx"] = i + 1
            return orig_image(img)

        model.embed_prefix, pwe.embed_image = embed_prefix, embed_image
        for i, mod in enumerate(self.mods):
            mod.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        group = self.groups[i]

        def hook(mod, args):
            x = args[0]
            if group in ("vision", "projector"):
                if self.ctx["skip_image"]:
                    return
                x = x.reshape(-1, x.shape[-1])
            elif group == "vlm":
                valid = self.ctx["prefix_valid"]
                if tuple(x.shape[:2]) != tuple(valid.shape):
                    raise RuntimeError(
                        f"{self.names[i]}: input {tuple(x.shape)} vs mask {tuple(valid.shape)}"
                    )
                x = x[valid]
            else:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float()
            T = None if self.calib_T is None else self.calib_T[i]
            C = None if self.calib_C is None else self.calib_C[i]
            st, cm = layer_stats(x, T, C)
            self.frame[i] += st
            self.chan_max[i] = self.torch.maximum(self.chan_max[i], cm)
            self.calls[i] += 1

        return hook

    def begin_condition(self):
        self.chan_max = [self.torch.zeros(m.in_features, device=self.device) for m in self.mods]

    def begin_frame(self):
        self.frame = self.torch.zeros(
            len(self.mods), len(STAT_COLS), dtype=self.torch.float64, device=self.device
        )

    def end_frame(self) -> np.ndarray:
        return self.frame.cpu().numpy()

    def set_calibration(self, chan_max):
        self.calib_C = [c.clone() for c in chan_max]
        self.calib_T = self.torch.stack([c.max() for c in chan_max])


# ---------------------------------------------------------------- experiment


def noise_for(uid: int) -> np.ndarray:
    return (
        np.random.default_rng(10_000 + uid)
        .standard_normal((HORIZON, ACTION_DIM_MODEL))
        .astype(np.float32)
    )


def run_condition(policy, probe, frames, cond):
    probe.begin_condition()
    per_frame, actions, t0 = [], [], time.time()
    for k, fr in enumerate(frames):
        rng = np.random.default_rng(
            20_000 + fr.uid
        )  # same pixel noise pattern for a frame across runs
        obs = {
            "observation/image": perturb(cond, fr.image, rng),
            "observation/wrist_image": perturb(cond, fr.wrist, rng),
            "observation/state": fr.state,
            "prompt": fr.task,
        }
        probe.begin_frame()
        out = policy.infer(obs, noise=noise_for(fr.uid))
        per_frame.append(probe.end_frame())
        actions.append(np.asarray(out["actions"], dtype=np.float32))
        if k == 0 or (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{cond}] {k + 1}/{len(frames)}  {el / (k + 1):.2f}s/frame", flush=True)
    return {
        "per_frame": np.stack(per_frame),
        "actions": np.stack(actions),
        "chan_max": [c.cpu().numpy() for c in probe.chan_max],
    }


def analyze(probe, frames_eval, res, cal_chan_max, cal_gt_mean) -> dict:
    groups = sorted(set(probe.groups))
    gt = np.stack([f.gt for f in frames_eval])
    clean = res["clean"]
    summary = {"conditions": {}, "layers": []}

    # Per-layer metrics (summed over frames).
    for cond, r in res.items():
        tot = r["per_frame"].sum(axis=0)
        for i, name in enumerate(probe.names):
            m = layer_metrics(tot[i])
            m.update(
                layer=name,
                group=probe.groups[i],
                cond=cond,
                in_features=len(cal_chan_max[i]),
                range_ratio=float(r["chan_max"][i].max() / max(cal_chan_max[i].max(), 1e-12)),
                outlier_jaccard=topk_jaccard(r["chan_max"][i], cal_chan_max[i]),
                chan_log2_shift=chan_log2_shift(r["chan_max"][i], cal_chan_max[i]),
            )
            summary["layers"].append(
                {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
            )

    # Per-frame, per-layer relative errors -> group mean per frame -> paired CI vs clean.
    def frame_group_metric(pf, col, g):
        idx = [i for i, gg in enumerate(probe.groups) if gg == g]
        num = pf[:, idx, S[col]]
        den = np.maximum(
            pf[:, idx, S["sumsq"]] if col.startswith(("err", "excess")) else pf[:, idx, S["n"]],
            1e-30,
        )
        return (num / den).mean(axis=1)  # mean over layers -> [frames]

    for cond, r in res.items():
        a = r["actions"]
        c = {
            "action_mae_vs_demo": float(np.abs(a[..., :6] - gt[..., :6]).mean()),
            "gripper_agree_vs_demo": float((np.sign(a[..., 6]) == np.sign(gt[..., 6])).mean()),
            "action_drift_vs_clean": float(np.abs(a[..., :6] - clean["actions"][..., :6]).mean()),
            "groups": {},
        }
        for g in groups:
            rows = [x for x in summary["layers"] if x["cond"] == cond and x["group"] == g]
            gd = {
                k: float(np.median([x[k] for x in rows]))
                for k in (
                    "exceed_T",
                    "exceed_C",
                    "clip_energy_T",
                    "relerr_static",
                    "relerr_dyn_tensor",
                    "relerr_dyn_token",
                    "range_ratio",
                    "outlier_jaccard",
                    "chan_log2_shift",
                )
            }
            gd["static_over_dyn_token"] = float(
                np.median([x["relerr_static"] / max(x["relerr_dyn_token"], 1e-30) for x in rows])
            )
            if cond != "clean":
                for col in ("err_static", "err_dyn_token", "n_over_T"):
                    d = frame_group_metric(r["per_frame"], col, g) - frame_group_metric(
                        clean["per_frame"], col, g
                    )
                    gd[f"paired_diff_{col}"] = paired_bootstrap_ci(d)
            c["groups"][g] = gd
        summary["conditions"][cond] = c
    summary["baseline_mean_action_mae"] = float(
        np.abs(cal_gt_mean[None, None, :6] - gt[..., :6]).mean()
    )
    return summary


def print_summary(s: dict):
    print("\n=== Behaviour (FP model, eval frames) ===")
    print(f"baseline (predict mean demo action) MAE: {s['baseline_mean_action_mae']:.4f}")
    print(f"{'cond':10s} {'MAE vs demo':>12s} {'grip agree':>11s} {'drift vs clean':>15s}")
    for cond, c in s["conditions"].items():
        print(
            f"{cond:10s} {c['action_mae_vs_demo']:12.4f} {c['gripper_agree_vs_demo']:11.3f} "
            f"{c['action_drift_vs_clean']:15.4f}"
        )
    cols = [
        "exceed_T",
        "clip_energy_T",
        "relerr_static",
        "relerr_dyn_token",
        "static_over_dyn_token",
        "range_ratio",
        "outlier_jaccard",
        "chan_log2_shift",
    ]
    order = ["vision", "projector", "vlm", "expert", "action_io"]
    for g in sorted(s["conditions"]["clean"]["groups"], key=order.index):
        print(f"\n=== Activations: {g} (median over layers) ===")
        print(f"{'cond':10s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
        for cond, c in s["conditions"].items():
            print(f"{cond:10s} " + " ".join(f"{c['groups'][g][k]:14.4g}" for k in cols))
        for cond, c in s["conditions"].items():
            if cond == "clean":
                continue
            m, lo, hi = c["groups"][g]["paired_diff_err_static"]
            m2, lo2, hi2 = c["groups"][g]["paired_diff_err_dyn_token"]
            print(
                f"  {cond:9s} paired Δ vs clean  static rel.err {m:+.3g} [{lo:+.3g},{hi:+.3g}]   "
                f"dyn-token rel.err {m2:+.3g} [{lo2:+.3g},{hi2:+.3g}]"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="/content/hf_cache")
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument(
        "--smoke", action="store_true", help="2 tasks, 1 frame each: checks the pipeline end to end"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_tasks, cal_fracs, eval_fracs = args.n_tasks, (0.2, 0.5, 0.8), (0.35, 0.65)
    if args.smoke:
        n_tasks, cal_fracs, eval_fracs = 2, (0.5,), (0.5,)

    print("[1/5] loading frames", flush=True)
    frames = load_frames(Path(args.cache), n_tasks, cal_fracs, eval_fracs, seed=0)
    fcal = [f for f in frames if f.split == "calib"]
    fev = [f for f in frames if f.split == "eval"]
    assert not {f.episode for f in fcal} & {f.episode for f in fev}, "calib/eval episodes overlap"
    print(f"  calib {len(fcal)} frames / eval {len(fev)} frames / {n_tasks} tasks", flush=True)

    print("[2/5] loading model", flush=True)
    import torch

    policy = load_policy(args.ckpt)
    model = policy._model  # noqa: SLF001
    print(f"  GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    print("[3/5] determinism check (no hooks yet)", flush=True)
    f0 = fev[0]
    obs = {
        "observation/image": f0.image,
        "observation/wrist_image": f0.wrist,
        "observation/state": f0.state,
        "prompt": f0.task,
    }
    a1 = policy.infer(obs, noise=noise_for(f0.uid))["actions"]
    a2 = policy.infer(obs, noise=noise_for(f0.uid))["actions"]
    det = float(np.abs(np.asarray(a1) - np.asarray(a2)).max())
    print(f"  max |Δaction| same input+noise: {det:.3g}", flush=True)

    probe = Probe(model)
    print(
        f"  hooked {len(probe.names)} Linear layers: "
        + ", ".join(f"{g}={probe.groups.count(g)}" for g in sorted(set(probe.groups))),
        flush=True,
    )

    print("[4/5] calibration pass (clean)", flush=True)
    cal = run_condition(policy, probe, fcal, "clean")
    never = [n for n, c in zip(probe.names, probe.calls, strict=True) if c == 0]
    if never:
        print(f"  WARNING {len(never)} hooked layers never ran, e.g. {never[:3]}", flush=True)
    probe.set_calibration([torch.from_numpy(c).cuda() for c in cal["chan_max"]])
    cal_gt_mean = np.stack([f.gt for f in fcal]).reshape(-1, 7).mean(axis=0)

    print("[5/5] eval passes", flush=True)
    res = {}
    for cond in CONDITIONS:
        res[cond] = run_condition(policy, probe, fev, cond)
        np.savez_compressed(
            out / f"eval_{cond}.npz", per_frame=res[cond]["per_frame"], actions=res[cond]["actions"]
        )

    s = analyze(probe, fev, res, cal["chan_max"], cal_gt_mean)
    s["meta"] = {
        "determinism_max_abs_diff": det,
        "n_calib_frames": len(fcal),
        "n_eval_frames": len(fev),
        "n_tasks": n_tasks,
        "smoke": args.smoke,
        "gpu": torch.cuda.get_device_name(),
        "layers": len(probe.names),
        "data_rev": DATA_REV,
        "stat_cols": STAT_COLS,
    }
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    cm = {f"calib__{i}": a for i, a in enumerate(cal["chan_max"])}
    cm.update({f"{c}__{i}": a for c, r in res.items() for i, a in enumerate(r["chan_max"])})
    np.savez_compressed(
        out / "chan_max.npz", **cm
    )  # key "<cond>__<layer idx>", see layer_names.json
    (out / "layer_names.json").write_text(
        json.dumps({"names": probe.names, "groups": probe.groups})
    )
    print_summary(s)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
