"""Fake activation quantization inside SGLang, following the released ModelOpt NVFP4 checkpoint's recipe.

ACTQ (comma list) selects what is quantized:
  mlp    : NVFP4 inputs to gate/up and down projections (per-16 dynamic FP8 block scale, static global input_scale)
  lmhead : NVFP4 input to lm_head (same recipe)
  attn   : static per-tensor FP8 inputs to all attention / Gated DeltaNet projections
Weights are whatever checkpoint the server loads (here arm B = dequantized released weights)."""
import json, os, re
from pathlib import Path
import torch
from safetensors import safe_open

G = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
CKPT = os.environ.get("ACTQ_CKPT", "/workspace/models/nvfp4")
STATS = {"calls": {}, "clip": {}}


def fp4_act(x, s2):
    shp, dt = x.shape, x.dtype
    xf = x.float().reshape(-1, shp[-1] // 16, 16)
    amax = xf.abs().amax(-1, keepdim=True)
    sc = (amax / (6 * s2)).clamp(max=448).to(torch.float8_e4m3fn).float() * s2
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    y = xf / sc
    g = G.to(x.device)
    idx = torch.bucketize(y.abs().clamp(max=6), (g[1:] + g[:-1]) / 2)
    return (g[idx] * torch.sign(y) * sc).reshape(shp).to(dt)


def fp8_act(x, s):
    return ((x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


SPLIT = {"gate_up_proj": ["gate_proj", "up_proj"], "qkv_proj": ["q_proj", "k_proj", "v_proj"],
         "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"]}


def install(model):
    modes = set(filter(None, os.environ.get("ACTQ", "").split(",")))
    if not modes:
        return
    idx = json.loads(Path(CKPT, "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    def scale(name):
        f = idx.get(name + ".input_scale")
        if f is None:
            return None
        if f not in handles:
            handles[f] = safe_open(Path(CKPT, f), "pt", device="cpu")
        return float(handles[f].get_tensor(name + ".input_scale").float().max())
    n_hooked, missing = {"mlp": 0, "attn": 0, "lmhead": 0}, []
    names = []
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            names.append(name)
        mm = re.search(r"layers\.(\d+)\.(.*)$", name)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or mm is None or "visual" in name:
            continue
        rest = mm.group(2)
        if rest in ("qkv_proj", "o_proj"):  # SGLang flattens full-attention projections onto the layer
            rest = "self_attn." + rest
        base, leaf = f"model.language_model.layers.{mm.group(1)}.{rest}".rsplit(".", 1)
        kind = "mlp" if rest.startswith("mlp.") else ("attn" if (rest.startswith("self_attn.") or rest.startswith("linear_attn.")) else None)
        if leaf == "in_proj_ba":  # not quantized in the released checkpoint
            continue
        if kind not in modes:
            continue
        cands = [f"{base}.{s}" for s in SPLIT.get(leaf, [leaf])]
        ss = [scale(c) for c in cands]
        if any(s is None for s in ss):
            missing.append(name); continue
        s = max(ss)
        fn = (lambda x, s=s: fp4_act(x, s)) if kind == "mlp" else (lambda x, s=s: fp8_act(x, s))
        mod.register_forward_pre_hook(lambda m, args, fn=fn: (fn(args[0]),) + tuple(args[1:]))
        n_hooked[kind] += 1
    Path("/workspace/logs/actq_names.txt").write_text("\n".join(names))
    if "lmhead" in modes:
        s = scale("lm_head")
        lp = getattr(model, "logits_processor", None) or getattr(getattr(model, "model", None), "logits_processor")
        def pre(m, args, kwargs, s=s):
            args = list(args)
            if len(args) >= 2:
                args[1] = fp4_act(args[1], s)
            else:
                kwargs["hidden_states"] = fp4_act(kwargs["hidden_states"], s)
            return tuple(args), kwargs
        lp.register_forward_pre_hook(pre, with_kwargs=True)
        n_hooked["lmhead"] = 1
    print(f"ACTQ modes={sorted(modes)} hooked={n_hooked} missing={missing[:8]} n_missing={len(missing)}", flush=True)
