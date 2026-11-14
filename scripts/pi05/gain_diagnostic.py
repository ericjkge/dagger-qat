"""Why does A4 shrink π0.5 actions? Per-group statistics of the per-token 4-bit grid on real expert inputs.

For saved BF16-rollout states (exact replay), records every action-expert linear input h (BF16 run,
no quantization) and its nearest-rounded 4-bit value ĥ, and reports per input kind and flow step:
  gain      <ĥ, h> / <h, h>               (1 = no systematic attenuation)
  zeroed    fraction of entries mapped exactly to 0 (|h| < s/2 around the zero point)
  energy    ||ĥ||² / ||h||²
  kurtosis  E[h⁴]/E[h²]² per token (heavy tails -> large grid step s)
  out_gain  <W ĥ, W h> / <W h, W h> for the consuming projection (down_proj / o_proj / q_proj / gate_proj)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    from dqat.pi05.actquant import ActivationController, grid_params, nearest_codes, dequant
    from dqat.pi05.policy import Pi05

    meta = json.loads((args.sweep / "meta.json").read_text())
    pi = Pi05(args.checkpoint)
    ctrl = ActivationController(pi.model).install()
    expert = pi.model.paligemma_with_expert.gemma_expert.model
    consumer = {"qkv": "self_attn.q_proj", "o": "self_attn.o_proj", "gateup": "mlp.gate_proj", "down": "mlp.down_proj"}
    files = sorted(args.sweep.glob("snaps_ep*.npz"))
    rng = np.random.default_rng(0)
    picks = []
    for f in files:
        z = np.load(f)
        for i in range(len(z["call"])):
            picks.append((f, i))
    picks = [picks[k] for k in rng.permutation(len(picks))[: args.states]]
    acc = {}
    for f, i in picks:
        z = np.load(f)
        el = {"observation/image": z["img"][i], "observation/wrist_image": z["wrist"][i],
              "observation/state": z["state"][i], "prompt": meta["prompt"]}
        ctrl.quant, ctrl.record = None, "all"
        ctrl.begin_call()
        pi.infer(el, z["noise"][i])
        for (group, step), (h, _) in ctrl.recorded.items():
            layer, kind = group.split(".")
            s, zp, qmax = grid_params(h, 4)
            codes = nearest_codes(h, s, zp, qmax)
            hq = dequant(codes, s, zp)
            W = expert.layers[int(layer[1:])].get_submodule(consumer[kind]).weight.float()
            y, yq = h @ W.T, hq @ W.T
            x = h.reshape(-1, h.shape[-1])
            kurt = ((x**4).mean(-1) / (x**2).mean(-1).clamp(min=1e-20) ** 2).mean()
            stats = [float((hq * h).sum() / (h * h).sum()), float((codes == zp).float().mean()),
                     float((hq * hq).sum() / (h * h).sum()), float(kurt), float((yq * y).sum() / (y * y).sum())]
            acc.setdefault((kind, "early" if step < 5 else "late"), []).append(stats)
        ctrl.recorded = {}
    out = {f"{k}/{ph}": dict(zip(["gain", "zeroed", "energy", "kurtosis", "out_gain"],
                                 np.round(np.mean(v, axis=0), 4).tolist()), n=len(v))
           for (k, ph), v in sorted(acc.items())}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gain_diagnostic.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
