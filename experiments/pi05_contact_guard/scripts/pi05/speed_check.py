"""One-step check on saved BF16-rollout states: does a config shrink the action chunk?

For each config: translational speed ratio ||a_q[:5,:3]|| / ||a_ref[:5,:3]||, cosine of the translational
chunk, and normalized action MSE vs BF16 (identical inputs and flow noise)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sweeps", nargs="+", type=Path, required=True)
    ap.add_argument("--configs", default="a4_all,a4_down,a4_nodown,a4_rotdown,a4_rot,a8_all")
    ap.add_argument("--states", type=int, default=40, help="per task")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from contact_sweep import quant_config

    from dqat.pi05.actquant import ActivationController
    from dqat.pi05.policy import Pi05

    pi = Pi05(args.checkpoint)
    ctrl = ActivationController(pi.model).install()
    cfgs = args.configs.split(",")
    res = {c: dict(ratio=[], cos=[], mse=[], rel_act_mse=[]) for c in cfgs}
    for sweep in args.sweeps:
        meta = json.loads((sweep / "meta.json").read_text())
        picks = [(f, i) for f in sorted(sweep.glob("snaps_ep*.npz")) for i in range(len(np.load(f)["call"]))]
        rng = np.random.default_rng(1)
        picks = [picks[k] for k in rng.permutation(len(picks))[: args.states]]
        for f, i in picks:
            z = np.load(f)
            el = {"observation/image": z["img"][i], "observation/wrist_image": z["wrist"][i],
                  "observation/state": z["state"][i], "prompt": meta["prompt"]}
            ctrl.quant = None
            ctrl.begin_call()
            a_ref = pi.infer(el, z["noise"][i])
            sc = pi.out_scale[: a_ref.shape[0]]
            for c in cfgs:
                ctrl.quant = quant_config(c)
                ctrl.begin_call()
                a_q = pi.infer(el, z["noise"][i])
                _, rel = ctrl.activation_mse()
                ctrl.quant = None
                r, q = a_ref[:5, :3].ravel(), a_q[:5, :3].ravel()
                res[c]["ratio"].append(np.linalg.norm(q) / max(np.linalg.norm(r), 1e-12))
                res[c]["cos"].append(float(r @ q / max(np.linalg.norm(r) * np.linalg.norm(q), 1e-12)))
                res[c]["mse"].append(float(np.mean(((a_q - a_ref) / sc) ** 2)))
                res[c]["rel_act_mse"].append(rel)
    out = {c: dict(n=len(v["ratio"]), speed_ratio_median=round(float(np.median(v["ratio"])), 4),
                   speed_ratio_mean=round(float(np.mean(v["ratio"])), 4),
                   cos_median=round(float(np.median(v["cos"])), 4),
                   action_mse_norm_mean=float(np.mean(v["mse"])), rel_act_mse_mean=float(np.mean(v["rel_act_mse"])))
           for c, v in res.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "speed_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
