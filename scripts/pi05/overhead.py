"""Runtime / peak-memory overhead of the fake-quant hooks per policy call (A100, BF16, no torch.compile).

Note: fake quantization SIMULATES integer activations in floating point; these timings measure the simulation
cost (and the online-Hadamard cost in the same PyTorch eager setting), not a deployed int4 kernel."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from contact_sweep import quant_config

    from dqat.pi05.actquant import ActivationController
    from dqat.pi05.policy import Pi05

    meta = json.loads((args.sweep / "meta.json").read_text())
    z = np.load(sorted(args.sweep.glob("snaps_ep*.npz"))[0])
    el = {"observation/image": z["img"][0], "observation/wrist_image": z["wrist"][0],
          "observation/state": z["state"][0], "prompt": meta["prompt"]}
    pi = Pi05(args.checkpoint)
    ctrl = ActivationController(pi.model).install()
    res = {}
    for name in ("bf16", "a8_all", "a4_all", "a4_rotdown", "a4_rot"):
        ctrl.quant = None if name == "bf16" else quant_config(name)
        for _ in range(3):
            ctrl.begin_call(); pi.infer(el, z["noise"][0])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(args.reps):
            ctrl.begin_call()
            t0 = time.perf_counter(); pi.infer(el, z["noise"][0]); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        res[name] = dict(ms_median=round(1e3 * float(np.median(ts)), 1),
                         peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overhead.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
