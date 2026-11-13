"""Per-frame action-damage scores for every quantizable layer and bit-width:
score[f, j, b] = mean over k projections u ~ N(0, I_56) of <grad_{W_j} u.A(f), Q_b(W_j) - W_j>^2,
where A(f) is the BF16 policy's normalized 8x7 action chunk at demo frame f."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quant import BITS, quantize, target_layers
from vla import action_hidden


def register_dot_hooks(layers, names):
    """Make every listed weight require grad. After each backward, row j of the
    returned tensor holds <grad W_j, Q_b(W_j) - W_j> for each b in BITS, and the
    weight's grad is freed."""
    dots = torch.zeros(len(names), len(BITS), device="cuda")

    def hook(j):
        @torch.no_grad()
        def fn(w):
            g = w.grad.float()
            for i, b in enumerate(BITS):
                dots[j, i] = (g * (quantize(w, b).float() - w.float())).sum()
            w.grad = None
        return fn

    for j, n in enumerate(names):
        layers[n].weight.requires_grad_(True)
        layers[n].weight.register_post_accumulate_grad_hook(hook(j))
    return dots


def frame_dots(policy, frame, us, dots):
    """(len(us), layers, bits) array of <grad_W u.A, Q_b(W) - W> at one frame."""
    a = policy.action_head.predict_action(action_hidden(policy, *frame)).float().flatten()  # 56, normalized
    out = []
    for i, u in enumerate(us):
        dots.fill_(float("nan"))  # every layer's hook must overwrite its row
        (a @ u).backward(retain_graph=i < len(us) - 1)
        if dots.isnan().any():
            raise RuntimeError("A quantizable layer received no gradient")
        out.append(dots.cpu().numpy().copy())
    return np.stack(out)


def frame(frames, f):
    return frames["full_image"][f], frames["wrist_image"][f], frames["state"][f], str(frames["instruction"][f])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True, help="frames.npz from demo_frames.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    from dqat.policy import OFTPolicy

    policy = OFTPolicy(args.checkpoint, args.seed)
    policy.model.requires_grad_(False)
    layers = target_layers(policy.model)
    names = sorted(layers)
    dots = register_dot_hooks(layers, names)

    frames = dict(np.load(args.frames))  # load once; NpzFile re-reads per access
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    scores = np.zeros((len(frames["phase"]), len(names), len(BITS)), dtype=np.float32)
    for f in range(len(scores)):
        us = [torch.randn(56, generator=rng, device="cuda") for _ in range(args.k)]
        scores[f] = (frame_dots(policy, frame(frames, f), us, dots) ** 2).mean(0)
        print(f"frame {f + 1}/{len(scores)} {frames['phase'][f]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, scores=scores, layers=np.array(names), bits=np.array(BITS),
        params=np.array([layers[n].weight.numel() for n in names]),
        phase=frames["phase"], task=frames["task"],
    )
    args.out.with_suffix(".json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()} | {"frames_config": str(args.frames.parent / "config.json")}, indent=2))


if __name__ == "__main__":
    main()
