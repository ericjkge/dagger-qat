"""Per-frame action-damage scores for every quantizable layer and bit-width:
score[f, j, b] = mean over k projections u ~ N(0, I_56) of <grad_{W_j} u.A(f), Q_b(W_j) - W_j>^2,
where A(f) is the BF16 policy's normalized 8x7 action chunk at demo frame f."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quant import BITS, quantize, target_layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True, help="frames.npz from demo_frames.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    from dqat.policy import OFTPolicy
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla

    policy = OFTPolicy(args.checkpoint, args.seed)
    model, cfg = policy.model, policy.cfg
    layers = target_layers(model)
    names = sorted(layers)
    model.requires_grad_(False)
    dots = torch.zeros(len(names), len(BITS), device="cuda")

    def hook(j):
        @torch.no_grad()
        def fn(w):
            g = w.grad.float()
            for i, b in enumerate(BITS):
                dots[j, i] = (g * (quantize(w, b) - w.float())).sum()
            w.grad = None
        return fn

    for j, n in enumerate(names):
        layers[n].weight.requires_grad_(True)
        layers[n].weight.register_post_accumulate_grad_hook(hook(j))

    def action(full, wrist, state, instruction):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        images = prepare_images_for_vla([full, wrist], cfg)
        inputs = policy.processor(prompt, images[0]).to("cuda", dtype=torch.bfloat16)
        wrist_pixels = policy.processor(prompt, images[1]).to("cuda", dtype=torch.bfloat16)["pixel_values"]
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_pixels], dim=1)
        proprio = normalize_proprio(state, model.norm_stats[cfg.unnorm_key]["proprio"])
        _, hidden = model.predict_action(
            **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio,
            proprio_projector=policy.proprio_projector, action_head=policy.action_head, use_film=False,
        )
        return policy.action_head.predict_action(hidden).float().flatten()  # 56, normalized

    frames = np.load(args.frames)
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    scores = np.zeros((len(frames["phase"]), len(names), len(BITS)), dtype=np.float32)
    for f in range(len(scores)):
        a = action(frames["full_image"][f], frames["wrist_image"][f], frames["state"][f], str(frames["instruction"][f]))
        for i in range(args.k):
            u = torch.randn(a.shape, generator=rng, device="cuda")
            (a @ u).backward(retain_graph=i < args.k - 1)
            scores[f] += (dots ** 2).cpu().numpy() / args.k
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
