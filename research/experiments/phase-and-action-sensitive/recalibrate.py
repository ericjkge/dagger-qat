"""Action-head recalibration: quantize the backbone with an allocation, then fine-tune only
the BF16 L1 action head so that, on quantized features, it reproduces the BF16
policy's normalized action chunks on demo frames (L1 loss, as in OFT training).
The head checkpoint with the lowest validation L1 is kept."""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from demo_frames import ROOT, SLUG
from quant import apply_bits
from score import frame
from vla import action_hidden


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True, help="frames.npz from demo_frames.py")
    p.add_argument("--alloc", type=Path, required=True, help="allocation json from allocate.py")
    p.add_argument("--out", type=Path, required=True, help="head state_dict .pt")
    p.add_argument("--val-frac", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--eval-every", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    from dqat.policy import OFTPolicy

    policy = OFTPolicy(args.checkpoint, args.seed)
    frames = dict(np.load(args.frames))  # load once; NpzFile re-reads per access
    n = len(frames["phase"])

    @torch.no_grad()
    def hiddens():
        return torch.cat([
            action_hidden(policy, *frame(frames, f)).cpu()
            for f in range(n)
        ])

    head = policy.action_head
    with torch.no_grad():
        target = torch.cat([head.predict_action(h.cuda()[None]).float().cpu() for h in hiddens()])  # (n, 8, 7)
    apply_bits(policy.model, json.loads(args.alloc.read_text())["bits"])
    h_q = hiddens()

    # Split by demo episode: frames of one demo are near-duplicates.
    rng = np.random.default_rng(args.seed)
    episodes = np.unique(frames["episode"])
    held = rng.choice(episodes, size=max(1, int(len(episodes) * args.val_frac)), replace=False)
    is_val = np.isin(frames["episode"], held)
    val, train = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    trained = copy.deepcopy(head).float().train()

    @torch.no_grad()
    def val_l1(model):
        return float(np.mean([
            (model.predict_action(h_q[i : i + 1].cuda().float()).float().cpu() - target[i : i + 1]).abs().mean()
            for i in val
        ]))

    before = val_l1(trained)
    best, best_state, history = before, copy.deepcopy(trained.state_dict()), [(0, before)]
    opt = torch.optim.AdamW(trained.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        idx = rng.choice(train, size=args.batch, replace=False)
        loss = (trained.predict_action(h_q[idx].cuda().float()) - target[idx].cuda()).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            v = val_l1(trained)
            history.append((step, v))
            print(f"step {step} train_l1 {loss.item():.4f} val_l1 {v:.4f}", flush=True)
            if v < best:
                best, best_state = v, copy.deepcopy(trained.state_dict())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in best_state.items()}, args.out)
    summary = dict(
        slug=SLUG, kind="head_recalibration", job=os.environ.get("SLURM_JOB_ID"), out=str(args.out),
        arguments={k: str(v) for k, v in vars(args).items()}, frames=n, train=len(train), val=len(val),
        val_l1_quantized_head=before, val_l1_recalibrated_head=best, history=history,
    )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(ROOT / "research" / "runs" / f"{SLUG}.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in summary.items() if k != "history"}) + "\n")


if __name__ == "__main__":
    main()
