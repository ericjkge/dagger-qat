"""Correctness checks for the phase-and-action-sensitive pipeline (GPU job).
Each check prints PASS or raises. Diagnostics print numbers without asserting.
Also writes a small frame subset to --out for the end-to-end run in tests.sbatch."""

import argparse
from pathlib import Path

import numpy as np
import torch

from allocate import greedy
from demo_frames import tag_phases
from quant import BITS, apply_bits, quantize, target_layers
from score import frame, frame_dots, register_dot_hooks
from vla import action_hidden


def check(name, ok, detail=""):
    if not ok:
        raise AssertionError(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}", flush=True)


def unit_tests():
    # Quantizer: at most 2^b - 1 symmetric levels per group, max |w| kept, error falls with bits.
    w = torch.randn(64, 4304, dtype=torch.bfloat16)  # 4304 is not a multiple of 128 (SigLIP fc2)
    errors = []
    for b in BITS:
        q = quantize(w, b)
        check(f"quantize b={b} shape/dtype", q.shape == w.shape and q.dtype == w.dtype)
        groups = torch.nn.functional.pad(q.float(), (0, (-4304) % 128)).view(64, -1, 128)[:, :-1]  # full groups
        levels = max(len(torch.unique(g)) for g in groups.reshape(-1, 128)[:50])
        check(f"quantize b={b} levels <= {2 ** b - 1}", levels <= 2 ** b - 1, f"(max {levels})")
        maxdiff = (q.float().abs().amax(1) - w.float().abs().amax(1)).abs().max().item()
        check(f"quantize b={b} keeps row max |w|", maxdiff < 2e-2 * w.float().abs().max().item(), f"({maxdiff:.4f})")
        errors.append((q.float() - w.float()).pow(2).mean().item())
    check("quantize error decreases with bits", all(a > b for a, b in zip(errors, errors[1:])), str(np.round(errors, 5)))

    # Phase tagger: open 20, close 30 (+1 blip of 2 open steps), open 20.
    g = np.array([-1] * 20 + [1] * 15 + [-1] * 2 + [1] * 13 + [-1] * 20, dtype=np.float32)
    labels, flips = tag_phases(g, window=3, min_hold=5)
    check("tag_phases flips ignore 2-step blip", flips == [(20, "close"), (50, "open")], str(flips))
    expect = ["approach"] * 17 + ["grasp"] * 7 + ["transport"] * 23 + ["place"] * 7 + ["approach"] * 16
    check("tag_phases labels", list(labels) == expect)

    # Greedy allocation: budget met; a layer with zero damage is demoted to 2 bits first.
    rng = np.random.default_rng(0)
    s = np.sort(rng.random((6, 4)), axis=1)[:, ::-1].copy()  # damage decreasing with bits
    s[2] = 0.0
    n = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    cur = greedy(s, n, list(BITS), 3.0)
    mean = (np.array(BITS)[cur] * n).sum() / n.sum()
    check("greedy meets budget", mean <= 3.0, f"(mean {mean:.3f})")
    check("greedy demotes zero-damage layer to 2", BITS[cur[2]] == 2)


def model_tests(policy, frames):
    from experiments.robot.openvla_utils import resize_image_for_policy

    model = policy.model
    vb = model.vision_backbone
    layers = target_layers(model)
    n_dino, n_siglip = len(vb.featurizer.blocks), len(vb.fused_featurizer.blocks)
    expect = 32 * 7 + 4 * (n_dino - 1) + 4 * (n_siglip - 1)
    check("target layer count", len(layers) == expect, f"({len(layers)}; dino {n_dino} siglip {n_siglip} blocks)")
    unused = (f"vision_backbone.featurizer.blocks.{n_dino - 1}.", f"vision_backbone.fused_featurizer.blocks.{n_siglip - 1}.")
    check("no last ViT block targeted", not any(n.startswith(unused) for n in layers))
    params = sum(m.weight.numel() for m in layers.values())
    print(f"INFO quantizable params {params / 1e9:.3f}B", flush=True)

    # Our differentiable forward equals the evaluation path (policy.predict_prepared) on demo frames.
    with torch.no_grad():
        for f in range(4):
            full, wrist, state, instruction = frame(frames, f)
            ours = policy.action_head.predict_action(action_hidden(policy, full, wrist, state, instruction))
            ours = model._unnormalize_actions(ours.float().cpu().numpy().reshape(8, 7), policy.cfg.unnorm_key)
            prepared = dict(full_image=resize_image_for_policy(full, 224), wrist_image=resize_image_for_policy(wrist, 224), state=state)
            theirs = np.asarray(policy.predict_prepared(prepared, instruction))
            check(f"forward matches eval path, frame {f}", np.abs(ours - theirs).max() < 1e-3, f"(max diff {np.abs(ours - theirs).max():.2e})")

    # Image orientation and state layout: the BF16 policy should reproduce demo actions on correctly
    # oriented demo frames, and much worse on 180-degree rotated ones.
    def demo_error(rotate):
        pos, grip = [], []
        with torch.no_grad():
            for f in range(len(frames["phase"])):
                full, wrist, state, instruction = frame(frames, f)
                if rotate:
                    full, wrist = full[::-1, ::-1].copy(), wrist[::-1, ::-1].copy()
                pred = policy.action_head.predict_action(action_hidden(policy, full, wrist, state, instruction))
                pred = model._unnormalize_actions(pred.float().cpu().numpy().reshape(8, 7), policy.cfg.unnorm_key)
                demo = frames["demo_chunk"][f]
                pos.append(np.abs(pred[:, :6] - demo[:, :6]).mean())
                grip.append(np.mean((pred[:, 6] < 0.5) == (demo[:, 6] > 0)))  # model: 1 = open; demo: +1 = close
        return float(np.mean(pos)), float(np.mean(grip))

    upright, rotated = demo_error(False), demo_error(True)
    print(f"INFO demo-action error upright (L1 pos/rot, gripper agreement) {upright}; rotated {rotated}", flush=True)
    check("frames are in model orientation", upright[0] < 0.7 * rotated[0] and upright[1] > 0.9)

    # Hooks: the dot products equal <autograd grad, Q_b(W) - W> for sample layers of each module.
    names = sorted(layers)
    model.requires_grad_(False)
    dots = register_dot_hooks(layers, names)
    sample = [next(n for n in names if n.startswith(p)) for p in
              ("vision_backbone.featurizer.", "vision_backbone.fused_featurizer.", "language_model.")]
    u = torch.randn(56, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    hooked = frame_dots(policy, frame(frames, 0), [u], dots)[0]
    weights = [layers[n].weight for n in sample]
    a = policy.action_head.predict_action(action_hidden(policy, *frame(frames, 0))).float().flatten()
    grads = torch.autograd.grad(a @ u, weights)  # does not accumulate .grad, so hooks do not fire
    for n, w, g in zip(sample, weights, grads):
        for i, b in enumerate(BITS):
            ref = (g.float() * (quantize(w, b).float() - w.float())).sum().item()
            got = hooked[names.index(n), i]
            check(f"hook dot {n.split('.')[0]}..{n.split('.')[-2]} b={b}", abs(got - ref) <= 1e-2 * abs(ref) + 1e-6, f"({got:.4e} vs {ref:.4e})")

    # Diagnostic: first-order score vs the true squared action change from quantizing one layer to 2 bits.
    us = [torch.randn(56, device="cuda", generator=torch.Generator(device="cuda").manual_seed(i)) for i in range(32)]
    est = (frame_dots(policy, frame(frames, 0), us, dots) ** 2).mean(0)[:, BITS.index(2)]
    with torch.no_grad():
        base = policy.action_head.predict_action(action_hidden(policy, *frame(frames, 0))).float().flatten()
        rows = []
        for n in names[:: max(1, len(names) // 12)]:
            w = layers[n].weight
            original = w.detach().clone()
            w.copy_(quantize(w, 2))
            moved = policy.action_head.predict_action(action_hidden(policy, *frame(frames, 0))).float().flatten()
            w.copy_(original)
            rows.append((est[names.index(n)], ((moved - base) ** 2).sum().item()))
    rows = np.array(rows)
    rank = np.corrcoef(np.argsort(np.argsort(rows[:, 0])), np.argsort(np.argsort(rows[:, 1])))[0, 1]
    print(f"INFO first-order vs true ||dA||^2 at 2 bits, {len(rows)} layers: spearman {rank:.2f}", flush=True)
    for e, t in rows:
        print(f"INFO   est {e:.3e}  true {t:.3e}", flush=True)

    # apply_bits changes exactly the quantizable weights.
    for w in model.parameters():
        w.requires_grad_(False)
    target_ids = {id(m.weight) for m in layers.values()}
    before = {n: p.detach().float().sum().item() for n, p in model.named_parameters()}
    apply_bits(model, {n: 3 for n in layers})
    after = {n: p.detach().float().sum().item() for n, p in model.named_parameters()}
    changed = {n for n in before if before[n] != after[n]}
    expected = {n for n, p in model.named_parameters() if id(p) in target_ids}
    check("apply_bits changes only quantizable weights", changed <= expected, f"(unexpected {sorted(changed - expected)[:3]})")
    check("apply_bits changes the quantizable weights", len(changed) >= 0.99 * len(expected), f"({len(changed)}/{len(expected)})")
    levels = len(torch.unique(layers[names[0]].weight[0, :128].float()))
    check("applied 3-bit layer has <= 7 levels per group", levels <= 7, f"({levels})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    unit_tests()
    frames = dict(np.load(args.frames))
    # Two frames per phase for the end-to-end score/allocation run.
    keep = np.concatenate([np.flatnonzero(frames["phase"] == ph)[:2] for ph in ("approach", "grasp", "transport", "place")])
    np.savez(args.out / "frames.npz", **{k: v[keep] for k, v in frames.items()})

    from dqat.policy import OFTPolicy

    policy = OFTPolicy(args.checkpoint, 7)
    model_tests(policy, {k: v[::8] for k, v in frames.items()})  # 32 frames, 8 per phase
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
