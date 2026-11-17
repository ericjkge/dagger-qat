"""Figures for the down_proj dead-zone finding and the Hadamard spin-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--closed", type=Path, required=True, help="closed_loop.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["bf16", "closed_a4_all", "closed_a4_nodown", "closed_a4_rotdown", "closed_a4_rot"]
    labels = ["BF16", "A4 all", "A4 w/o down", "A4 + rot(down)", "A4 + rot(all)"]
    colors = ["#444", "#b03a2e", "#d4a017", "#1f6f8b", "#2e8b57"]
    tasks = ["spatial_bowl", "object_soup", "goal_cheese"]
    cl = json.loads(args.closed.read_text())
    speed = json.loads((args.raw / "speed" / "speed_check.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.16
    x = np.arange(len(tasks))
    for i, (a, lab, c) in enumerate(zip(arms, labels, colors)):
        axes[0].bar(x + (i - 2) * w, [cl[t][a]["successes"] for t in tasks], w, color=c, label=lab)
        axes[1].bar(x + (i - 2) * w, [cl[t][a]["first_grasp_delay_vs_bf16"]["median"] if a != "bf16" else 0
                                      for t in tasks], w, color=c, label=lab)
    for ax, title in ((axes[0], "closed-loop success (/20, paired init/noise)"),
                      (axes[1], "median first-grasp delay vs BF16 [control steps]")):
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylim(10, 20.5)
    axes[0].legend(fontsize=7, loc="lower left")
    cfgs = ["a8_all", "a4_all", "a4_down", "a4_nodown", "a4_rotdown", "a4_rot"]
    mse = [speed[c]["action_mse_norm_mean"] for c in cfgs]
    ratio = [speed[c]["speed_ratio_median"] for c in cfgs]
    ax = axes[2]
    ax.bar(range(len(cfgs)), ratio, color=["#888", "#b03a2e", "#b03a2e", "#d4a017", "#1f6f8b", "#2e8b57"])
    ax.axhline(1, color="k", lw=0.7, ls="--")
    for i, m in enumerate(mse):
        ax.text(i, ratio[i] + 0.01, f"MSE\n{m:.1e}", ha="center", fontsize=7)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=20, fontsize=8)
    ax.set_ylim(0.5, 1.15)
    ax.set_title("one-step: translational speed ratio |a_q|/|a_ref| (median, n=90)", fontsize=10)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "spinoff_closed_loop_and_speed.png", dpi=150)
    print("saved")


if __name__ == "__main__":
    main()
