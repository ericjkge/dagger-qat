"""Greedy bit allocation from score.py output.

Layer damage s_j(b): worst-phase = max over phases of the phase-mean score;
phase-agnostic = mean over all frames. Every layer starts at 8 bits; repeatedly
demote the (layer, lower bit) with the least damage increase per bit-parameter
saved until the parameter-weighted mean bit-width is <= budget."""

import argparse
import json
from pathlib import Path

import numpy as np

PHASES = ("approach", "grasp", "transport", "place")


def greedy(s, n, bits, budget):
    """s: (layer, bit) damage, n: params per layer, bits ascending with 8 last.
    Returns the bit index per layer."""
    cur = np.full(len(n), bits.index(8))
    while (np.array(bits)[cur] * n).sum() / n.sum() > budget:
        best = None
        for j in range(len(n)):
            for i in range(cur[j]):
                ratio = (s[j, i] - s[j, cur[j]]) / ((bits[cur[j]] - bits[i]) * n[j])
                if best is None or ratio < best[0]:
                    best = (ratio, j, i)
        cur[best[1]] = best[2]
    return cur


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--mode", choices=("worst-phase", "phase-agnostic"), required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = np.load(args.scores)
    bits, n = list(d["bits"]), d["params"].astype(np.float64)
    per_phase = np.stack([d["scores"][d["phase"] == ph].mean(0) for ph in PHASES])  # (phase, layer, bit)
    s = per_phase.max(0) if args.mode == "worst-phase" else d["scores"].mean(0)

    cur = greedy(s, n, bits, args.budget)

    alloc = {str(name): int(bits[i]) for name, i in zip(d["layers"], cur)}
    worst = per_phase[:, np.arange(len(n)), cur].argmax(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        mode=args.mode, budget=args.budget, scores=str(args.scores),
        mean_bits=float((np.array(bits)[cur] * n).sum() / n.sum()),
        layers_per_bit={str(b): int((np.array(bits)[cur] == b).sum()) for b in bits},
        worst_phase_counts={ph: int((worst == k).sum()) for k, ph in enumerate(PHASES)},
        bits=alloc,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
