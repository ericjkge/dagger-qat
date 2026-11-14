"""Closed-loop comparison: BF16-driven rollouts (the Step-2 sweep's driving trajectory) vs quantized-policy
rollouts with identical initial states, env seed and per-call noise schedule. Paired per episode.

Reports success counts, and contact-event timing on the driving trajectory (first finger contact,
first grasp, first placement contact, number of grasp losses = grasped True->False transitions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode_events(rec):
    ev = rec["events"]

    def first(k):
        return next((e["t"] for e in ev if e[k]), None)

    g = [e["grasped"] for e in ev]
    losses = sum(1 for a, b in zip(g, g[1:]) if a and not b)
    regrasps = sum(1 for a, b in zip(g, g[1:]) if (not a) and b)
    return dict(success=rec["success"], steps=rec["steps"], first_contact=first("c_grasp"),
                first_grasp=first("grasped"), first_place=first("c_place"), grasp_losses=losses,
                grasp_onsets=regrasps + (1 if g and g[0] else 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="results root with sweep/ and closed_*/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for task_dir in sorted((args.root / "sweep").iterdir()):
        task = task_dir.name
        ref = {r["ep"]: episode_events(r) for r in map(json.loads, (task_dir / "episodes.jsonl").open())}
        out[task] = {"bf16": ref}
        for arm_dir in sorted(args.root.glob("closed_*")):
            f = arm_dir / task / "episodes.jsonl"
            if f.exists():
                out[task][arm_dir.name] = {r["ep"]: episode_events(r) for r in map(json.loads, f.open())}
    summary = {}
    for task, arms in out.items():
        summary[task] = {}
        for arm, eps in arms.items():
            paired = sorted(set(eps) & set(arms["bf16"]))
            if not paired:
                continue

            def agg(key, eps=eps, paired=paired):
                v = [eps[e][key] for e in paired if eps[e][key] is not None]
                return (float(np.mean(v)) if v else None, len(v))

            summary[task][arm] = dict(
                n=len(paired), successes=sum(eps[e]["success"] for e in paired),
                both_success=sum(eps[e]["success"] and arms["bf16"][e]["success"] for e in paired),
                mean_steps=agg("steps"), first_contact=agg("first_contact"), first_grasp=agg("first_grasp"),
                first_place=agg("first_place"), grasp_losses=int(sum(eps[e]["grasp_losses"] for e in paired)),
                grasp_onsets=int(sum(eps[e]["grasp_onsets"] for e in paired)),
                first_grasp_delay_vs_bf16=agg_delta(eps, arms["bf16"], paired, "first_grasp"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "closed_loop.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def agg_delta(a, b, paired, key):
    d = [a[e][key] - b[e][key] for e in paired if a[e][key] is not None and b[e][key] is not None]
    return dict(mean=float(np.mean(d)) if d else None, median=float(np.median(d)) if d else None, n=len(d))


if __name__ == "__main__":
    main()
