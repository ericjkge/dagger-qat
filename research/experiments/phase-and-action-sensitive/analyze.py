"""Summarize evaluation episodes: success per arm, paired differences with
standard errors, and the last phase reached in failed episodes.
Keeps the last record per (arm, task, init state), so requeued duplicates collapse."""

import argparse
import json
from pathlib import Path

import numpy as np

PHASES = ("approach", "grasp", "transport", "place")


def load(root):
    arms = {}
    for path in sorted((root / "eval").glob("*.jsonl")):
        rows = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[r["task_id"], r["init_state_id"]] = r
        arms[path.stem] = rows
    return arms


def paired(a, b):
    """Mean and SE of success(a) - success(b) over episodes both arms finished."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k]["success"] - b[k]["success"] for k in keys], dtype=float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    arms = load(args.root)

    print("arm | episodes | success | failures by last phase (approach/grasp/transport/place)")
    for arm, rows in arms.items():
        s = np.mean([r["success"] for r in rows.values()])
        fails = [r["last_phase"] for r in rows.values() if not r["success"]]
        print(f"{arm} | {len(rows)} | {s:.3f} | " + "/".join(str(fails.count(ph)) for ph in PHASES))

    print("\npaired difference | mean | SE | episodes")
    seeds = sorted({a.split("-s")[1] for a in arms if a.startswith("worst-s") and not a.endswith("-head")})
    for s in seeds:
        w = arms[f"worst-s{s}"]
        for other in ("uniform", f"phase-s{s}"):
            if other in arms:
                m, se, n = paired(w, arms[other])
                print(f"worst-s{s} - {other} | {m:+.3f} | {se:.3f} | {n}")
        if f"worst-s{s}-head" in arms:
            m, se, n = paired(arms[f"worst-s{s}-head"], w)
            print(f"worst-s{s}-head - worst-s{s} | {m:+.3f} | {se:.3f} | {n}")
    complete = [s for s in seeds if all(len(arms.get(f"{k}-s{s}", {})) == 500 for k in ("worst", "phase"))]
    if complete and "uniform" in arms:
        for other in ("uniform", "phase"):
            diffs = []
            for s in complete:
                b = arms["uniform"] if other == "uniform" else arms[f"phase-s{s}"]
                w = arms[f"worst-s{s}"]
                diffs += [w[k]["success"] - b[k]["success"] for k in sorted(set(w) & set(b))]
            d = np.array(diffs, dtype=float)
            print(f"pooled over seeds {complete}: worst - {other} | {d.mean():+.3f} | {d.std(ddof=1) / np.sqrt(len(d)):.3f} | {len(d)}")


if __name__ == "__main__":
    main()
