"""Delay vs miss: re-simulate Step-2 counterfactuals over a 10-step horizon (CPU only, no policy calls).

The Step-2 window is the 5 executed actions, so a contact whose reference time is step 4-5 can leave
the window after a 1-step delay and be scored as a "flip". Here every arm executes its own perturbed
5-action prefix and then the SAME continuation a_ref[5:10] (the reference chunk's unexecuted half), from
the same snapshot. A contact present in the reference within 10 steps but absent in the arm is a MISS;
present in both at different steps is a SHIFT (signed Δt, arm - ref).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="transition,pre_contact")
    args = ap.parse_args()

    from dqat.pi05 import sim
    from dqat.pi05.phases import CHANNELS, first_change, phase

    meta = json.loads((args.sweep / "meta.json").read_text())
    suite_name, suite, task_id = sim.find_task(meta["task"])
    env = sim.make_env(suite.get_task(task_id))
    env.reset()
    guards = sim.Guards(env)
    wanted = set(args.phases.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    out = (args.out / "rescore.jsonl").open("w")
    cache = {}
    n = 0
    for line in (args.sweep / "states.jsonl").open():
        r = json.loads(line)
        ph = phase(r["trace_ref"])
        if ph not in wanted:
            continue
        f = args.sweep / f"snaps_ep{r['ep']:03d}.npz"
        if not f.exists():
            continue
        if r["ep"] not in cache:
            cache = {r["ep"]: dict(np.load(f))}
        z = cache[r["ep"]]
        i = int(np.where(z["call"] == r["call"])[0][0])
        snap = sim.Snapshot(z["mj"][i], z["grip"][i], float(z["time"][i]), int(z["step"][i]))
        a_ref = np.asarray(r["a_ref"])
        tr_ref = sim.branch(env, guards, snap, a_ref[:10])
        rec = dict(ep=r["ep"], call=r["call"], phase=ph, arms={})
        for cfg in meta["configs"]:
            for arm, prefix in (("q", np.asarray(r[cfg]["a"])[:5]), ("rand", np.asarray(r[cfg]["a_rand"]))):
                tr = sim.branch(env, guards, snap, np.concatenate([prefix, a_ref[5:10]]))
                res = {}
                for c in CHANNELS:
                    a, b = first_change(tr_ref, c), first_change(tr, c)
                    res[c] = dict(ref=a, alt=b, miss=a is not None and b is None, extra=a is None and b is not None,
                                  dt=(b - a) if (a is not None and b is not None) else None)
                rec["arms"][f"{cfg}/{arm}"] = dict(dnorm=r[cfg]["dnorm"], **res)
        out.write(json.dumps(rec) + "\n")
        n += 1
    print(f"rescored {n} states", flush=True)


if __name__ == "__main__":
    main()
