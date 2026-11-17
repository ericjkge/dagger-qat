"""Frame manifest: which recorded state is rendered under which condition, and for what role.

eval   : 40 held-out states (4 per task) x {clean, camera, light, background, noise, language}.
         Every perturbed frame has a clean twin with the identical simulator state.
calib  : 32 other states (task quotas as the baseline), each rendered twice:
         clean (standard calibration) and with one perturbation (types balanced), using variants
         from the calibration half of each task's LIBERO-Plus pool (disjoint from eval variants).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONDITIONS = ("clean", "camera", "light", "background", "noise", "language")
PERTURBATIONS = CONDITIONS[1:]


def load_states(demo_dir: Path) -> list[dict]:
    states = []
    for path in sorted(demo_dir.glob("task*.npz"), key=lambda p: int(p.stem[4:])):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        for i, rec in enumerate(meta["records"]):
            if rec["role"] not in ("eval", "calib"):
                continue  # earlier extraction also stored the baseline's own manifest
            states.append(
                dict(
                    rec,
                    key=f"t{rec['task_id']}_{rec['demo']}_{rec['frame']}",
                    xml=meta["xmls"][rec["xml_sha"]],
                    sim_state=z[f"sim_state_{i}"],
                    proprio=np.concatenate([z[f"ee_states_{i}"], z[f"gripper_states_{i}"]]),
                    demo_actions=z[f"demo_actions_{i}"],
                    hdf5_agentview=z[f"hdf5_agentview_{i}"],
                )
            )
    return states


def build_manifest(states: list[dict], pools: dict, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    frames = []

    def pick(task, category, half, used):
        pool = pools[task][category][half]
        fresh = [j for j in range(len(pool)) if (category, half, j) not in used[task]]
        j = int(rng.choice(fresh if fresh else range(len(pool))))
        used[task].add((category, half, j))
        return pool[j]

    used: dict = {}
    eval_states = [s for s in states if s["role"] == "eval"]
    calib_states = [s for s in states if s["role"] == "calib"]
    for s in eval_states:
        used.setdefault(s["task"], set())
        for condition in CONDITIONS:
            variant = None if condition == "clean" else pick(s["task"], condition, "eval", used)
            frames.append(dict(role="eval", state=s["key"], task_id=s["task_id"], condition=condition, variant=variant))
    order = rng.permutation(len(calib_states))
    for rank, idx in enumerate(order):
        s = calib_states[idx]
        used.setdefault(s["task"], set())
        frames.append(dict(role="calib", state=s["key"], task_id=s["task_id"], condition="clean", variant=None))
        category = PERTURBATIONS[rank % len(PERTURBATIONS)]
        frames.append(
            dict(role="calib", state=s["key"], task_id=s["task_id"], condition=category,
                 variant=pick(s["task"], category, "calib", used))
        )
    for i, f in enumerate(frames):
        f["id"] = i
    return frames
