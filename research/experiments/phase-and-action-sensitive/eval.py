"""LIBERO-Long evaluation of quantized arms, one (arm, task) unit at a time.

Several processes may run this with the same --root: each claims a unit with an
exclusive marker file. A unit whose allocation file does not exist yet is
skipped until it appears. Arms: `fp` (BF16), `uniform` (every quantizable layer
at --uniform-bits), or the name of <root>/alloc/<arm>.json from allocate.py.

Per episode: success, steps, and gripper-flip phases of the executed actions
(the same tagger as the demos), appended to <root>/eval/<arm>.jsonl. One line
per finished unit goes to research/runs/phase-and-action-sensitive.jsonl.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from demo_frames import PHASES, ROOT, SLUG, tag_phases
from quant import apply_bits, target_layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True, help="in priority order")
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--uniform-bits", type=int, required=True)
    p.add_argument("--phase-window", type=int, required=True)
    p.add_argument("--flip-min-hold", type=int, required=True)
    args = p.parse_args()
    (args.root / "eval" / "claims").mkdir(parents=True, exist_ok=True)

    from dqat.policy import OFTPolicy
    from dqat.rollout import run_episode
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    pending = [(arm, t) for arm in args.arms for t in args.tasks]
    policy, loaded = None, None
    while pending:
        unit = next((u for u in pending if u[0] in ("fp", "uniform") or (args.root / "alloc" / f"{u[0]}.json").exists()), None)
        if unit is None:
            time.sleep(60)
            continue
        pending.remove(unit)
        try:
            os.close(os.open(args.root / "eval" / "claims" / f"{unit[0]}-{unit[1]}", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        arm, task_id = unit

        if arm != loaded:
            policy = None
            torch.cuda.empty_cache()
            policy = OFTPolicy(args.checkpoint, args.seed)
            if arm == "uniform":
                apply_bits(policy.model, {n: args.uniform_bits for n in target_layers(policy.model)})
            elif arm != "fp":
                apply_bits(policy.model, json.loads((args.root / "alloc" / f"{arm}.json").read_text())["bits"])
            loaded = arm

        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, instruction = policy.upstream.get_libero_env(task, "openvla", resolution=256)
        successes = 0
        for i in range(args.episodes):
            policy.upstream.set_seed_everywhere(args.seed + 1000 * task_id + i)
            env.seed(0)
            gripper = []

            def process_action(action, family):
                action = policy.upstream.process_action(action, family)
                gripper.append(action[-1])
                return action

            started = time.monotonic()
            result = run_episode(env, policy, instruction, init_states[i], process_action)
            labels, flips = tag_phases(np.array(gripper), args.phase_window, args.flip_min_hold)
            record = dict(
                arm=arm, task_id=task_id, init_state_id=i, seed=args.seed + 1000 * task_id + i, **result,
                last_phase=str(labels[-1]), close_flips=sum(k == "close" for _, k in flips),
                open_flips=sum(k == "open" for _, k in flips),
                steps_per_phase={ph: int((labels == ph).sum()) for ph in PHASES},
                seconds=round(time.monotonic() - started, 1),
            )
            successes += result["success"]
            with open(args.root / "eval" / f"{arm}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        env.close()
        with open(ROOT / "research" / "runs" / f"{SLUG}.jsonl", "a") as f:
            f.write(json.dumps(dict(
                slug=SLUG, kind="eval_unit", job=os.environ.get("SLURM_JOB_ID"), arm=arm, task_id=task_id,
                episodes=args.episodes, successes=successes, seed=args.seed, uniform_bits=args.uniform_bits,
            )) + "\n")


if __name__ == "__main__":
    main()
