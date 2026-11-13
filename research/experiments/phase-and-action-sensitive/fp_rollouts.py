"""Full-precision LIBERO rollouts that keep per-step gripper traces and the
prepared observation at every policy call, then tag phases from gripper flips.

Outputs under --out:
  config.json        every setting used, plus checkpoint and git provenance
  episodes/<t>-<i>.npz  per-episode traces and policy-call observations
  phases.json        per-episode phase spans and per-task phase statistics
One summary line is appended to research/runs/phase-and-action-sensitive.jsonl.
"""

import argparse
import json
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SLUG = "phase-and-action-sensitive"
ROOT = Path(__file__).resolve().parents[3]


def tag_phases(gripper_cmd, window, min_hold):
    """Per-step phase labels from the executed gripper command (-1 open, +1 close).

    A flip counts only if the new command holds for min_hold steps. Steps within
    `window` of a close flip are grasp, within `window` of an open flip are
    place; remaining steps are approach while open and transport while closed.
    """
    closed = gripper_cmd > 0
    flips, state, t = [], closed[0], 0
    while t < len(closed):
        if closed[t] != state and np.all(closed[t : t + min_hold] == closed[t]):
            flips.append((t, "close" if closed[t] else "open"))
            state = closed[t]
        t += 1
    # Debounced open/closed state, so a brief blip does not relabel the step.
    labels = np.empty(len(closed), dtype="<U9")
    state, last = closed[0], 0
    for t, kind in flips:
        labels[last:t] = np.where(state, "transport", "approach")
        state, last = kind == "close", t
    labels[last:] = np.where(state, "transport", "approach")
    for t, kind in flips:
        labels[max(0, t - window) : t + window + 1] = "grasp" if kind == "close" else "place"
    return labels, flips


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--init-states", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--settle-steps", type=int, required=True)
    p.add_argument("--open-loop-steps", type=int, required=True)
    p.add_argument("--phase-window", type=int, required=True)
    p.add_argument("--flip-min-hold", type=int, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()

    from dqat.policy import OFTPolicy

    policy = OFTPolicy(args.checkpoint, args.seed)
    import torch
    from libero.libero import benchmark

    git = lambda *a: subprocess.check_output(("git", *a), text=True, cwd=ROOT).strip()
    config = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_source": json.loads((args.checkpoint / "source.json").read_text()),
        "policy_config": {k: str(v) for k, v in vars(policy.cfg).items()},
        "precision": "unquantized BF16",
        "environment_resolution": 256,
        "environment_seed": 0,
        "episode_seed": "seed + 1000 * task_id + init_state_id, then env.seed(0)",
        "gripper_command_convention": "-1 open, +1 close (after upstream process_action)",
        "phase_rule": "debounced flips of executed gripper command; grasp/place = +-window around close/open flips; else approach (open) / transport (closed)",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "submodules": git("submodule", "status"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    episodes = []
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, instruction = policy.upstream.get_libero_env(task, "openvla", resolution=256)
        for i in args.init_states:
            policy.upstream.set_seed_everywhere(args.seed + 1000 * task_id + i)
            env.seed(0)
            started = time.monotonic()
            env.reset()
            obs = env.set_init_state(init_states[i])
            for _ in range(args.settle_steps):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            queue, success = deque(), False
            grip_cmd, grip_qpos, call_step, chunks = [], [], [], []
            call_full, call_wrist, call_state = [], [], []
            for step in range(args.max_steps):
                if not queue:
                    prepared, _ = policy.upstream.prepare_observation(obs, policy.resize_size)
                    chunk = np.asarray(policy.predict_prepared(prepared, instruction), dtype=np.float32)
                    queue.extend(chunk[: args.open_loop_steps])
                    call_step.append(step)
                    chunks.append(chunk)
                    call_full.append(prepared["full_image"])
                    call_wrist.append(prepared["wrist_image"])
                    call_state.append(prepared["state"])
                action = policy.upstream.process_action(np.array(queue.popleft(), copy=True), "openvla")
                grip_cmd.append(action[-1])
                grip_qpos.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
            grip_cmd = np.asarray(grip_cmd, dtype=np.float32)
            labels, flips = tag_phases(grip_cmd, args.phase_window, args.flip_min_hold)
            call_step = np.asarray(call_step)
            np.savez_compressed(
                args.out / "episodes" / f"{task_id}-{i}.npz",
                gripper_cmd=grip_cmd,
                gripper_qpos=np.stack(grip_qpos),
                phase=labels,
                call_step=call_step,
                call_phase=labels[call_step],
                chunk=np.stack(chunks),
                full_image=np.stack(call_full),
                wrist_image=np.stack(call_wrist),
                state=np.stack(call_state),
            )
            ep = dict(
                task_id=task_id,
                task=task.name,
                init_state_id=i,
                success=success,
                steps=len(grip_cmd),
                policy_calls=len(call_step),
                flips=[(int(t), k) for t, k in flips],
                steps_per_phase={ph: int((labels == ph).sum()) for ph in ("approach", "grasp", "transport", "place")},
                calls_per_phase={ph: int((labels[call_step] == ph).sum()) for ph in ("approach", "grasp", "transport", "place")},
                seconds=round(time.monotonic() - started, 1),
            )
            episodes.append(ep)
            print(json.dumps(ep), flush=True)
        env.close()

    per_task = {}
    for task_id in args.tasks:
        eps = [e for e in episodes if e["task_id"] == task_id]
        per_task[str(task_id)] = dict(
            successes=sum(e["success"] for e in eps),
            episodes=len(eps),
            close_flips=[sum(k == "close" for _, k in e["flips"]) for e in eps],
            open_flips=[sum(k == "open" for _, k in e["flips"]) for e in eps],
            calls_per_phase_successful={
                ph: sum(e["calls_per_phase"][ph] for e in eps if e["success"])
                for ph in ("approach", "grasp", "transport", "place")
            },
        )
    (args.out / "phases.json").write_text(json.dumps(dict(episodes=episodes, per_task=per_task), indent=2) + "\n")
    record = dict(
        slug=SLUG,
        kind="fp_rollouts",
        job=__import__("os").environ.get("SLURM_JOB_ID"),
        out=str(args.out),
        git_commit=config["git_commit"],
        suite=args.suite,
        tasks=args.tasks,
        init_states=args.init_states,
        seed=args.seed,
        episodes=len(episodes),
        successes=sum(e["success"] for e in episodes),
        calls_per_phase_successful={
            ph: sum(t["calls_per_phase_successful"][ph] for t in per_task.values())
            for ph in ("approach", "grasp", "transport", "place")
        },
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(ROOT / "research" / "runs" / f"{SLUG}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
