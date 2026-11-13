"""Tag phases on the LIBERO-Long demonstrations OpenVLA-OFT was fine-tuned on
(openvla/modified_libero_rlds, libero_10_no_noops) and draw a task-balanced,
phase-balanced frame set for the gradient step.

Outputs under --out:
  config.json   every setting used, dataset/checkpoint provenance, git state
  phases.json   per-episode flips and phase lengths, per-task phase statistics
  frames.npz    sampled frames: 256px agentview + wrist images (model
                orientation), 8D state, instruction, task, episode, step, phase,
                demo action chunk (next 8 actions, raw -1 open / +1 close)
One summary line is appended to research/runs/phase-and-action-sensitive.jsonl.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SLUG = "phase-and-action-sensitive"
ROOT = Path(__file__).resolve().parents[3]
PHASES = ("approach", "grasp", "transport", "place")


def tag_phases(gripper_cmd, window, min_hold):
    """Per-step phase labels from the gripper command (-1 open, +1 close).

    A flip counts only if the new command holds for min_hold steps. Steps within
    `window` of a close flip are grasp, within `window` of an open flip are
    place; remaining steps are approach while open and transport while closed.
    """
    closed = gripper_cmd > 0
    flips, state = [], closed[0]
    for t in range(len(closed)):
        if closed[t] != state and np.all(closed[t : t + min_hold] == closed[t]):
            flips.append((t, "close" if closed[t] else "open"))
            state = closed[t]
    labels = np.empty(len(closed), dtype="<U9")
    state, last = closed[0], 0
    for t, kind in flips:
        labels[last:t] = "transport" if state else "approach"
        state, last = kind == "close", t
    labels[last:] = "transport" if state else "approach"
    for t, kind in flips:
        labels[max(0, t - window) : t + window + 1] = "grasp" if kind == "close" else "place"
    return labels, flips


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rlds-dir", type=Path, required=True, help=".../libero_10_no_noops/1.0.0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase-window", type=int, required=True)
    p.add_argument("--flip-min-hold", type=int, required=True)
    p.add_argument("--frames-per-phase", type=int, required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    git = lambda *a: subprocess.check_output(("git", *a), text=True, cwd=ROOT).strip()
    config = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_source": json.loads((args.rlds_dir / "source.json").read_text()),
        "frame_source": "OFT fine-tuning demonstrations (regenerated 256px, no-op filtered), train split, all episodes",
        "gripper_command_convention": "raw RLDS action[-1]: -1 open, +1 close",
        "phase_rule": "debounced flips of demo gripper command; grasp/place = +-window steps around close/open flips; else approach (open) / transport (closed)",
        "sampling": "per phase: frames_per_phase frames, split as evenly as possible over tasks that have that phase, uniform without replacement within task; chunk must fit inside the episode",
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "submodules": git("submodule", "status"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Two passes over the same ordered split: tag phases from actions only, then
    # load images just for the chosen frames (all images would not fit in RAM).
    builder = tfds.builder_from_directory(str(args.rlds_dir))
    dataset = lambda: builder.as_dataset(split="train", shuffle_files=False)
    episodes, pool, actions_of = [], [], []  # pool: (task, phase, episode index, step)
    instructions = {}
    for e, ep in enumerate(dataset()):
        instruction = next(iter(ep["steps"].take(1)))["language_instruction"].numpy().decode()
        task = instructions.setdefault(instruction, len(instructions))
        actions = np.stack([a.numpy() for a in ep["steps"].map(lambda s: s["action"])]).astype(np.float32)
        labels, flips = tag_phases(actions[:, -1], args.phase_window, args.flip_min_hold)
        actions_of.append(actions)
        episodes.append(dict(
            episode=e, task=task, instruction=instruction, steps=len(actions),
            flips=[(int(t), k) for t, k in flips],
            steps_per_phase={ph: int((labels == ph).sum()) for ph in PHASES},
        ))
        for t in range(len(actions) - args.chunk + 1):
            pool.append((task, labels[t], e, t))

    rng = np.random.default_rng(args.seed)
    chosen = []
    for ph in PHASES:
        by_task = {}
        for task, lab, e, t in pool:
            if lab == ph:
                by_task.setdefault(task, []).append((e, t))
        tasks = sorted(by_task)
        quota = [args.frames_per_phase // len(tasks) + (i < args.frames_per_phase % len(tasks)) for i in range(len(tasks))]
        for task, q in zip(tasks, quota):
            cand = by_task[task]
            for j in rng.choice(len(cand), size=min(q, len(cand)), replace=False):
                chosen.append((task, ph, *cand[j]))

    wanted = {(e, t) for _, _, e, t in chosen}
    loaded = {}
    for e, ep in enumerate(dataset()):
        if not any(k[0] == e for k in wanted):
            continue
        for t, s in enumerate(ep["steps"].as_numpy_iterator()):
            if (e, t) in wanted:
                o = s["observation"]
                loaded[e, t] = (o["image"], o["wrist_image"], o["state"])
    imgs = [loaded[e, t] for _, _, e, t in chosen]
    np.savez_compressed(
        args.out / "frames.npz",
        full_image=np.stack([i[0] for i in imgs]),
        wrist_image=np.stack([i[1] for i in imgs]),
        state=np.stack([i[2] for i in imgs]).astype(np.float32),
        instruction=np.array([episodes[e]["instruction"] for _, _, e, _ in chosen]),
        task=np.array([c[0] for c in chosen]),
        phase=np.array([c[1] for c in chosen]),
        episode=np.array([c[2] for c in chosen]),
        step=np.array([c[3] for c in chosen]),
        demo_chunk=np.stack([actions_of[e][t : t + args.chunk] for _, _, e, t in chosen]),
    )

    per_task = {}
    for task in range(len(instructions)):
        eps = [x for x in episodes if x["task"] == task]
        per_task[str(task)] = dict(
            instruction=eps[0]["instruction"],
            episodes=len(eps),
            close_flips=np.bincount([sum(k == "close" for _, k in x["flips"]) for x in eps]).tolist(),
            open_flips=np.bincount([sum(k == "open" for _, k in x["flips"]) for x in eps]).tolist(),
            steps_per_phase={ph: sum(x["steps_per_phase"][ph] for x in eps) for ph in PHASES},
            frames_chosen={ph: sum(c[0] == task and c[1] == ph for c in chosen) for ph in PHASES},
        )
    (args.out / "phases.json").write_text(json.dumps(dict(per_task=per_task, episodes=episodes), indent=2) + "\n")
    record = dict(
        slug=SLUG,
        kind="demo_frames",
        job=os.environ.get("SLURM_JOB_ID"),
        out=str(args.out),
        git_commit=config["git_commit"],
        dataset=config["dataset_source"],
        episodes=len(episodes),
        tasks=len(instructions),
        steps_per_phase={ph: sum(x["steps_per_phase"][ph] for x in episodes) for ph in PHASES},
        frames_per_phase={ph: sum(c[1] == ph for c in chosen) for ph in PHASES},
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(ROOT / "research" / "runs" / f"{SLUG}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
