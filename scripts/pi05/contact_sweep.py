"""Step 2: paired one-step counterfactuals along BF16 reference rollouts (+ optional closed-loop arm).

For every policy call on a BF16 closed-loop rollout (state x_k, observation o_k, noise ε_k):
  * a_ref  = π_BF16(o_k, ε_k)                    (10x7 env-unit chunk)
  * a_cfg  = π_cfg(o_k, ε_k) for each activation config (identical o_k, ε_k: paired)
  * a_rand = a_ref + random direction with ||a_rand - a_ref|| = ||a_cfg - a_ref|| in normalized action
             space (magnitude-matched control), one per config
  * each executed prefix (first 5 actions) is simulated from the SAME snapshot of x_k with physics-only
    stepping, recording the guard trace (6 readings: before + after each action).
The reference rollout then continues from x_k with a_ref[:5] via env.step (rendered), exactly as openpi's
LIBERO client. These are ONE-STEP counterfactuals; the quantized policy never drives the state.

`--closed-loop CFG` instead runs full rollouts where the CFG-quantized policy drives the state, with the
same initial states, env seed and per-call noise schedule as the BF16 rollouts (task success + traces).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CONFIGS = {
    "a4_all": dict(bits=4),
    "a8_all": dict(bits=8),
    "a4_down": dict(bits=4, kinds={"down"}),
    "a4_nodown": dict(bits=4, kinds={"qkv", "o", "gateup"}),
    # spin-off: QuaRot-style randomized Hadamard rotation before the 4-bit grid (no calibration data)
    "a4_rotdown": dict(bits=4, rotate={"down"}),
    "a4_rot": dict(bits=4, rotate={"qkv", "o", "gateup", "down"}),
}


def quant_config(name):
    from dqat.pi05.actquant import QuantConfig

    return QuantConfig(**CONFIGS[name])


def noise_seed(task_idx, episode, call):
    return 1_000_000 * (task_idx + 1) + 1000 * episode + call


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--first-episode", type=int, default=0)
    p.add_argument("--configs", default="a4_all,a8_all,a4_down,a4_nodown")
    p.add_argument("--closed-loop", default=None, help="config name; run quantized-policy rollouts instead")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    import torch

    from dqat.pi05 import sim
    from dqat.pi05.actquant import ActivationController
    from dqat.pi05.policy import REPLAN_STEPS, Pi05, make_element

    args.out.mkdir(parents=True, exist_ok=True)
    task_names = list(sim.TASKS)
    task_idx = task_names.index(args.task)
    suite_name, suite, task_id = sim.find_task(args.task)
    task = suite.get_task(task_id)
    init_states = sim.init_states(task)
    max_steps = sim.MAX_STEPS[suite_name]

    t0 = time.time()
    pi = Pi05(args.checkpoint)
    ctrl = ActivationController(pi.model).install()
    env = sim.make_env(task, seed=args.seed)
    guards = sim.Guards(env)
    configs = [c for c in args.configs.split(",") if c]
    drive = quant_config(args.closed_loop) if args.closed_loop else None
    meta = dict(task=args.task, suite=suite_name, task_id=task_id, prompt=task.language, configs=configs,
                closed_loop=args.closed_loop, config_defs={k: {kk: sorted(vv) if isinstance(vv, set) else vv
                                                               for kk, vv in v.items()} for k, v in CONFIGS.items()},
                objects=guards.names, has_geom_distance=guards.has_geom_distance,
                n_fingers=len(guards.fingers), n_obj=len(guards.obj), n_place=len(guards.place),
                gpu=torch.cuda.get_device_name(), load_seconds=time.time() - t0)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    rng = np.random.default_rng(12345 + task_idx)
    scale_norm = pi.out_scale[:REPLAN_STEPS]  # env units per normalized unit (5x7)

    states_log = (args.out / "states.jsonl").open("a", buffering=1)
    episodes_log = (args.out / "episodes.jsonl").open("a", buffering=1)
    checked_equivalence = 0
    for ep in range(args.first_episode, args.first_episode + args.episodes):
        ep_t0 = time.time()
        torch.manual_seed(args.seed + ep)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(sim.SETTLE_STEPS):
            obs, _, _, _ = env.step(sim.DUMMY_ACTION)
        if ep == args.first_episode:
            snap = sim.snapshot(env)
            sim.verify_branch_determinism(env, guards, snap, [sim.DUMMY_ACTION] * 5)
            sim.restore(env, snap)
        t, call, done = 0, 0, False
        snaps = []  # exact replay material for Step 3 (sim snapshot + policy inputs + noise)
        events = []  # per env step guard reading on the driving trajectory
        while t < max_steps and not done:
            snap = sim.snapshot(env)
            g0 = guards.read()
            element = make_element(obs, task.language)
            noise = pi.noise(noise_seed(task_idx, ep, call))
            ctrl.quant = None
            ctrl.begin_call()
            a_ref = pi.infer(element, noise)
            row = dict(ep=ep, call=call, t=t, g0=g0, a_ref=a_ref)
            if drive is None:
                snaps.append(dict(mj=snap.mj_state, grip=snap.gripper_action, time=snap.cur_time,
                                  step=snap.timestep, img=element["observation/image"],
                                  wrist=element["observation/wrist_image"], state=element["observation/state"],
                                  noise=noise, call=call))
                tr_ref = sim.branch(env, guards, snap, a_ref[:REPLAN_STEPS])
                row["trace_ref"] = tr_ref
                end_ref = env.env.sim.get_state().flatten().copy()
                for name in configs:
                    ctrl.quant = quant_config(name)
                    ctrl.begin_call()
                    tq = time.time()
                    a_q = pi.infer(element, noise)
                    infer_s = time.time() - tq
                    mse, rel = ctrl.activation_mse()
                    per_group = {g: v[0] / v[2] for g, v in ctrl.stats.items()}
                    ctrl.quant = None
                    d_norm = (a_q[:REPLAN_STEPS] - a_ref[:REPLAN_STEPS]) / scale_norm
                    mag = float(np.linalg.norm(d_norm))
                    u = rng.standard_normal(d_norm.shape)
                    u *= mag / max(np.linalg.norm(u), 1e-12)
                    a_rand = a_ref[:REPLAN_STEPS] + u * scale_norm
                    row[name] = dict(
                        a=a_q, act_mse=mse, act_rel_mse=rel, per_group_mse=per_group, infer_s=infer_s,
                        dnorm=mag, trace=sim.branch(env, guards, snap, a_q[:REPLAN_STEPS]),
                        trace_rand=sim.branch(env, guards, snap, a_rand), a_rand=a_rand,
                    )
                states_log.write(json.dumps(jsonable(row)) + "\n")
                sim.restore(env, snap)
                executed = a_ref
            else:
                ctrl.quant = drive
                ctrl.begin_call()
                executed = pi.infer(element, noise)
                ctrl.quant = None
                states_log.write(json.dumps(jsonable(dict(ep=ep, call=call, t=t, g0=g0, a_ref=a_ref,
                                                          a_drive=executed))) + "\n")
            for k in range(REPLAN_STEPS):
                obs, _, done, _ = env.step(np.asarray(executed[k]).tolist())
                t += 1
                events.append(dict(t=t, **{key: val for key, val in guards.read().items()
                                           if key in ("d_grasp", "c_grasp", "grasped", "d_place", "c_place")}))
                if done or t >= max_steps:
                    break
            if drive is None and checked_equivalence < 3 and k == REPLAN_STEPS - 1:
                # physics_step branch == rendered env.step trajectory (same actions, same snapshot)
                now = env.env.sim.get_state().flatten()
                err = float(np.abs(now - end_ref).max())
                print(f"[check] branch vs env.step max|dq|={err:.3g}", flush=True)
                if err > 1e-6:
                    raise RuntimeError("physics_step branch diverges from env.step")
                checked_equivalence += 1
            call += 1
        rec = dict(ep=ep, success=bool(done), steps=t, calls=call, seconds=time.time() - ep_t0,
                   arm=args.closed_loop or "bf16", events=events)
        episodes_log.write(json.dumps(jsonable(rec)) + "\n")
        if snaps:
            np.savez_compressed(args.out / f"snaps_ep{ep:03d}.npz",
                                **{key: np.stack([s[key] for s in snaps]) for key in snaps[0]})
        print(json.dumps({k: v for k, v in rec.items() if k != "events"}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
