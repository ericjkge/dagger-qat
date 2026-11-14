"""Step 4 (ORACLE proof of concept): guard-coordinated rounding vs nearest vs output-aware rounding.

Oracle: uses BF16 activations, the simulator's ∂g/∂a (finite differences) and per-state Jacobians.
Not deployable as-is; it tests whether *choosing roundings on the fixed 4-bit grid* can preserve a
contact guard better than nearest rounding and better than a generic action-error-aware rounding.

Per state (from Step-3 output) and its single activation group G at flow step k (the single-group probe
with the largest observed |Δg_end|):
  * only G@k is quantized (per-token asymmetric 4-bit, grid fixed: same scale/zero/bit width);
  * the K coordinates with the largest |M_i (ĥ_i - h_i)| are re-rounded to floor or ceil codes;
    all other coordinates keep nearest rounding.
  methods: nearest | guard (greedy on |g_lin error|) | action (greedy on ||J (ĥ-h)||² in normalized
           action units, a generic output-aware coupled rounding) | brute (exhaustive over 2^K for the
           guard objective; diagnostic of the greedy guard method only, K <= 14)
  metric: ACTUAL |Δg_end| from the simulator (not the linearized objective), plus actual ||Δa||.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def greedy(obj, lo, hi, base):
    """Coordinate-wise greedy over binary choices starting from `base` (nearest)."""
    choice = base.copy()
    best = obj(choice)
    improved = True
    while improved:
        improved = False
        for i in range(len(choice)):
            choice[i] = hi[i] if choice[i] == lo[i] else lo[i]
            val = obj(choice)
            if val < best - 1e-15:
                best, improved = val, True
            else:
                choice[i] = hi[i] if choice[i] == lo[i] else lo[i]
    return choice, best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--mechanism", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--max-states", type=int, default=40)
    args = p.parse_args()

    import time

    import torch

    from dqat.pi05 import sim
    from dqat.pi05.actquant import ActivationController, QuantConfig, grid_params, nearest_codes
    from dqat.pi05.phases import guard_values
    from dqat.pi05.policy import REPLAN_STEPS, Pi05

    meta = json.loads((args.sweep / "meta.json").read_text())
    rows = [json.loads(line) for line in (args.mechanism / "mechanism.jsonl").open()]
    rows = [r for r in rows if r["phase"] in ("pre_contact", "transition")]
    cand = []
    for r in rows:
        singles = {k: v for k, v in r["arms"].items() if k.startswith("single:")}
        name, best = max(singles.items(), key=lambda kv: abs(kv[1]["dg_end"]))
        if abs(best["dg_end"]) > 1e-5:
            cand.append((abs(best["dg_end"]), r, name))
    cand.sort(key=lambda c: -c[0])
    cand = cand[: args.max_states]

    suite_name, suite, task_id = sim.find_task(meta["task"])
    pi = Pi05(args.checkpoint)
    ctrl = ActivationController(pi.model).install()
    env = sim.make_env(suite.get_task(task_id))
    guards = sim.Guards(env)
    scale = pi.out_scale[:REPLAN_STEPS]
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "rounding.jsonl").open("a", buffering=1)

    for _, r, name in cand:
        t0 = time.time()
        group, step = name.split(":")[1].split("@")
        step = int(step)
        z = np.load(args.sweep / f"snaps_ep{r['ep']:03d}.npz")
        i = int(np.where(z["call"] == r["call"])[0][0])
        snap = sim.Snapshot(z["mj"][i], z["grip"][i], float(z["time"][i]), int(z["step"][i]))
        element = {"observation/image": z["img"][i], "observation/wrist_image": z["wrist"][i],
                   "observation/state": z["state"][i], "prompt": meta["prompt"]}
        noise = z["noise"][i]
        key = r["guard"]
        grad_end = np.asarray(r["grad_end"])  # ∂g_end/∂a (5x7), from Step 3 finite differences

        ctrl.quant, ctrl.rounder, ctrl.inject = None, None, None
        ctrl.begin_call()
        a_ref = pi.infer(element, noise)
        g_ref = guard_values(sim.branch(env, guards, snap, a_ref[:REPLAN_STEPS]), key)["end"]

        # Jacobian of the 35 executed action coords wrt this group's input (BF16 graph)
        ctrl.probe, ctrl.record = {(group, step)}, {(group, step)}
        ctrl.begin_call()
        out = pi.actions_with_grad(element, noise)
        leaf = ctrl.leaves[(group, step)]
        h = ctrl.recorded[(group, step)][0][0]  # (tokens, dim) fp32
        flat = out[:REPLAN_STEPS].reshape(-1)
        J = torch.stack([torch.autograd.grad(flat[j], leaf, retain_graph=True)[0][0] for j in range(flat.numel())])
        ctrl.probe, ctrl.record, ctrl.leaves, ctrl.recorded = None, None, {}, {}
        del out, flat
        Jn = (J / torch.from_numpy(scale.reshape(-1, 1, 1)).to(J)).reshape(J.shape[0], -1)  # normalized units
        M = torch.einsum("j,jtd->td", torch.from_numpy(grad_end.reshape(-1)).to(J), J)  # ∂g/∂h

        s, zp, qmax = grid_params(h, 4)
        near = nearest_codes(h, s, zp, qmax)
        lo = torch.clamp(torch.floor(h / s) + zp, 0, qmax)
        hi = torch.clamp(lo + 1, 0, qmax)
        dh_near = (near - zp) * s - h
        contrib = (M * dh_near).abs().reshape(-1)
        idx = torch.argsort(contrib, descending=True)[: args.k]
        flat_lo, flat_hi, flat_near = lo.reshape(-1)[idx], hi.reshape(-1)[idx], near.reshape(-1)[idx]
        s_flat = s.expand_as(h).reshape(-1)[idx]
        zp_flat = zp.expand_as(h).reshape(-1)[idx]
        h_flat = h.reshape(-1)[idx]
        dh_rest = dh_near.reshape(-1).clone()
        dh_rest[idx] = 0
        Mf = M.reshape(-1)
        g_rest = float((Mf * dh_rest).sum())
        a_rest = Jn @ dh_rest
        Mi, Ji = Mf[idx], Jn[:, idx]

        def dh_of(choice):
            return (torch.as_tensor(choice, dtype=h.dtype, device=h.device) - zp_flat) * s_flat - h_flat

        def guard_obj(choice):
            return abs(g_rest + float((Mi * dh_of(choice)).sum()))

        def action_obj(choice):
            return float((a_rest + Ji @ dh_of(choice)).pow(2).sum())

        lo_np, hi_np, near_np = (x.cpu().numpy() for x in (flat_lo, flat_hi, flat_near))
        choices = {"nearest": near_np.copy()}
        choices["guard"], _ = greedy(guard_obj, lo_np, hi_np, near_np.copy())
        choices["action"], _ = greedy(action_obj, lo_np, hi_np, near_np.copy())
        if args.k <= 14:
            best, best_val = None, np.inf
            for bits_ in itertools.product((0, 1), repeat=args.k):
                c = np.where(np.array(bits_) == 1, hi_np, lo_np)
                v = guard_obj(c)
                if v < best_val:
                    best, best_val = c, v
            choices["brute"] = best

        res = dict(ep=r["ep"], call=r["call"], phase=r["phase"], guard=key, group=group, step=step, k=args.k,
                   g_ref=g_ref, methods={})
        for mname, c in choices.items():
            codes = near.clone().reshape(-1)
            codes[idx] = torch.as_tensor(c, dtype=codes.dtype, device=codes.device)
            codes = codes.reshape(h.shape)

            def rounder(g_, k_, h_, scale_, zero_, qmax_, nearest_, codes=codes):
                return codes[None].to(nearest_) if (g_, k_) == (group, step) else nearest_

            for background in ("single", "a4_all"):
                ctrl.quant = (QuantConfig(bits=4, groups={group}, steps={step}) if background == "single"
                              else QuantConfig(bits=4))
                ctrl.rounder = rounder
                ctrl.begin_call()
                a_q = pi.infer(element, noise)
                ctrl.quant, ctrl.rounder = None, None
                g_q = guard_values(sim.branch(env, guards, snap, a_q[:REPLAN_STEPS]), key)["end"]
                res["methods"][f"{mname}/{background}"] = dict(
                    dg_actual=g_q - g_ref, guard_obj=guard_obj(c), action_obj=action_obj(c),
                    da_norm=float(np.linalg.norm((a_q[:REPLAN_STEPS] - a_ref[:REPLAN_STEPS]) / scale)),
                    sign_preserved=bool(np.sign(g_q) == np.sign(g_ref)))
        res["seconds"] = time.time() - t0
        log.write(json.dumps(res) + "\n")
        print(json.dumps({k: (v if k != "methods" else {m: round(x["dg_actual"], 6) for m, x in v.items()})
                          for k, v in res.items()}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
