"""Step 3: does the linearization Δg ≈ M (ĥ − h), M = (∂g/∂a)(∂a/∂h), predict real guard changes?

Replays saved BF16-rollout states exactly (sim snapshot + policy inputs + flow noise from Step 2) and,
per state:
  1. ∂g/∂a by central finite differences in the simulator (g is not differentiable: MuJoCo contact,
     and the Panda gripper integrates sign(a_grip)), for the active guard of the state, two read-outs:
       g_end = guard value after the 5 executed actions;  g_min = min over the 5-step window.
  2. ∂a/∂h for every quantizable activation (72 groups x 10 flow steps) by autograd through the BF16
     action expert: one backward per executed action coordinate (35), each contracted with the
     quantization error ĥ − h of every group (ĥ = nearest rounding on the per-token 4-bit grid; and 8-bit).
     This gives J·Δh per group: the model-side linear prediction of Δa.
  3. Actual quantized runs (identical inputs/noise): configs a4_all/a4_down/a4_nodown/a8_all and a set of
     single-group probes; each executed prefix simulated from the same snapshot.
Predictions recorded: model-side  Δa_pred = Σ_G J_G Δh_G   vs  Δa_actual
                      sim-side    Δg_sim  = ∇_a g · Δa_actual vs  Δg_actual
                      full chain  Δg_pred = ∇_a g · Δa_pred   vs  Δg_actual
and linearized sign preservation sign(g_ref)·(g_ref + Δg_pred) > 0 vs the observed sign.
Δh uses the FP value of each input (first-order: errors of upstream groups do not alter downstream h).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SINGLE_PROBE_LAYERS = (0, 9, 17)
SINGLE_PROBE_STEPS = (0, 9)
CONFIG_KINDS = {"a4_all": (4, None), "a4_down": (4, {"down"}), "a4_nodown": (4, {"qkv", "o", "gateup"}),
                "a8_all": (8, None)}


def select_states(sweep_dir, per_phase, seed):
    """Stratified sample of states by phase from the Step-2 log (BF16 reference trace)."""
    from dqat.pi05.phases import phase

    rows = [json.loads(line) for line in (sweep_dir / "states.jsonl").open()]
    # only episodes whose replay snapshots are on disk (the sweep writes them when an episode ends)
    rows = [r for r in rows if (sweep_dir / f"snaps_ep{r['ep']:03d}.npz").exists()]
    by_phase = {}
    for r in rows:
        by_phase.setdefault(phase(r["trace_ref"]), []).append(r)
    rng = np.random.default_rng(seed)
    chosen = []
    for ph, rs in sorted(by_phase.items()):
        idx = rng.permutation(len(rs))[:per_phase]
        chosen += [(ph, rs[i]) for i in sorted(idx)]
    return chosen, {k: len(v) for k, v in by_phase.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-phase", type=int, default=25)
    p.add_argument("--fd-delta", type=float, default=0.02, help="central-difference step, normalized units")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    from dqat.pi05 import sim
    from dqat.pi05.actquant import ActivationController, QuantConfig, fake_quant
    from dqat.pi05.phases import active_guard, event_compare, guard_values
    from dqat.pi05.policy import REPLAN_STEPS, Pi05

    meta = json.loads((args.sweep / "meta.json").read_text())
    suite_name, suite, task_id = sim.find_task(meta["task"])
    task = suite.get_task(task_id)
    pi = Pi05(args.checkpoint)
    ctrl = ActivationController(pi.model).install()
    env = sim.make_env(task)
    guards = sim.Guards(env)
    chosen, phase_counts = select_states(args.sweep, args.per_phase, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps(dict(sweep=str(args.sweep), task=meta["task"],
                                                        phase_counts=phase_counts, n=len(chosen),
                                                        fd_delta=args.fd_delta), indent=1))
    log = (args.out / "mechanism.jsonl").open("a", buffering=1)
    scale = pi.out_scale[:REPLAN_STEPS]  # (5,7) env units per normalized unit
    npz_cache = {}

    def run_branch(snap, actions, key):
        tr = sim.branch(env, guards, snap, actions)
        return tr, guard_values(tr, key)

    for ph, row in chosen:
        t0 = time.time()
        ep, call = row["ep"], row["call"]
        if ep not in npz_cache:
            npz_cache.clear()
            npz_cache[ep] = dict(np.load(args.sweep / f"snaps_ep{ep:03d}.npz"))
        z = npz_cache[ep]
        i = int(np.where(z["call"] == call)[0][0])
        snap = sim.Snapshot(z["mj"][i], z["grip"][i], float(z["time"][i]), int(z["step"][i]))
        element = {"observation/image": z["img"][i], "observation/wrist_image": z["wrist"][i],
                   "observation/state": z["state"][i], "prompt": meta["prompt"]}
        noise = z["noise"][i]
        g0 = row["g0"]
        key = active_guard(g0)

        ctrl.quant, ctrl.probe, ctrl.record, ctrl.inject = None, None, None, None
        ctrl.begin_call()
        a_ref = pi.infer(element, noise)
        replay_err = float(np.abs(a_ref - np.asarray(row["a_ref"])).max())
        tr_ref, gv_ref = run_branch(snap, a_ref[:REPLAN_STEPS], key)

        # 1. ∂g/∂a by central differences (env units)
        grad = {w: np.zeros((REPLAN_STEPS, 7)) for w in ("end", "min")}
        for s in range(REPLAN_STEPS):
            for d in range(7):
                h = args.fd_delta * scale[s, d]
                vals = {}
                for sign in (1, -1):
                    a = a_ref[:REPLAN_STEPS].copy()
                    a[s, d] += sign * h
                    vals[sign] = run_branch(snap, a, key)[1]
                for w in ("end", "min"):
                    grad[w][s, d] = (vals[1][w] - vals[-1][w]) / (2 * h)

        # 2. J·Δh for all (group, step) via 35 backward passes through the BF16 expert
        ctrl.probe, ctrl.record = "all", "all"
        ctrl.begin_call()
        out = pi.actions_with_grad(element, noise)
        grad_path_err = float(np.abs(out.detach().cpu().numpy() - a_ref).max())
        keys = sorted(ctrl.leaves)
        leaves = [ctrl.leaves[k] for k in keys]
        dh = {b: [fake_quant(ctrl.recorded[k][0], b) - ctrl.recorded[k][0] for k in keys] for b in (4, 8)}
        flat = out[:REPLAN_STEPS].reshape(-1)
        contrib = {b: np.zeros((len(keys), flat.numel())) for b in (4, 8)}  # J_G Δh_G, per output coord
        zero = torch.zeros((), device=flat.device)
        for j in range(flat.numel()):
            grads = torch.autograd.grad(flat[j], leaves, retain_graph=True, allow_unused=True)
            for b in (4, 8):  # all dot products on device, one host transfer per output coordinate
                dots = torch.stack([zero if g is None else (g * d_).sum() for g, d_ in zip(grads, dh[b])])
                contrib[b][:, j] = dots.double().cpu().numpy()
        del out, flat, leaves
        ctrl.probe, ctrl.record = None, None
        ctrl.leaves, ctrl.recorded = {}, {}
        key_index = {k: n for n, k in enumerate(keys)}

        def lin_pred(selector, bits):
            rows_ = [key_index[k] for k in keys if selector(k)]
            da = contrib[bits][rows_].sum(axis=0).reshape(REPLAN_STEPS, 7)
            return da

        record = dict(ep=ep, call=call, t=row["t"], phase=ph, guard=key, g_ref=gv_ref, g0=g0[key],
                      replay_err=replay_err, grad_path_err=grad_path_err,
                      grad_end=grad["end"], grad_min=grad["min"], arms={})

        def evaluate(name, quant, selector, bits):
            ctrl.quant = quant
            ctrl.begin_call()
            a_q = pi.infer(element, noise)
            mse, rel = ctrl.activation_mse()
            ctrl.quant = None
            tr_q, gv_q = run_branch(snap, a_q[:REPLAN_STEPS], key)
            da = a_q[:REPLAN_STEPS] - a_ref[:REPLAN_STEPS]
            da_pred = lin_pred(selector, bits)
            res = dict(act_mse=mse, act_rel_mse=rel, da_norm=float(np.linalg.norm(da / scale)),
                       da_pred_norm=float(np.linalg.norm(da_pred / scale)),
                       da_cos=float(np.sum(da * da_pred / scale**2) /
                                    max(np.linalg.norm(da / scale) * np.linalg.norm(da_pred / scale), 1e-30)),
                       events=event_compare(tr_ref, tr_q), grip_sign_flip=bool(np.any(
                           np.sign(a_q[:REPLAN_STEPS, 6]) != np.sign(a_ref[:REPLAN_STEPS, 6]))))
            for w in ("end", "min"):
                res[f"dg_{w}"] = gv_q[w] - gv_ref[w]
                res[f"dg_{w}_sim"] = float(np.sum(grad[w] * da))
                res[f"dg_{w}_pred"] = float(np.sum(grad[w] * da_pred))
            record["arms"][name] = res

        for name, (bits, kinds) in CONFIG_KINDS.items():
            evaluate(name, QuantConfig(bits=bits, kinds=kinds),
                     (lambda k, kinds=kinds: kinds is None or k[0].split(".")[1] in kinds), bits)
        for layer in SINGLE_PROBE_LAYERS:
            for kind in ("qkv", "o", "gateup", "down"):
                for step in SINGLE_PROBE_STEPS:
                    g = f"L{layer}.{kind}"
                    evaluate(f"single:{g}@{step}", QuantConfig(bits=4, groups={g}, steps={step}),
                             (lambda k, g=g, step=step: k == (g, step)), 4)
        record["seconds"] = time.time() - t0
        log.write(json.dumps(record, default=lambda o: o.tolist() if hasattr(o, "tolist") else o) + "\n")
        print(json.dumps(dict(ep=ep, call=call, phase=ph, guard=key, s=round(record["seconds"], 1),
                              replay_err=replay_err, grad_path_err=grad_path_err,
                              a4=dict((k, record["arms"]["a4_all"][k]) for k in
                                      ("dg_end", "dg_end_sim", "dg_end_pred", "da_cos")))), flush=True)
    env.close()


if __name__ == "__main__":
    main()
