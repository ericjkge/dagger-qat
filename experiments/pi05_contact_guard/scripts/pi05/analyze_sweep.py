"""Step-2 analysis: contact-event flip rates by phase, controlling for perturbation magnitude and distance.

Inputs: one or more contact_sweep output dirs (states.jsonl). Uncertainty: cluster bootstrap over
episodes (states within an episode are not independent). Outputs a JSON summary and PNG plots.

Matched-magnitude control: for each state and config, `trace_rand` executes a_ref + u with
||u||_norm == ||a_cfg - a_ref||_norm in a random direction. Comparing flip(cfg) to flip(rand) within a
phase tests whether quantization-induced directions are special beyond their size; comparing phases
at matched ||Δa|| (bins) and |g0| tests the phase hypothesis itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHASES = ("free", "pre_contact", "transition", "holding", "contact")


def load(dirs):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from dqat.pi05.phases import active_guard, event_compare, phase

    rows = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        for line in (d / "states.jsonl").open():
            r = json.loads(line)
            ph = phase(r["trace_ref"])
            key = active_guard(r["g0"])
            base = dict(task=meta["task"], ep=r["ep"], call=r["call"], phase=ph, guard=key,
                        dist=r["g0"][key], grasped=r["g0"]["grasped"])
            for cfg in meta["configs"]:
                c = r[cfg]
                for arm, tr in (("q", c["trace"]), ("rand", c["trace_rand"])):
                    ev = event_compare(r["trace_ref"], tr)
                    g_ref_end = r["trace_ref"][-1][key]
                    rows.append(dict(base, cfg=cfg, arm=arm, dnorm=c["dnorm"], act_rel_mse=c["act_rel_mse"],
                                     flip=ev["any_flip"], change=ev["any_change"],
                                     flip_grasp=ev["grasped"]["flip"], flip_cgrasp=ev["c_grasp"]["flip"],
                                     flip_cplace=ev["c_place"]["flip"],
                                     dg_end=tr[-1][key] - g_ref_end,
                                     grip_flip=bool(np.any(np.sign(np.asarray(c["a"])[:5, 6]) !=
                                                           np.sign(np.asarray(r["a_ref"])[:5, 6])))
                                     if arm == "q" else None))
    return rows


def cluster_boot(rows, stat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    clusters = {}
    for r in rows:
        clusters.setdefault((r["task"], r["ep"]), []).append(r)
    keys = list(clusters)
    if not keys:
        return (np.nan, np.nan, np.nan)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [x for i in pick for x in clusters[keys[i]]]
        vals.append(stat(sample))
    return (stat(rows), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def rate(field):
    return lambda rs: float(np.mean([r[field] for r in rs])) if rs else np.nan


def logistic(X, y, iters=50, l2=1e-3):
    """Plain IRLS logistic regression with a tiny ridge; returns coefficients."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(len(w))
        w = w + np.linalg.solve(H, X.T @ (y - p) - l2 * w)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.dirs)
    args.out.mkdir(parents=True, exist_ok=True)
    cfgs = sorted({r["cfg"] for r in rows})
    summary = {"n_states": len({(r["task"], r["ep"], r["call"]) for r in rows}),
               "phase_counts": {}, "by_cfg": {}}
    for ph in PHASES:
        summary["phase_counts"][ph] = len({(r["task"], r["ep"], r["call"]) for r in rows if r["phase"] == ph})
    for cfg in cfgs:
        out = {}
        for ph in PHASES:
            q = [r for r in rows if r["cfg"] == cfg and r["arm"] == "q" and r["phase"] == ph]
            rd = [r for r in rows if r["cfg"] == cfg and r["arm"] == "rand" and r["phase"] == ph]
            if not q:
                continue
            out[ph] = dict(n=len(q), flip_q=cluster_boot(q, rate("flip")), flip_rand=cluster_boot(rd, rate("flip")),
                           change_q=cluster_boot(q, rate("change")),
                           flip_grasp_q=rate("flip_grasp")(q), flip_cplace_q=rate("flip_cplace")(q),
                           grip_sign_flip=rate("grip_flip")(q),
                           dnorm_median=float(np.median([r["dnorm"] for r in q])),
                           act_rel_mse_median=float(np.median([r["act_rel_mse"] for r in q])),
                           abs_dg_end_median=float(np.median([abs(r["dg_end"]) for r in q])))
        # Logistic: flip ~ 1 + phase dummies(pre_contact, transition, holding) + log dnorm + log dist + rand
        sub = [r for r in rows if r["cfg"] == cfg and r["phase"] in ("free", "pre_contact", "transition", "holding")]
        if len(sub) > 50 and any(r["flip"] for r in sub):
            X = np.array([[1, r["phase"] == "pre_contact", r["phase"] == "transition", r["phase"] == "holding",
                           np.log(r["dnorm"] + 1e-6), np.log(max(r["dist"], 1e-4)), r["arm"] == "rand"]
                          for r in sub], dtype=float)
            y = np.array([r["flip"] for r in sub], dtype=float)
            out["logit_coef"] = dict(zip(["const", "pre_contact", "transition", "holding", "log_dnorm",
                                          "log_dist", "rand_arm"], logistic(X, y).round(3).tolist()))
        summary["by_cfg"][cfg] = out

    # Plot 1: flip rate by phase, quantized vs magnitude-matched random, per config
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.2 * len(cfgs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cfg in zip(axes, cfgs):
        phs = [ph for ph in PHASES if ph in summary["by_cfg"][cfg]]
        x = np.arange(len(phs))
        for off, arm, color in ((-0.18, "flip_q", "#1f6f8b"), (0.18, "flip_rand", "#c9a227")):
            m = [summary["by_cfg"][cfg][ph][arm] for ph in phs]
            ax.bar(x + off, [v[0] for v in m], 0.34, color=color, label="quantized" if arm == "flip_q" else
                   "random, same ||Δa||")
            ax.errorbar(x + off, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]],
                        fmt="none", ecolor="k", lw=0.8, capsize=2)
        ax.set_xticks(x, [f"{ph}\n(n={summary['by_cfg'][cfg][ph]['n']})" for ph in phs], fontsize=7)
        ax.set_title(cfg)
    axes[0].set_ylabel("contact-event flip rate (5-step window)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "flip_by_phase.png", dpi=150)

    # Plot 2: flip probability vs perturbation magnitude, per phase (all configs pooled, quantized arm)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    q = [r for r in rows if r["arm"] == "q"]
    edges = np.quantile([r["dnorm"] for r in q], np.linspace(0, 1, 7))
    for ph, color in zip(("free", "pre_contact", "transition", "holding"), ("#888", "#1f6f8b", "#b03a2e", "#6c8e3b")):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = [r for r in q if r["phase"] == ph and lo <= r["dnorm"] <= hi]
            if len(b) >= 10:
                xs.append(np.median([r["dnorm"] for r in b]))
                ys.append(np.mean([r["flip"] for r in b]))
        ax.plot(xs, ys, "o-", color=color, label=ph)
    ax.set_xscale("log")
    ax.set_xlabel("||Δa|| executed prefix (normalized action units)")
    ax.set_ylabel("flip rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "flip_vs_magnitude.png", dpi=150)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
