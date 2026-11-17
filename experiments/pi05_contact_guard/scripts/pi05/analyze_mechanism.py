"""Step-3 analysis: predicted vs observed guard perturbations under 4-bit (and 8-bit) activation quantization.

Three predictors of Δg (observed: simulator, same snapshot/inputs/noise):
  sim     ∇_a g · Δa_actual                    (only the simulator is linearized)
  chain   ∇_a g · Σ_G J_G Δh_G  = M (ĥ − h)     (model AND simulator linearized; the hypothesis' predictor)
and model-side agreement of Δa_pred with Δa_actual (cosine, norm ratio).

Sign preservation: predicted  sign(g_ref)·(g_ref + Δg_pred) > 0  vs observed sign(g_q) == sign(g_ref), for
g_end and g_min, excluding |g_ref| < MARGIN (ambiguous; reported separately). Distances are clipped at
DIST_MAX = 0.25 m by the guard; states with g_ref at the clip are excluded from regression metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MARGIN = 0.002  # m
CLIP = 0.2499
NEAR = 0.05  # m; "near" subset: |g_ref| < 5 cm, where the guard is dominated by one geom pair
NOISE_FLOOR = 2e-5  # m; |Δg| below this treated as zero when scoring sign agreement


def metrics(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    if len(obs) < 3:
        return dict(n=len(obs))
    keep = np.abs(obs) > NOISE_FLOOR
    r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 and np.std(obs) > 0 else np.nan
    from scipy.stats import spearmanr

    rho = float(spearmanr(obs, pred).correlation) if np.std(pred) > 0 else np.nan
    slope = float(np.sum(obs * pred) / max(np.sum(pred * pred), 1e-30))  # obs ≈ slope * pred
    r2 = 1 - float(np.sum((obs - pred) ** 2) / max(np.sum((obs - obs.mean()) ** 2), 1e-30))
    return dict(n=len(obs), pearson=r, spearman=rho, r2_identity=r2, slope_obs_on_pred=slope,
                sign_agree=float(np.mean(np.sign(obs[keep]) == np.sign(pred[keep]))) if keep.any() else np.nan,
                n_nonzero=int(keep.sum()), median_abs_obs=float(np.median(np.abs(obs))),
                median_abs_err=float(np.median(np.abs(obs - pred))),
                # robust: fraction predicted within max(0.5 mm, 30% of |obs|) (the min-distance guard is
                # discontinuous when the closest geom pair switches, which produces cm-scale outliers)
                within=float(np.mean(np.abs(obs - pred) <= np.maximum(5e-4, 0.3 * np.abs(obs)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = []
    for d in args.dirs:
        for line in (d / "mechanism.jsonl").open():
            r = json.loads(line)
            r["task"] = d.name
            recs.append(r)
    summary = dict(n_states=len(recs), phases={}, replay_err_max=max(r["replay_err"] for r in recs),
                   grad_path_err_max=max(r["grad_path_err"] for r in recs))
    for r in recs:
        summary["phases"][r["phase"]] = summary["phases"].get(r["phase"], 0) + 1

    def arm_groups(name):
        if name.startswith("single:"):
            return "single@" + name.split("@")[1]
        return name

    rows = []
    for r in recs:
        for name, a in r["arms"].items():
            for w in ("end", "min"):
                rows.append(dict(task=r["task"], phase=r["phase"], arm=arm_groups(name), w=w, g_ref=r["g_ref"][w],
                                 obs=a[f"dg_{w}"], sim=a[f"dg_{w}_sim"], chain=a[f"dg_{w}_pred"],
                                 cos=a["da_cos"], ratio=a["da_pred_norm"] / max(a["da_norm"], 1e-12),
                                 da=a["da_norm"], grip=a["grip_sign_flip"],
                                 flip=a["events"]["any_flip"], change=a["events"]["any_change"]))
    out = {}
    for arm in sorted({x["arm"] for x in rows}):
        out[arm] = {}
        for w in ("end", "min"):
            for ph in ["all", "near"] + sorted(summary["phases"]):
                sel = [x for x in rows if x["arm"] == arm and x["w"] == w and x["g_ref"] < CLIP and
                       (ph == "all" or x["phase"] == ph or (ph == "near" and abs(x["g_ref"]) < NEAR))]
                if len(sel) < 3:
                    continue
                nogrip = [x for x in sel if not x["grip"]]
                out[arm][f"{w}/{ph}"] = dict(
                    sim=metrics([x["obs"] for x in sel], [x["sim"] for x in sel]),
                    chain=metrics([x["obs"] for x in sel], [x["chain"] for x in sel]),
                    chain_no_grip_flip=metrics([x["obs"] for x in nogrip], [x["chain"] for x in nogrip]),
                    model_cos_median=float(np.median([x["cos"] for x in sel])),
                    model_norm_ratio_median=float(np.median([x["ratio"] for x in sel])),
                    grip_sign_flip_rate=float(np.mean([x["grip"] for x in sel])),
                    da_median=float(np.median([x["da"] for x in sel])))
                # sign preservation (contact-crossing) test, non-ambiguous margins only
                amb = [x for x in sel if abs(x["g_ref"]) < MARGIN]
                clear = [x for x in sel if abs(x["g_ref"]) >= MARGIN]
                if clear:
                    pred_keep = np.array([np.sign(x["g_ref"]) * (x["g_ref"] + x["chain"]) > 0 for x in clear])
                    obs_keep = np.array([np.sign(x["g_ref"] + x["obs"]) == np.sign(x["g_ref"]) for x in clear])
                    out[arm][f"{w}/{ph}"]["sign"] = dict(
                        n_clear=len(clear), n_ambiguous=len(amb), observed_crossings=int((~obs_keep).sum()),
                        predicted_crossings=int((~pred_keep).sum()),
                        agree=float(np.mean(pred_keep == obs_keep)),
                        ambiguous_observed_crossings=int(sum(np.sign(x["g_ref"] + x["obs"]) != np.sign(x["g_ref"])
                                                             for x in amb)))
    summary["by_arm"] = out
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mechanism_summary.json").write_text(json.dumps(summary, indent=1))

    # Scatter: observed vs chain (and vs sim) for a4_all, g_end, colored by phase
    colors = {"free": "#888", "pre_contact": "#1f6f8b", "transition": "#b03a2e", "holding": "#6c8e3b",
              "contact": "#7d3c98"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (arm, pred) in zip(axes, (("a4_all", "chain"), ("a4_all", "sim"), ("single@9", "chain"))):
        sel = [x for x in rows if x["arm"] == arm and x["w"] == "end" and x["g_ref"] < CLIP]
        for ph, c in colors.items():
            s = [x for x in sel if x["phase"] == ph]
            if s:
                ax.scatter([1e3 * x[pred] for x in s], [1e3 * x["obs"] for x in s], s=9, c=c, label=ph, alpha=0.7)
        lim = max([abs(1e3 * x["obs"]) for x in sel] + [abs(1e3 * x[pred]) for x in sel] + [1e-3])
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel(f"predicted Δg_end ({pred}) [mm]")
        ax.set_ylabel("observed Δg_end [mm]")
        ax.set_title({"chain": "M(ĥ−h)", "sim": "∇g·Δa_actual"}[pred] + f"  [{arm}]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "predicted_vs_observed.png", dpi=150)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
