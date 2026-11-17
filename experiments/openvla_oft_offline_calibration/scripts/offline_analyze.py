"""Analyze the offline arms: quantization-induced action error, paired by simulator state.

e(arm, frame) = mean |a_arm - a_ref| over the 8x7 normalized action chunk (OFT's [-1, 1] action
space), i.e. the action change caused by quantizing, on the identical input. All intervals are 95%
percentile bootstraps that resample *states* (frames of one state stay together), 10k draws.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

PERTS = ("camera", "light", "background", "noise", "language")
CONDS = ("clean",) + PERTS
QUANT = ("rtn", "gptq_clean", "gptq_pert", "gptq_oracle")
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
B = 10_000


def boot(values_by_state: dict, stat, rng) -> tuple[float, float, float]:
    """values_by_state: state -> array-like row; stat(matrix) -> scalar. Cluster bootstrap over states."""
    keys = list(values_by_state)
    mat = np.array([values_by_state[k] for k in keys], dtype=float)
    point = stat(mat)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    draws = np.array([stat(mat[i]) for i in idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fmt(t, pct=False):
    p, lo, hi = t
    if pct:
        return f"{100 * p:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/offline/run"))
    parser.add_argument("--out", type=Path, default=Path("results/offline"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest = json.loads((args.run / "manifest.json").read_text())
    frames = {f["id"]: f for f in manifest["frames"]}
    arms_info = json.loads((args.run / "arms.json").read_text())
    oracle_states = set(arms_info["oracle_states"])
    z = np.load(args.run / "actions.npz")
    metrics = pickle.loads((args.run / "metrics.pkl").read_bytes())
    st = np.load(args.run / "states.npz")
    demo = {k: a for k, a in zip(st["keys"], st["demo_actions"])}

    norm = {}
    act = {}
    for arm in ("ref", "bf16") + QUANT:
        for fid, n, a in zip(z[f"{arm}_ids"], z[f"{arm}_norm"], z[f"{arm}_act"]):
            norm[(arm, int(fid))] = n
            act[(arm, int(fid))] = a
    eval_frames = [f for f in frames.values() if f["role"] == "eval"]
    by_sc = {(f["state"], f["condition"]): f["id"] for f in eval_frames}
    states = sorted({f["state"] for f in eval_frames})
    held = [s for s in states if s not in oracle_states]

    def err(arm, fid):
        return float(np.abs(norm[(arm, fid)] - norm[("ref", fid)]).mean())

    E = {arm: {s: [err(arm, by_sc[(s, c)]) for c in CONDS] for s in states} for arm in QUANT}
    report = {}
    lines = ["# Offline results (auto-generated)", ""]

    # ---- sanity: reference policy vs recorded demo actions; numerical floor ---------------------
    first = lambda arm, fid: act[(arm, fid)][0]  # noqa: E731
    clean_ids = [by_sc[(s, "clean")] for s in states]
    ref_vs_demo = np.mean([np.abs(first("ref", i)[:6] - demo[frames[i]["state"]][0][:6]).mean() for i in clean_ids])
    grip_agree = np.mean([np.sign(first("ref", i)[6]) == np.sign(demo[frames[i]["state"]][0][6]) for i in clean_ids])
    floor_ids = [int(i) for i in z["bf16_ids"]]
    floor = np.mean([err("bf16", i) for i in floor_ids])
    rtn_on_floor = np.mean([err("rtn", i) for i in floor_ids])
    report["sanity"] = dict(ref_vs_demo_l1_first_action_arm=float(ref_vs_demo), gripper_sign_agreement=float(grip_agree),
                            bf16_vs_fp16_err=float(floor), rtn_err_same_frames=float(rtn_on_floor))
    lines += ["## Sanity", f"- FP16 reference vs recorded demo, first action, arm dims mean |diff| (raw units): "
              f"{ref_vs_demo:.4f}; gripper sign agreement {grip_agree:.0%}",
              f"- Numerical floor (BF16 vs FP16 weights): {floor:.4f}; RTN W4 on the same frames: {rtn_on_floor:.4f} "
              f"({rtn_on_floor / floor:.1f}x floor)", ""]

    # FP sensitivity to each perturbation (scale for everything else)
    sens = {c: np.mean([np.abs(norm[("ref", by_sc[(s, c)])] - norm[("ref", by_sc[(s, 'clean')])]).mean()
                        for s in states]) for c in PERTS}
    report["fp_sensitivity"] = sens

    # ---- C1: is quantization error larger under perturbation? -------------------------------------
    lines += ["## C1: quantization error under perturbation vs the clean twin (same state)",
              "| arm | clean e | " + " | ".join(f"{c} (ratio to clean)" for c in PERTS) + " | all perturbed |",
              "|---|---|" + "---|" * (len(PERTS) + 1)]
    report["C1"] = {}
    for arm in QUANT:
        row = [arm, f"{np.mean([E[arm][s][0] for s in states]):.4f}"]
        rc = {}
        for j, c in enumerate(PERTS, 1):
            t = boot(E[arm], lambda m, j=j: m[:, j].mean() / m[:, 0].mean() - 1, rng)
            rc[c] = t
            row.append(fmt(t, True))
        t = boot(E[arm], lambda m: m[:, 1:].mean() / m[:, 0].mean() - 1, rng)
        rc["all"] = t
        row.append(fmt(t, True))
        report["C1"][arm] = rc
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- C2/C3: does the calibration data matter? -------------------------------------------------
    def contrast(a, b, cols, subset):
        data = {s: [E[a][s][j] for j in cols] + [E[b][s][j] for j in cols] for s in subset}
        k = len(cols)
        return boot(data, lambda m: m[:, k:].mean() / m[:, :k].mean() - 1, rng)  # b relative to a

    pert_cols = list(range(1, 6))
    lines += ["## C2/C3: calibration data (relative change in error vs gptq_clean; negative = better)",
              "| arm | on perturbed frames | on clean frames | states |", "|---|---|---|---|"]
    report["C2C3"] = {}
    for arm, subset, label in (("gptq_pert", states, "all 40"), ("gptq_oracle", held, f"{len(held)} held-out"),
                               ("gptq_pert", held, f"{len(held)} held-out")):
        tp = contrast("gptq_clean", arm, pert_cols, subset)
        tc = contrast("gptq_clean", arm, [0], subset)
        per = {c: contrast("gptq_clean", arm, [j], subset) for j, c in enumerate(PERTS, 1)}
        report["C2C3"][f"{arm}|{label}"] = dict(pert=tp, clean=tc, per_condition=per)
        lines.append(f"| {arm} | {fmt(tp, True)} | {fmt(tc, True)} | {label} |")
    lines.append("")
    lines += ["Per perturbation type (perturbed frames, relative to gptq_clean):",
              "| arm | " + " | ".join(PERTS) + " |", "|---|" + "---|" * len(PERTS)]
    for key, v in report["C2C3"].items():
        lines.append(f"| {key} | " + " | ".join(fmt(v["per_condition"][c], True) for c in PERTS) + " |")
    lines.append("")

    # ---- C4: GPTQ's advantage over RTN under shift -------------------------------------------------
    lines += ["## C4: GPTQ (clean calibration) error relative to RTN", "| condition | gptq_clean / rtn - 1 |",
              "|---|---|"]
    report["C4"] = {}
    for j, c in enumerate(CONDS):
        data = {s: [E["rtn"][s][j], E["gptq_clean"][s][j]] for s in states}
        t = boot(data, lambda m: m[:, 1].mean() / m[:, 0].mean() - 1, rng)
        report["C4"][c] = t
        lines.append(f"| {c} | {fmt(t, True)} |")
    data = {s: E["rtn"][s] + E["gptq_clean"][s] for s in states}
    t = boot(data, lambda m: (m[:, 7:12].mean() / m[:, 1:6].mean()) / (m[:, 6].mean() / m[:, 0].mean()) - 1, rng)
    report["C4"]["advantage_shift"] = t
    lines += [f"| change of the ratio, perturbed vs clean | {fmt(t, True)} |", ""]

    # ---- C5: where the error comes from -----------------------------------------------------------
    lines += ["## C5: per-layer error (gptq_clean), perturbed / clean twin, action tokens"]
    cond_of = {f["id"]: f["condition"] for f in eval_frames}
    state_of = {f["id"]: f["state"] for f in eval_frames}
    loc = {}
    for arm in QUANT:
        mat = np.zeros((32, 7, 2))
        for i in range(32):
            vals = metrics["linear"].get((arm, i))
            if vals is None:
                continue
            for k, name in enumerate(LINEARS):
                full = [n for n in vals if n.endswith(name)][0]
                d = vals[full]
                cl = [d[f][0] for f in d if cond_of.get(f) == "clean"]
                pe = [d[f][0] for f in d if cond_of.get(f) in PERTS]
                mat[i, k] = (np.mean(cl), np.mean(pe))
        loc[arm] = mat
    np.save(args.out / "layer_errors.npy", np.stack([loc[a] for a in QUANT]))
    m = loc["gptq_clean"]
    ratio = m[..., 1] / m[..., 0]
    top = np.dstack(np.unravel_index(np.argsort(-ratio.ravel())[:5], ratio.shape))[0]
    lines.append("- Largest perturbed/clean increases: " + ", ".join(
        f"L{i}.{LINEARS[k]} x{ratio[i, k]:.2f}" for i, k in top))
    contrib = (m[..., 1] - m[..., 0]).sum(1)
    lines.append("- Layers with largest absolute increase: " + ", ".join(
        f"L{i}" for i in np.argsort(-contrib)[:5]))
    gr = loc["gptq_clean"][..., 1].sum() / loc["gptq_clean"][..., 0].sum()
    rr = loc["rtn"][..., 1].sum() / loc["rtn"][..., 0].sum()
    lines.append(f"- Summed linear error, perturbed/clean: gptq_clean x{gr:.3f}, rtn x{rr:.3f}")
    drift = {}
    for arm in QUANT:
        drift[arm] = np.array([[np.mean([metrics["drift"][(arm, i)][f][0] for f in metrics["drift"][(arm, i)]
                                         if cond_of.get(f) == c]) for c in CONDS] for i in range(32)])
    np.save(args.out / "drift.npy", np.stack([drift[a] for a in QUANT]))
    lines.append("- Hidden-state drift vs reference (action tokens) at layers 7/15/23/31, clean -> mean perturbed:")
    for arm in QUANT:
        d = drift[arm]
        lines.append(f"  - {arm}: " + ", ".join(f"L{i} {d[i, 0]:.3f}->{d[i, 1:].mean():.3f}" for i in (7, 15, 23, 31)))
    report["C5"] = dict(top=[(int(i), LINEARS[k], float(ratio[i, k])) for i, k in top], gptq_sum_ratio=float(gr),
                        rtn_sum_ratio=float(rr))
    lines.append("")
    lines += ["## Scale", "FP16 policy's own action change under each perturbation (normalized L1 vs clean): "
              + ", ".join(f"{c} {v:.3f}" for c, v in sens.items())]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    (args.out / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
