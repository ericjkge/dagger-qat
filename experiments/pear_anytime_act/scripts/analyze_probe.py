"""E1 analysis: convergence of PEAR's provisional action during thinking (offline, recorded rounds).

For each round: provisional greedy actions at cuts c_1..c_K (K = full thinking).
Reports
  - oracle convergence point: first cut after which every provisional action stays within tolerance of the final one
  - exit rules replayed offline (consecutive-probe agreement): tokens saved and error of the exit action vs the final
    (full-thinking) greedy action, for exact agreement (Dynasor/DEER-style) and feedback tolerance (ours)
  - natural spread: disagreement between two samples / greedy at the full thinking (what re-sampling alone gives)
usage: python3 analyze_probe.py probe.jsonl
"""
import json, math, sys, random
from collections import defaultdict

ARM = ("x", "y", "z", "roll", "pitch", "yaw")


def vec(a):
    try:
        return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception:
        return None


def gcls(g):
    return 0 if abs(g) < 1.0 else (1 if g > 0 else -1)


def close(va, vb, tol_abs, tol_rel, tol_rot):
    if va is None or vb is None:
        return False
    if tol_abs == 0:
        return va == vb
    dt = math.dist(va[:3], vb[:3]); mag = max(math.hypot(*va[:3]), math.hypot(*vb[:3]))
    return dt <= max(tol_abs, tol_rel * mag) and max(abs(va[i] - vb[i]) for i in (3, 4, 5)) <= tol_rot and gcls(va[6]) == gcls(vb[6])


def err(va, vb):
    if va is None or vb is None:
        return None
    return math.dist(va[:3], vb[:3]), max(abs(va[i] - vb[i]) for i in (3, 4, 5)), int(gcls(va[6]) != gcls(vb[6]))


def main(path):
    R = defaultdict(list)
    for l in open(path):
        x = json.loads(l); R[(x["ep"], x["round"])].append(x)
    rounds = []
    for k, xs in R.items():
        xs.sort(key=lambda x: x["cut"])
        if not xs[-1]["final"]:
            continue
        rounds.append((k, xs))
    T = sum(xs[-1]["think_tok"] for _, xs in rounds)
    print(f"{len(rounds)} rounds, {T} thinking tokens")
    fin_ok = [vec(xs[-1]["greedy"]) for _, xs in rounds]
    # consistency of the probe at full thinking with the recorded action (sampled) -> natural spread
    print("\n== natural spread at full thinking (same context): disagreement rate under tolerance")
    TOLS = {"exact": (0, 0, 0), "tol1": (1.0, 0.15, 0.1), "tol2": (2.0, 0.25, 0.15)}
    for name, t in TOLS.items():
        a = sum(close(vec(xs[-1]["greedy"]), vec(xs[-1]["recorded"]), *t) for _, xs in rounds) / len(rounds)
        b = [close(vec(xs[-1]["samples"][0]), vec(xs[-1]["samples"][1]), *t) for _, xs in rounds if len(xs[-1]["samples"]) > 1]
        print(f"  {name:6s} greedy==recorded {a:.2f}   sample==sample {sum(b)/max(1,len(b)):.2f}")
    print("\n== oracle convergence (first cut after which greedy stays within tol of final greedy)")
    for name, t in TOLS.items():
        saved = 0; fr = []
        for _, xs in rounds:
            fv = vec(xs[-1]["greedy"]); n = xs[-1]["think_tok"]
            conv = n
            for i in range(len(xs) - 1, -1, -1):
                if close(vec(xs[i]["greedy"]), fv, *t):
                    conv = xs[i]["tok"]
                else:
                    break
            saved += n - conv; fr.append(conv / max(1, n))
        fr.sort()
        print(f"  {name:6s} tokens after convergence {saved/T:.3f}   median conv fraction {fr[len(fr)//2]:.2f}")
    print("\n== exit rule replay: stop at first cut where two consecutive greedy probes agree (min tokens m)")
    print("   rule   m    tok_saved  err_cm(mean/p90)  rot>0.1  grip_flip  wrong@tol1  wrong@tol2")
    for name, t in TOLS.items():
        for m in (0, 300, 600):
            saved = 0; E = []
            for _, xs in rounds:
                n = xs[-1]["think_tok"]; fv = vec(xs[-1]["greedy"]); stop = None
                for i in range(1, len(xs) - 1):
                    if xs[i]["tok"] >= m and close(vec(xs[i - 1]["greedy"]), vec(xs[i]["greedy"]), *t):
                        stop = i; break
                if stop is None:
                    E.append((0, 0, 0, True, True)); continue
                saved += n - xs[stop]["tok"]
                ex = vec(xs[stop]["greedy"]); e = err(ex, fv)
                if e is None:
                    E.append((99, 9, 1, False, False)); continue
                E.append((*e, close(ex, fv, *TOLS["tol1"]), close(ex, fv, *TOLS["tol2"])))
            cm = sorted(e[0] for e in E)
            print(f"  {name:6s} {m:4d}   {saved/T:.3f}     {sum(cm)/len(cm):5.2f}/{cm[int(.9*len(cm))]:5.2f}"
                  f"     {sum(e[1]>0.1 for e in E)/len(E):.3f}   {sum(e[2] for e in E)/len(E):.3f}"
                  f"     {1-sum(e[3] for e in E)/len(E):.3f}      {1-sum(e[4] for e in E)/len(E):.3f}")
    # by round size
    print("\n== tokens after tol1-convergence by round size")
    for lo, hi in ((0, 600), (600, 2000), (2000, 6000), (6000, 10**9)):
        sel = [(k, xs) for k, xs in rounds if lo <= xs[-1]["think_tok"] < hi]
        if not sel:
            continue
        tt = sum(xs[-1]["think_tok"] for _, xs in sel); sv = 0
        for _, xs in sel:
            fv = vec(xs[-1]["greedy"]); conv = xs[-1]["think_tok"]
            for i in range(len(xs) - 1, -1, -1):
                if close(vec(xs[i]["greedy"]), fv, *TOLS["tol1"]):
                    conv = xs[i]["tok"]
                else:
                    break
            sv += xs[-1]["think_tok"] - conv
        print(f"  [{lo},{hi}) rounds {len(sel):3d} tok share {tt/T:.2f} saved-after-conv {sv/tt:.2f}")
    print("\nunparsable greedy probes:", sum(vec(x["greedy"]) is None for _, xs in rounds for x in xs),
          "of", sum(len(xs) for _, xs in rounds))


if __name__ == "__main__":
    main(sys.argv[1])
