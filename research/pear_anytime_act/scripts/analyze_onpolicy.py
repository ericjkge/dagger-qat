"""E1 on-policy: convergence of provisional actions logged by the `full` arm (probes never affect the trajectory).
For each round: probes (tokens so far, greedy provisional action) and the executed action (sampled after full thinking).
usage: python3 analyze_onpolicy.py results/arm_full"""
import json, math, sys
from pathlib import Path

ARM = ("x", "y", "z", "roll", "pitch", "yaw")
TOLS = {"exact": (0, 0, 0), "tol1": (1.0, 0.15, 0.1), "tol2": (2.0, 0.25, 0.15)}


def vec(a):
    try:
        return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception:
        return None


def gcls(g):
    return 0 if abs(g) < 1.0 else (1 if g > 0 else -1)


def close(va, vb, ta, tr, trot):
    if va is None or vb is None:
        return False
    if ta == 0:
        return va == vb
    dt = math.dist(va[:3], vb[:3]); mag = max(math.hypot(*va[:3]), math.hypot(*vb[:3]))
    return dt <= max(ta, tr * mag) and max(abs(va[i] - vb[i]) for i in (3, 4, 5)) <= trot and gcls(va[6]) == gcls(vb[6])


def main(d):
    rounds = []
    for p in sorted(Path(d).glob("libero_*.jsonl")):
        for l in p.open():
            r = json.loads(l)
            if r.get("action") and r.get("probes") is not None:
                rounds.append(r)
    T = sum(r["completion_tokens"] for r in rounds)
    print(f"{len(rounds)} rounds, {T} tokens, rounds with >=2 probes: {sum(len(r['probes'])>=2 for r in rounds)}")
    print("\n== oracle: tokens after the provisional action stays within tol of the executed action")
    for name, t in TOLS.items():
        sv = 0
        for r in rounds:
            fv = vec(r["action"]); n = r["completion_tokens"]; conv = n
            for pr in reversed(r["probes"]):
                if close(vec(pr["greedy"]), fv, *t):
                    conv = pr["tok"]
                else:
                    break
            sv += n - conv
        print(f"  {name:6s} {sv/T:.3f}")
    print("\n== exit-rule replay (two consecutive probes agree, tokens >= m): saved tokens, agreement of exit action with executed")
    for name, t in TOLS.items():
        for m in (0, 400, 800):
            sv = 0; ok1 = ok2 = n_exit = 0
            for r in rounds:
                ps = r["probes"]; fv = vec(r["action"]); stop = None
                for i in range(1, len(ps)):
                    if ps[i]["tok"] >= m and close(vec(ps[i - 1]["greedy"]), vec(ps[i]["greedy"]), *t):
                        stop = i; break
                if stop is None:
                    ok1 += 1; ok2 += 1; continue
                n_exit += 1; sv += r["completion_tokens"] - ps[stop]["tok"]
                ex = vec(ps[stop]["greedy"])
                ok1 += close(ex, fv, *TOLS["tol1"]); ok2 += close(ex, fv, *TOLS["tol2"])
            print(f"  {name:6s} m={m:4d} exits {n_exit/len(rounds):.2f} saved {sv/T:.3f}  exit~exec@tol1 {ok1/len(rounds):.3f} @tol2 {ok2/len(rounds):.3f}")
    print("\n== consecutive-probe jitter (cm) in the last 3 probes of rounds with >=4 probes")
    js = []
    for r in rounds:
        ps = [vec(p["greedy"]) for p in r["probes"][-3:]]
        if len(r["probes"]) >= 4 and all(ps):
            js += [math.dist(ps[i][:3], ps[i + 1][:3]) for i in range(len(ps) - 1)]
    js.sort()
    if js:
        print(f"  n={len(js)} exact-equal {sum(j==0 for j in js)/len(js):.2f}  median {js[len(js)//2]:.2f}  p75 {js[int(.75*len(js))]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
