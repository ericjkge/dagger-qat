"""E3 analysis: k full-thinking samples per recorded round.
- action spread: translation distance between samples, relative to move size
- does chain length predict error vs the round's consensus (geometric median of the other samples)?
- order statistics of thinking length: mean single-sample tokens vs min / 2nd-smallest of k (short-m@k latency proxy)
usage: python3 analyze_samples.py sample_k.jsonl"""
import json, math, sys, statistics as st

ARM = ("x", "y", "z", "roll", "pitch", "yaw")


def vec(a):
    try:
        return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception:
        return None


def gmed(P, it=200):
    y = [sum(p[i] for p in P) / len(P) for i in range(3)]
    for _ in range(it):
        w = [1 / max(1e-6, math.dist(p, y)) for p in P]
        y = [sum(wi * p[i] for wi, p in zip(w, P)) / sum(w) for i in range(3)]
    return y


def main(path):
    R = [json.loads(l) for l in open(path)]
    pair, rel, lens, devs = [], [], [], []
    order = {1: [], 2: [], "mean": []}
    gflip = 0; gn = 0
    for r in R:
        V = [(s["tok"], vec(s["action"])) for s in r["samples"]]
        V = [(t, v) for t, v in V if v]
        if len(V) < 3:
            continue
        ts = sorted(t for t, _ in V)
        order[1].append(ts[0]); order[2].append(ts[1]); order["mean"].append(sum(ts) / len(ts))
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                d = math.dist(V[i][1][:3], V[j][1][:3]); m = max(1.0, math.hypot(*V[i][1][:3]), math.hypot(*V[j][1][:3]))
                pair.append(d); rel.append(d / m)
                gn += 1; gflip += (abs(V[i][1][6]) >= 1) != (abs(V[j][1][6]) >= 1)
        for i, (t, v) in enumerate(V):
            others = [u[:3] for k, (_, u) in enumerate(V) if k != i]
            c = gmed(others)
            lens.append(t); devs.append(math.dist(v[:3], c))
    pair.sort(); rel.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    print(f"{len(R)} rounds")
    print(f"pairwise translation distance (cm): median {q(pair,.5):.2f} p75 {q(pair,.75):.2f} p90 {q(pair,.9):.2f}; "
          f"relative to move: median {q(rel,.5):.2f} p75 {q(rel,.75):.2f}; gripper-command disagreement {gflip/max(1,gn):.2f}")
    # length vs deviation from consensus of others
    n = len(lens); ml, md = sum(lens) / n, sum(devs) / n
    cov = sum((a - ml) * (b - md) for a, b in zip(lens, devs)) / n
    r = cov / (st.pstdev(lens) * st.pstdev(devs) + 1e-9)
    rk = lambda xs: {i: k for k, i in enumerate(sorted(range(len(xs)), key=lambda j: xs[j]))}
    rl, rd = rk(lens), rk(devs)
    rs = 1 - 6 * sum((rl[i] - rd[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    print(f"chain length vs deviation from others' consensus: pearson {r:.2f}, spearman {rs:.2f} (n={n})")
    for lo, hi in ((0, 500), (500, 1500), (1500, 4000), (4000, 10**9)):
        sel = [d for t, d in zip(lens, devs) if lo <= t < hi]
        if sel:
            print(f"   len [{lo},{hi}) n={len(sel):3d} mean dev {sum(sel)/len(sel):.2f} cm")
    print(f"thinking tokens per round: single sample mean {sum(order['mean'])/len(order['mean']):.0f}, "
          f"min of k {sum(order[1])/len(order[1]):.0f}, 2nd-smallest {sum(order[2])/len(order[2]):.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
