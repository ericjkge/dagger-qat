"""Delta-regularized GPTQ (DIRECTION.md §9): dNLL vs BF16 (A) per arm, and the pre-registered paired comparison of the
alpha>0 arm against plain GPTQ (G0), by episode bootstrap. Optional pilot dir adds the released checkpoint (B) and
RTN (E) scored on the earlier pod, if its A matches this run's A.
usage: python3 analyze_gptq.py results/q [results/pilot]"""
import json, random, sys
from pathlib import Path


def load(d):
    return {f.stem[3:]: {r["episode"]: r for r in map(json.loads, open(f))} for f in sorted(Path(d).glob("tf_*.jsonl"))}


def per_ep(tf, arm, eps, cls):
    out = {}
    for e in eps:
        a, b = tf["A"][e]["tokens"], tf[arm][e]["tokens"]
        assert len(a) == len(b), (arm, e)
        d = [(-y[1]) - (-x[1]) for x, y in zip(a, b) if x[0] in cls]
        out[e] = (sum(d), len(d))
    return out


def main(q, pilot=None):
    tf = load(q)
    if pilot:
        P = load(pilot)
        same = all(abs(sum(x[1] for x in P["A"][e]["tokens"]) - sum(x[1] for x in tf["A"][e]["tokens"])) < 1e-3 * max(1, len(tf["A"][e]["tokens"])) for e in tf["A"])
        print(f"pilot A matches this run's A: {same}")
        if same:
            for k in ("B", "E"):
                if k in P:
                    tf[k] = P[k]
    eps = sorted(set.intersection(*[set(v) for v in tf.values()]))
    rnd = random.Random(0)
    m = lambda v: sum(x[0] for x in v) / max(1, sum(x[1] for x in v))
    print(f"{len(eps)} episodes; arms {sorted(tf)}")
    for cls, nm in (("a", "action"), ("t", "thinking")):
        print(f"\n{nm} dNLL vs BF16 (nats/token, episode-bootstrap 95% CI)")
        PE = {arm: per_ep(tf, arm, eps, cls) for arm in tf if arm != "A"}
        for arm, v in PE.items():
            vals = list(v.values()); bs = sorted(m([rnd.choice(vals) for _ in vals]) for _ in range(2000))
            print(f"  {arm:4s} {m(vals):+.5f} [{bs[50]:+.5f}, {bs[1949]:+.5f}]")
        for arm in [x for x in PE if x.startswith("G") and x != "G0"]:
            diffs = [(PE[arm][e][0] - PE["G0"][e][0], PE[arm][e][1]) for e in eps]
            bs = sorted(m([rnd.choice(diffs) for _ in diffs]) for _ in range(2000))
            print(f"  {arm} - G0: {m(diffs):+.5f} [{bs[50]:+.5f}, {bs[1949]:+.5f}]  (negative = {arm} better)")


if __name__ == "__main__":
    main(*sys.argv[1:])
