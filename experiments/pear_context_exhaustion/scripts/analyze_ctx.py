"""Compare compressed histories against full (S0) and stripped (S1): action and thinking dNLL vs full, fraction of the
S1 shift recovered, kept context, with episode-bootstrap CIs. Inputs: strip_A.jsonl, ctx_scores.jsonl."""
import json, random, sys
from pathlib import Path
P = Path(sys.argv[1]); random.seed(0)
strip = {(r["episode"], r["round"]): r for r in map(json.loads, (P / "strip_A.jsonl").open())}
ctx = {(r["episode"], r["round"]): r for r in map(json.loads, (P / "ctx_scores.jsonl").open()) if not r.get("skip")}
keys = sorted(k for k in ctx if k in strip)
modes = ["strip", "card", "geo", "summ", "geo+card"]
def get(k, m):
    return strip[k]["strip"] if m == "strip" else ctx[k][m]
_c = {}
def per(k, m, cls):
    if (k, m, cls) not in _c:
        F = strip[k]["full"]["tokens"]; X = get(k, m)["tokens"]
        assert len(F) == len(X), (k, m)
        d = n = ag = 0
        for f, x in zip(F, X):
            if f[0] == cls: d += f[1] - x[1]; n += 1; ag += x[2]
        _c[(k, m, cls)] = (d, n, ag)
    return _c[(k, m, cls)]
def agg(ks, m, cls):
    t = [per(k, m, cls) for k in ks]
    n = sum(x[1] for x in t)
    return sum(x[0] for x in t) / max(1, n), sum(x[2] for x in t) / max(1, n)
eps = sorted({k[0] for k in keys}); byep = {e: [k for k in keys if k[0] == e] for e in eps}
print(f"{len(keys)} rounds, {len(eps)} episodes")
for cls, nm in (("a", "ACTION"), ("t", "thinking")):
    base = agg(keys, "strip", cls)[0]
    print(f"== {nm} tokens: dNLL vs full history (S0); recovered = 1 - dNLL/dNLL(strip)")
    for m in modes:
        v, ag = agg(keys, m, cls)
        bs = []
        for _ in range(1000):
            ks = [k for e in (random.choice(eps) for _ in eps) for k in byep[e]]
            s_ = agg(ks, "strip", cls)[0]; bs.append(1 - agg(ks, m, cls)[0] / s_ if s_ > 0 else 0)
        bs.sort()
        rec = 1 - v / base if base > 0 else float("nan")
        print(f"  {m:9s} dNLL {v:+.4f}  top1 {ag:.4f}  recovered {rec:+.2f} [95% CI {bs[25]:+.2f}, {bs[974]:+.2f}]")
pt = lambda m: sum(get(k, m)["prompt_tokens"] for k in keys) / len(keys)
full = sum(strip[k]["full"]["prompt_tokens"] for k in keys) / len(keys)
print("== mean context tokens: full %.0f, " % full + ", ".join(f"{m} {pt(m):.0f}" for m in modes))
ex = next((r.get("card_example") for r in ctx.values() if r.get("card_example")), None)
print("== example card:", ex)
