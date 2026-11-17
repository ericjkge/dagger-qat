"""Pilot analysis. Phase 1: teacher-forced dNLL vs BF16 (= KL(A||arm) estimate on A's own samples), split thinking /
answer(action) tokens, with episode-bootstrap CIs and gap recovery (B-C)/(B-A). Phase 2: closed-loop success,
thinking tokens per round, context exhaustion, paired by episode."""
import json, glob, sys, random, statistics as st
from pathlib import Path
R = Path(sys.argv[1]); random.seed(0)
tf = {}
for f in sorted(R.glob('tf_*.jsonl')):
    arm = f.stem[3:]
    tf[arm] = {r['episode']: r for r in map(json.loads, open(f))}
if 'A' in tf:
    eps = sorted(set.intersection(*[set(v) for v in tf.values()]))
    print(f'== Phase 1: teacher-forced, {len(eps)} episodes, arms {sorted(tf)}')
    def per_ep(arm, cls):  # summed dNLL and token count per episode
        out = []
        for e in eps:
            a, b = tf['A'][e]['tokens'], tf[arm][e]['tokens']
            assert len(a) == len(b), (arm, e, len(a), len(b))
            d = [(-y[1]) - (-x[1]) for x, y in zip(a, b) if x[0] in cls]
            ag = [y[2] for x, y in zip(a, b) if x[0] in cls]
            out.append((sum(d), len(d), sum(ag)))
        return out
    def boot(v):
        m = lambda s: sum(x[0] for x in s) / max(1, sum(x[1] for x in s))
        bs = sorted(m([random.choice(v) for _ in v]) for _ in range(2000))
        return m(v), bs[50], bs[1949]
    res = {}
    for cls, nm in [('t', 'thinking'), ('a', 'answer/action'), ('ta', 'all')]:
        for arm in sorted(tf):
            v = per_ep(arm, cls); mean, lo, hi = boot(v)
            agree = sum(x[2] for x in v) / max(1, sum(x[1] for x in v))
            res[(arm, cls)] = mean
            print(f'  {nm:14s} arm {arm}: dNLL/token {mean:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]  top1 {agree:.4f}  n={sum(x[1] for x in v)}')
        if ('B', cls) in res and ('C', cls) in res and res[('B', cls)] > 0:
            print(f'  {nm:14s} gap recovered by exact Delta (B-C)/(B-A): {(res[("B", cls)] - res[("C", cls)]) / res[("B", cls)]:.2f}')
cl = {}
for d in sorted(R.glob('cl_*')):
    rows = {}
    for f in d.glob('libero_*.jsonl'):
        rs = [json.loads(l) for l in open(f)]
        if not rs: continue
        last = rs[-1]
        th = [r['requests'][0]['meta']['completion_tokens'] for r in rs if r.get('requests')]
        rows[f.stem] = dict(success=bool(last.get('success')), end=last.get('end_reason'), rounds=len(rs),
                            think=sum(th), think_per_round=sum(th) / max(1, len(th)))
    cl[d.name[3:]] = rows
if cl:
    eps = sorted(set.intersection(*[set(v) for v in cl.values()]))
    print(f'== Phase 2: closed loop, {len(eps)} paired episodes')
    for arm, rows in sorted(cl.items()):
        s = [rows[e] for e in eps]
        print(f'  arm {arm}: success {sum(x["success"] for x in s)}/{len(s)}  budget/context end {sum(x["end"] in ("budget", "context") for x in s)}'
              f'  max_rounds {sum(x["end"] == "max_rounds" for x in s)}  median think/round {st.median(x["think_per_round"] for x in s):.0f}'
              f'  mean think/episode {st.mean(x["think"] for x in s):.0f}  mean rounds {st.mean(x["rounds"] for x in s):.1f}')
