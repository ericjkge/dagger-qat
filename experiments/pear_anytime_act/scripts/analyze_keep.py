"""Grounded-claim retention: pre-registered criteria (DIRECTION.md §10) on score_keep.py output.
Per round: action-token NLL (answer tokens 'a') and thinking NLL ('t'), top-1 agreement, prompt tokens.
Gap recovery R = (strip - grounded) / (strip - full) on pooled action NLL; bootstrap over rounds."""
import json, random, sys

CONDS = ("full", "strip", "grounded", "random", "last")


def nll(toks, kind):
    v = [-x[1] for x in toks if x[0] == kind]
    return sum(v), len(v)


def main(path):
    R = [json.loads(l) for l in open(path)]
    print(f"{len(R)} rounds from {len({r['episode'] for r in R})} episodes")
    rows = []
    for r in R:
        d = {}
        for c in CONDS:
            sa, na = nll(r[c]["tokens"], "a"); st, nt = nll(r[c]["tokens"], "t")
            agree = [x[2] for x in r[c]["tokens"] if x[0] == "a"]
            d[c] = dict(a=sa, na=na, t=st, nt=nt, agr=sum(agree), pt=r[c]["prompt_tokens"])
        rows.append(d)
    def pooled(rs, c, k="a"):
        n = sum(x[c]["n" + k] for x in rs); return sum(x[c][k] for x in rs) / max(1, n)
    print(f"{'cond':9s} {'action NLL':>10s} {'think NLL':>10s} {'action top1':>11s} {'prompt tok':>10s}")
    for c in CONDS:
        print(f"{c:9s} {pooled(rows, c):10.4f} {pooled(rows, c, 't'):10.4f} "
              f"{sum(x[c]['agr'] for x in rows)/sum(x[c]['na'] for x in rows):11.4f} {sum(x[c]['pt'] for x in rows)/len(rows):10.0f}")
    def rec(rs, c):
        g = pooled(rs, "strip") - pooled(rs, "full")
        return (pooled(rs, "strip") - pooled(rs, c)) / g if g > 0 else float("nan")
    def diff(rs, a, b):
        return pooled(rs, a) - pooled(rs, b)
    rnd = random.Random(0); B = 2000
    bs = [[rnd.choice(rows) for _ in rows] for _ in range(B)]
    def ci(f):
        v = sorted(f(s) for s in bs); return v[int(.025 * B)], v[int(.975 * B)]
    gap = diff(rows, "strip", "full")
    print(f"\nstrip - full action NLL gap: {gap:+.4f} {ci(lambda s: diff(s, 'strip', 'full'))}")
    for c in ("grounded", "random", "last"):
        lo, hi = ci(lambda s: rec(s, c))
        print(f"gap recovery {c:9s}: {rec(rows, c):.2f} [{lo:.2f}, {hi:.2f}]")
    lo, hi = ci(lambda s: diff(s, "random", "grounded"))
    print(f"random - grounded action NLL: {diff(rows, 'random', 'grounded'):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    lo, hi = ci(lambda s: diff(s, "last", "grounded"))
    print(f"last - grounded action NLL:   {diff(rows, 'last', 'grounded'):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    r_lo = ci(lambda s: rec(s, "grounded"))[0]; d_lo = ci(lambda s: diff(s, "random", "grounded"))[0]
    go = rec(rows, "grounded") >= 0.6 and r_lo > 0.3 and d_lo > 0
    print(f"\nPRE-REGISTERED: recovery>=0.6 & lower>0.3: {rec(rows,'grounded')>=0.6 and r_lo>0.3}; "
          f"beats random (CI>0): {d_lo>0}  ->  {'GO' if go else 'NO-GO'}")


if __name__ == "__main__":
    main(sys.argv[1])
