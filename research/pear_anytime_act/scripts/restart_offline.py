"""Exact expected decode tokens per round under restart schedules, using each context's empirical length distribution
(4 fresh samples + the recorded one). E = sum_i prod_{j<i} P(T>L_j) * E[min(T, L_i)]."""
import json, sys


def luby(i):
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    return (1 << (k - 1)) if i == (1 << k) - 1 else luby(i - (1 << (k - 1)) + 1)


def cost(d, sched, imax=200):
    n = len(d); surv = 1.0; e = 0.0
    for i in range(1, imax):
        L = sched(i)
        e += surv * sum(min(t, L) for t in d) / n
        surv *= sum(t > L for t in d) / n
        if surv < 1e-9:
            break
    return e


R = [json.loads(l) for l in open(sys.argv[1])]
D = [[s["tok"] for s in r["samples"]] + [r["rec_tok"]] for r in R]
base = sum(sum(d) / len(d) for d in D)
print(len(D), "contexts; mean tokens/round", round(base / len(D)))
r = sorted(max(d) / min(d) for d in D)
print("per-context max/min length: median %.2f p75 %.2f p90 %.2f" % (r[len(r) // 2], r[int(.75 * len(r))], r[int(.9 * len(r))]))
for u in (250, 500, 1000, 2000, 4000):
    print(f"Luby unit {u:5d}: tokens ratio {sum(cost(d, lambda i: u * luby(i)) for d in D) / base:.3f}")
for L in (1500, 3000, 5000, 8000):
    print(f"fixed cutoff {L:5d}: tokens ratio {sum(cost(d, lambda i: L) for d in D) / base:.3f}")
opt = sum(min(cost(d, lambda i, L=L: L) for L in set(d)) for d in D)
print(f"oracle per-context fixed cutoff: {opt / base:.3f}")
# relative cutoff: restart when a chain exceeds c x the context's own median (unknowable online; upper bound on shape)
for c in (1.25, 1.5, 2.0):
    tot = 0
    for d in D:
        m = sorted(d)[len(d) // 2]
        tot += cost(d, lambda i: c * m)
    print(f"cutoff {c} x context median (oracle scale): {tot / base:.3f}")
