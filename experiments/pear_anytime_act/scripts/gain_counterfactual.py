"""Offline counterfactual for online action-gain calibration on PEAR's lateral moves.
Every reached lateral move m (>3 cm, |dz|<3, no gripper) is a trial. Its 'intended' displacement is m plus the lateral
component of the follow-up if the next round is a lateral follow-up (no gripper, |dz|<3); otherwise m itself (the
model accepted the position and went on, e.g. descended). Policies scale m by k and we measure the residual
|intended - k m| (cm) and whether a follow-up round would still be needed (residual > tol)."""
import json, glob, math, statistics as st
ARM = ("x", "y", "z", "roll", "pitch", "yaw")
def vec(a):
    try: return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception: return None
files = [f for f in glob.glob("**/libero_*_r*.jsonl", recursive=True) if "/frames/" not in f]
seen = set(); eps = []
for f in sorted(files):
    R = [json.loads(l) for l in open(f)]
    key = f.split("/")[-1] + json.dumps(R[0].get("action"))
    if len(R) >= 3 and key not in seen: seen.add(key); eps.append(R)
trials = []  # per episode list of (m(2), intended(2), followed)
for R in eps:
    T = []
    for i in range(len(R) - 1):
        p, r = R[i], R[i+1]
        pv, v = vec(p.get("action") or {}), vec(r.get("action") or {})
        if not pv or abs(pv[6]) >= 1 or abs(pv[2]) >= 3 or math.hypot(pv[0], pv[1]) <= 3 or p.get("outcome") != "reached": continue
        fol = v is not None and abs(v[6]) < 1 and abs(v[2]) < 3 and math.hypot(v[0], v[1]) > 0.5
        it = [pv[0] + (v[0] if fol else 0), pv[1] + (v[1] if fol else 0)]
        T.append((pv[:2], it, fol))
    eps_T = T
    if T: trials.append(T)
n = sum(len(T) for T in trials); nf = sum(t[2] for T in trials for t in T)
print(f"{n} reached lateral moves in {len(trials)} episodes; followed by lateral follow-up: {nf/n:.0%}")
def evaluate(policy, tol=1.5):
    res = []; need = 0; over = 0
    for T in trials:
        hist = []
        for m, it, fol in T:
            k = policy(hist)
            e = math.dist([k*m[0], k*m[1]], it); res.append(e); need += e > tol
            ml = math.hypot(*m); along = (it[0]*m[0] + it[1]*m[1]) / ml
            over += k*ml > along + tol
            hist.append(along / ml)
    return st.mean(res), sorted(res)[len(res)//2], need / len(res), over / len(res)
pols = {"k=1 (as is)": lambda h: 1.0, "k=1.25 fixed": lambda h: 1.25, "k=1.5 fixed": lambda h: 1.5,
        "online median of own ratios (prior 1.0, from 2nd move)": lambda h: st.median(h) if h else 1.0,
        "online shrunk mean (prior 1.0, weight 2)": lambda h: (2*1.0 + sum(h)) / (2 + len(h)),
        "online shrunk, capped [1,1.6]": lambda h: min(1.6, max(1.0, (2*1.0 + sum(h)) / (2 + len(h))))}
print(f"{'policy':55s} mean err  med err  still needs follow-up  overshoot")
for name, pol in pols.items():
    m, md, need, over = evaluate(pol)
    print(f"{name:55s} {m:7.2f}  {md:7.2f}   {need:8.0%}              {over:6.0%}")
# oracle per-episode constant
best = 0; tot = 0
for T in trials:
    rat = [((it[0]*m[0]+it[1]*m[1])/math.hypot(*m))/math.hypot(*m) for m, it, _ in T]
    k = st.median(rat)
    best += sum(math.dist([k*m[0], k*m[1]], it) > 1.5 for m, it, _ in T); tot += len(T)
print(f"oracle per-episode median k: still needs follow-up {best/tot:.0%}")
# variance decomposition of ratio: within vs between episodes
allr = [((it[0]*m[0]+it[1]*m[1])/math.hypot(*m))/math.hypot(*m) for T in trials for m, it, _ in T]
print(f"ratio along/|m|: median {st.median(allr):.2f}, IQR [{sorted(allr)[len(allr)//4]:.2f},{sorted(allr)[3*len(allr)//4]:.2f}]")
