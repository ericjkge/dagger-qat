"""Where do PEAR's rounds go? Classify rounds and test for systematic undershoot/overshoot in follow-up corrections."""
import json, glob, math, collections, sys
ARM = ("x", "y", "z", "roll", "pitch", "yaw")
def vec(a):
    try: return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception: return None
files = [f for f in glob.glob("**/libero_*_r*.jsonl", recursive=True) if "/frames/" not in f]
seen = set(); eps = []
for f in sorted(files):
    key = f.split("/")[-1] + str(open(f).readline()[:200])
    R = [json.loads(l) for l in open(f)]
    if len(R) < 3 or key in seen: continue
    seen.add(key); eps.append((f, R))
print(len(eps), "episodes,", sum(len(R) for _, R in eps), "rounds")
cls = collections.Counter(); tok = collections.Counter(); proj = []; projz = []
for f, R in eps:
    prev = None
    for r in R:
        v = vec(r.get("action") or {}); t = r.get("completion_tokens", 0)
        if v is None: prev = None; continue
        lat = math.hypot(v[0], v[1]); g = abs(v[6]) >= 1; dz = v[2]
        if g: c = "gripper"
        elif abs(dz) >= 3 and lat < 3: c = "vertical"
        elif lat >= 3 and abs(dz) < 3: c = "lateral"
        elif lat >= 3: c = "diagonal"
        elif lat < 3 and abs(dz) < 3 and max(abs(x) for x in v[3:6]) > 0.05: c = "rotate"
        else: c = "nudge(<3cm)"
        # correction: lateral/nudge after a lateral/diagonal move with no gripper change in between
        if prev is not None and c in ("lateral", "nudge(<3cm)") and prev[1] in ("lateral", "diagonal"):
            pv = prev[0]; pl = math.hypot(pv[0], pv[1])
            if pl > 3:
                p = (v[0] * pv[0] + v[1] * pv[1]) / pl  # component of correction along previous move direction
                proj.append((p, pl, math.hypot(v[0], v[1])))
                c = "lat-correction"
        if prev is not None and c == "vertical" and prev[1] == "vertical" and abs(prev[0][2]) > 3:
            projz.append((v[2] * math.copysign(1, prev[0][2]), abs(prev[0][2])))
            c = "vert-correction"
        cls[c] += 1; tok[c] += t; prev = (v, c)
N = sum(cls.values()); T = sum(tok.values())
for c, n in cls.most_common(): print(f"{c:16s} rounds {n/N:5.1%}  tokens {tok[c]/T:5.1%}  mean tok {tok[c]/n:6.0f}")
if proj:
    same = sum(p > 0 for p, _, _ in proj)
    print(f"lateral corrections: {len(proj)}; along previous direction (undershoot) {same/len(proj):.0%}; "
          f"median |corr|/|prev| {sorted(c/l for _, l, c in proj)[len(proj)//2]:.2f}; mean signed proj/prev {sum(p/l for p,l,_ in proj)/len(proj):+.2f}")
if projz:
    same = sum(p > 0 for p, _ in projz)
    print(f"vertical corrections: {len(projz)}; continuing same direction {same/len(projz):.0%}; mean signed ratio {sum(p/l for p,l in projz)/len(projz):+.2f}")
