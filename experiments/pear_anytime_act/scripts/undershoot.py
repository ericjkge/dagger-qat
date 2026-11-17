"""Undershoot test, conditioned: previous round a lateral move (>3 cm, no gripper change) whose outcome was 'reached';
this round a lateral follow-up (no gripper, |dz|<3). Ratio r = component of follow-up along previous direction / |prev|.
Also: does the previous round's thinking announce a deliberate partial move?"""
import json, glob, math, re, collections
ARM = ("x", "y", "z", "roll", "pitch", "yaw")
def vec(a):
    try: return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception: return None
files = [f for f in glob.glob("**/libero_*_r*.jsonl", recursive=True) if "/frames/" not in f]
seen = set(); eps = []
for f in sorted(files):
    R = [json.loads(l) for l in open(f)]
    key = f.split("/")[-1] + json.dumps(R[0].get("action"))
    if len(R) >= 3 and key not in seen: seen.add(key); eps.append((f, R))
PART = re.compile(r"\b(partway|part of the way|halfway|most of the way|a bit less|conservative|cautious|smaller step|in steps|first move|then (fine[- ]?tune|adjust|refine))", re.I)
rows = []
for f, R in eps:
    for i in range(1, len(R)):
        p, r = R[i-1], R[i]
        pv, v = vec(p.get("action") or {}), vec(r.get("action") or {})
        if not pv or not v or abs(pv[6]) >= 1 or abs(v[6]) >= 1: continue
        pl = math.hypot(pv[0], pv[1])
        if pl <= 3 or abs(pv[2]) >= 3 or abs(v[2]) >= 3: continue
        along = (v[0]*pv[0] + v[1]*pv[1]) / pl; perp = abs(v[0]*pv[1] - v[1]*pv[0]) / pl
        think = p.get("generated", "").partition("</think>")[0]
        rows.append(dict(f=f, out=p.get("outcome"), pl=pl, r=along/pl, perp=perp/pl, part=bool(PART.search(think[-1500:])),
                         ax="x" if abs(pv[0]) > 2*abs(pv[1]) else "y" if abs(pv[1]) > 2*abs(pv[0]) else "xy",
                         grip=(p.get("state") or {}).get("gripper", {}).get("width", 8) < 7))
def summ(name, S):
    if not S: return
    rs = sorted(x["r"] for x in S); n = len(rs)
    print(f"{name:34s} n={n:4d} same-dir {sum(x>0.05 for x in rs)/n:4.0%}  median r {rs[n//2]:+.2f}  IQR [{rs[n//4]:+.2f},{rs[3*n//4]:+.2f}]  median perp {sorted(x['perp'] for x in S)[n//2]:.2f}")
summ("all lateral->lateral", rows)
summ("prev reached", [x for x in rows if x["out"] == "reached"])
summ("prev stopped", [x for x in rows if x["out"] == "stopped"])
summ("prev reached, no partial-move words", [x for x in rows if x["out"] == "reached" and not x["part"]])
summ("prev reached, partial-move words", [x for x in rows if x["out"] == "reached" and x["part"]])
for ax in ("x", "y", "xy"): summ(f"prev reached, axis {ax}", [x for x in rows if x["out"] == "reached" and x["ax"] == ax])
summ("prev reached, holding object", [x for x in rows if x["out"] == "reached" and x["grip"]])
summ("prev reached, empty gripper", [x for x in rows if x["out"] == "reached" and not x["grip"]])
for lo, hi in ((3, 6), (6, 10), (10, 15), (15, 100)):
    summ(f"prev reached, |prev| {lo}-{hi} cm", [x for x in rows if x["out"] == "reached" and lo <= x["pl"] < hi])
