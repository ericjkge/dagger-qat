"""Closed-loop comparison of exit rules. usage: python3 analyze_arms.py <results_dir> [arm ...]
Each arm dir holds episodes.jsonl and per-episode <name>.jsonl round logs from run_anytime.py.
Reports per arm: success, thinking+answer tokens/episode, probe tokens, rounds; paired differences vs 'full' on the
episodes both arms finished, with episode-bootstrap 95% CIs."""
import json, random, sys
K = 4
from pathlib import Path


def load(d):
    eps = {}
    p = Path(d) / "episodes.jsonl"
    if not p.exists():
        return eps
    for l in p.open():
        x = json.loads(l)
        if "error" in x:
            continue
        lp = Path(d) / f"{x['episode']}.jsonl"
        rounds = [json.loads(r) for r in lp.open()] if lp.exists() else []
        x["probe_tokens"] = sum(r.get("probe_tokens", 0) for r in rounds)
        x["exits"] = sum(r.get("exited", False) for r in rounds)
        x["think_tokens"] = sum(r["completion_tokens"] for r in rounds)
        # latency proxy (decode steps on the critical path) and total compute (all chains)
        lat = comp = 0
        for r in rounds:
            c = r.get("consensus")
            if c and c.get("m_tok"):
                lat += c.get("latency_tok", max(c["m_tok"]))
                comp += sum(c["m_tok"]) + ((1 if d.name == "arm_single" else K) - len(c["m_tok"])) * c.get("latency_tok", max(c["m_tok"]))
            else:
                lat += r["completion_tokens"]; comp += r["completion_tokens"] + r.get("probe_tokens", 0)
        if not rounds:
            lat = comp = x["tokens"]
        x["lat_tokens"] = lat; x["compute_tokens"] = comp
        eps[x["episode"]] = x
    return eps


def boot(vals, f=lambda v: sum(v) / len(v), n=2000, seed=0):
    rnd = random.Random(seed); s = sorted(f([rnd.choice(vals) for _ in vals]) for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main(root, arms):
    A = {a: load(Path(root) / f"arm_{a}") for a in arms}
    print(f"{'arm':9s} {'n':>3s} {'success':>8s} {'lat/ep':>8s} {'comp/ep':>8s} {'rounds':>6s}")
    for a, E in A.items():
        if not E:
            continue
        v = list(E.values()); n = len(v)
        print(f"{a:9s} {n:3d} {sum(x['success'] for x in v)/n:8.3f} {sum(x['lat_tokens'] for x in v)/n:8.0f} "
              f"{sum(x['compute_tokens'] for x in v)/n:8.0f} {sum(x['rounds'] for x in v)/n:6.1f}")
    base = arms[0]
    A["full"] = A.pop(base) if base != "full" else A["full"]
    for a, E in A.items():
        if a == "full":
            continue
        common = sorted(set(E) & set(A["full"]))
        if not common:
            continue
        ds = [E[k]["success"] - A["full"][k]["success"] for k in common]
        tf = [A["full"][k]["lat_tokens"] for k in common]; ta = [E[k]["lat_tokens"] for k in common]
        ratio = lambda idx: sum(ta[i] for i in idx) / sum(tf[i] for i in idx)
        idx = list(range(len(common)))
        rnd = random.Random(0); rs = sorted(ratio([rnd.choice(idx) for _ in idx]) for _ in range(2000))
        lo, hi = boot(ds)
        print(f"\n{a} vs full on {len(common)} paired episodes:")
        print(f"  success diff {sum(ds)/len(ds):+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  "
              f"(full {sum(A['full'][k]['success'] for k in common)}/{len(common)}, {a} {sum(E[k]['success'] for k in common)}/{len(common)})")
        print(f"  latency-token ratio (critical path) {ratio(idx):.3f} [95% CI {rs[50]:.3f}, {rs[1949]:.3f}]")
        cr = sum(E[k]["compute_tokens"] for k in common) / sum(A["full"][k]["compute_tokens"] for k in common)
        print(f"  compute-token ratio (all chains) {cr:.2f}")
        rr = [E[k]["rounds"] / A["full"][k]["rounds"] for k in common]
        print(f"  mean rounds ratio {sum(rr)/len(rr):.2f}")
        for suite in ("spatial", "object", "goal", "10"):
            ks = [k for k in common if f"libero_{suite}_" in k]
            if ks:
                print(f"   {suite:8s} success full {sum(A['full'][k]['success'] for k in ks)}/{len(ks)} {a} {sum(E[k]['success'] for k in ks)}/{len(ks)}"
                      f"  latency ratio {sum(E[k]['lat_tokens'] for k in ks)/max(1,sum(A['full'][k]['lat_tokens'] for k in ks)):.2f}"
                      f"  rounds {sum(E[k]['rounds'] for k in ks)}/{sum(A['full'][k]['rounds'] for k in ks)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["full", "exact", "tol"])
