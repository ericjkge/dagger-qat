"""Summarize the paired replay benchmark (results/bench/<config>.jsonl, exact_<config>.jsonl).

Throughput metric: decode tokens/s = completion_tokens / (e2e_latency - prefill_latency) (server-side), plus SGLang's
own decode_throughput. Pairing: same (episode, round, rep) contexts across configs; speedup CIs by bootstrap over
contexts. Exactness: greedy (temperature 0) outputs of each config vs AR, longest common prefix of token ids.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p: Path):
    return [json.loads(x) for x in p.open()] if p.exists() else []


def decode_tps(r):
    d = (r["e2e_latency"] or r["wall_s"]) - (r["prefill_latency"] or 0)
    return r["completion_tokens"] / d if d > 0 else float("nan")


def main():
    bench = Path(sys.argv[1])
    cfgs = [p.stem for p in sorted(bench.glob("*.jsonl")) if not p.stem.startswith("exact_")]
    data = {c: load(bench / f"{c}.jsonl") for c in cfgs}
    out = {"configs": {}}
    key = lambda r: (r["episode"], r["round"], r["rep"])  # noqa: E731
    ar = {key(r): r for r in data.get("ar", [])}
    for c, rows in data.items():
        if not rows:
            continue
        tps = np.array([decode_tps(r) for r in rows])
        tpv = [r["completion_tokens"] / r["verify_ct"] for r in rows if r.get("verify_ct")]
        per_tok_ms = [1000 / t for t in tps]
        step_ms = [1000 * ((r["e2e_latency"] - r["prefill_latency"]) / r["verify_ct"]) for r in rows if r.get("verify_ct")]
        d = {"n": len(rows), "decode_tps_mean": float(np.nanmean(tps)),
             "decode_tps_by_ctx": sorted([(r["prompt_tokens"], round(decode_tps(r), 1)) for r in rows]),
             "tokens_per_verify": float(np.mean(tpv)) if tpv else None,
             "ms_per_verify_step": float(np.mean(step_ms)) if step_ms else None,
             "ms_per_token": float(np.mean(per_tok_ms)),
             "prefill_s_mean_uncached": float(np.mean([r["prefill_latency"] for r in rows if r["rep"] == 0])),
             "completion_tokens_mean": float(np.mean([r["completion_tokens"] for r in rows])),
             "round_latency_s_mean": float(np.mean([r["e2e_latency"] for r in rows]))}
        if ar and c != "ar":
            pairs = [(decode_tps(r), decode_tps(ar[key(r)])) for r in rows if key(r) in ar]
            if pairs:
                s = np.array([a / b for a, b in pairs])
                rng = np.random.default_rng(0)
                bs = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(2000)]
                d["decode_speedup_vs_ar"] = float(s.mean())
                d["decode_speedup_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["configs"][c] = d
    # exactness
    ex = {c: {(r["episode"], r["round"]): r.get("output_ids") for r in load(bench / f"exact_{c}.jsonl")} for c in cfgs}
    if ex.get("ar"):
        out["exactness_vs_ar"] = {}
        for c, m in ex.items():
            if c == "ar" or not m:
                continue
            res = []
            for k, ids in m.items():
                ref = ex["ar"].get(k)
                if ref is None or ids is None:
                    continue
                lcp = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
                res.append({"context": f"{k[0]}:{k[1]}", "len": len(ids), "ref_len": len(ref), "common_prefix": lcp,
                            "identical": ids == ref})
            out["exactness_vs_ar"][c] = res
    print(json.dumps(out, indent=1))
    (bench / "bench_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
