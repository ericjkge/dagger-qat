"""Offline analysis of recorded PEAR episodes (run_episodes.py output).

1. Experiment A (phases): DFlash2 server counters per phase (reason request vs answer request): tokens, verify steps,
   tokens/verify, accepted-drafts histograms; Amdahl bound for making each phase free.
2. Experiment B (cross-round reuse): replay deterministic retrieval drafters over the realized token sequence.
   For a deterministic drafter under lossless verification, P(first m drafts accepted) = prod p(d_j | prefix), which is
   exactly P(realized sample matches the drafts for m positions). So the realized match length is an UNBIASED estimate
   of expected acceptance, and replaying the drafter along the realized sequence reproduces the online process in
   distribution. Drafters (K = 7, as DFlash2):
     ctx        suffix match over the episode's earlier generated tokens + the current round so far (prompt-lookup;
                earlier rounds' thinking is in PEAR's context because preserve_thinking=true)
     prev       only the immediately preceding round's generated tokens
     shuffled   the same-index round of a DIFFERENT episode (control: generic phrasing, not this episode's history)
     self       only the current round so far (within-round repetition)
3. Complementarity proxy: share of positions where a retrieval drafter would propose >= 4 matching tokens.
Usage: python analyze_traces.py <results_dir> [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pear_specdec.drafters import NgramIndex  # noqa: E402

K = 7


DEGENERATE_TPV = 7.0  # reasoning tokens per verify step at/above this = near-full blocks every step (repetition loop)
DEGENERATE_LOG = []


def degenerate(r) -> bool:
    """A round whose reasoning is a repetition loop (e.g. '~?~?~?...' at 180k context); excluded from aggregates and
    listed in DEGENERATE_LOG. Rounds after an excluded round are kept (their contexts are what was really served)."""
    q = next((q for q in r["requests"] if q["phase"] in ("reason", "all")), None)
    if q is None:
        return False
    m = q["meta"]
    tpv = m["completion_tokens"] / max(1, m.get("spec_verify_ct") or 1)
    if tpv >= DEGENERATE_TPV and m["completion_tokens"] > 1000:
        DEGENERATE_LOG.append({"episode": r["episode"], "round": r["round"], "tokens": m["completion_tokens"],
                               "tokens_per_verify": round(tpv, 2)})
        return True
    return False


def load(results: Path):
    eps = {}
    for p in sorted(results.glob("*.jsonl")):
        if p.name == "episodes.jsonl":
            continue
        rows = [json.loads(line) for line in p.open()]
        rows = [r for r in rows if not degenerate(r)]
        if rows:
            eps[p.stem] = rows
    return eps


def phase_stats(eps):
    agg = defaultdict(lambda: {"tokens": 0, "verify": 0, "hist": np.zeros(K + 1), "lat": 0.0, "n": 0})
    per_round = []
    for name, rows in eps.items():
        for r in rows:
            for q in r["requests"]:
                m = q["meta"]
                a = agg[q["phase"]]
                a["tokens"] += m["completion_tokens"]
                a["verify"] += m.get("spec_verify_ct") or 0
                h = m.get("spec_correct_drafts_histogram")
                if h:
                    a["hist"][: len(h)] += h
                a["lat"] += m.get("e2e_latency", q["wall_s"])
                a["n"] += 1
            reason = next((q["meta"] for q in r["requests"] if q["phase"] == "reason"), None)
            if reason:
                per_round.append({"episode": name, "round": r["round"], "prompt_tokens": reason["prompt_tokens"],
                                  "cached": reason.get("cached_tokens"), "reason_tokens": reason["completion_tokens"],
                                  "tau": reason["completion_tokens"] / max(1, reason.get("spec_verify_ct") or 1),
                                  "decode_tps": reason.get("decode_throughput"),
                                  "prefill_s": reason.get("prefill_latency"), "env_s": r.get("env_s")})
    out = {}
    tot = sum(a["tokens"] for a in agg.values())
    for ph, a in agg.items():
        h = a["hist"]
        out[ph] = {"tokens": a["tokens"], "verify": a["verify"], "requests": a["n"],
                   "tokens_per_verify": a["tokens"] / max(1, a["verify"]),
                   "share_of_tokens": a["tokens"] / tot,
                   "accepted_hist_frac": (h / max(1, h.sum())).round(4).tolist(),
                   "full_block_frac": float(h[-1] / max(1, h.sum())),
                   "amdahl_if_free": tot / max(1, tot - a["tokens"]),
                   "server_latency_s": a["lat"]}
    return out, per_round


def replay(seq, index: NgramIndex | None, live: bool, seed_hist, k=K):
    """Simulate lossless decoding of `seq` with a deterministic n-gram drafter over `index`; returns accepted counts.
    live=True: the current round's tokens are appended to the index as they are emitted (ctx / self drafters)."""
    i, accepted = 0, []
    hist = list(seed_hist)
    while i < len(seq):
        d = index.propose(hist[-8:], k) if (index is not None and hist) else []
        m = 0
        while m < len(d) and i + m < len(seq) and d[m] == seq[i + m]:
            m += 1
        accepted.append(m)
        step = seq[i:i + m + 1]
        hist.extend(step)
        if live:
            index.extend(step)
        i += m + 1
    return accepted


def retrieval(eps, seed=0):
    rng = random.Random(seed)
    names = list(eps)
    gen = {n: [sum((q["meta"]["output_ids"] for q in r["requests"]), []) for r in rows] for n, rows in eps.items()}
    res = defaultdict(list)
    for n in names:
        others = [o for o in names if o != n]
        ctx = NgramIndex(gen[n][0])  # grows with the episode
        for ri in range(1, len(gen[n])):
            seq = gen[n][ri]
            tail = gen[n][ri - 1][-8:]  # history at round start: end of the previous turn (the user turn's ids are
            # not recorded; matching resumes after the first emitted tokens)
            runs = {}
            runs["prev"] = replay(seq, NgramIndex(gen[n][ri - 1]), False, [])
            if others:
                o = rng.choice(others)
                runs["shuffled"] = replay(seq, NgramIndex(gen[o][min(ri, len(gen[o]) - 1)]), False, [])
            runs["self"] = replay(seq, NgramIndex(), True, [])
            runs["ctx"] = replay(seq, ctx, True, [])  # appends this round to ctx as it goes
            del tail
            for dn, acc in runs.items():
                res[dn].append({"episode": n, "round": ri, "tokens": len(seq), "iters": len(acc),
                                "ge4": sum(a >= 4 for a in acc)})
    summ = {}
    for dn, rows in res.items():
        t = sum(r["tokens"] for r in rows)
        it = sum(r["iters"] for r in rows)
        summ[dn] = {"rounds": len(rows), "tokens": t, "tokens_per_iter": t / max(1, it),
                    "frac_iters_ge4": sum(r["ge4"] for r in rows) / max(1, it),
                    "per_round_tpi": [r["tokens"] / r["iters"] for r in rows]}
    return summ


def bootstrap_ci(xs, n=2000, seed=0):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    means = [rng.choice(xs, len(xs)).mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out")
    args = ap.parse_args()
    eps = load(Path(args.results))
    phases, per_round = phase_stats(eps)
    ret = retrieval(eps)
    for d in ret.values():
        d["per_round_tpi_ci95"] = bootstrap_ci(d["per_round_tpi"])
        d["per_round_tpi_mean"] = float(np.mean(d["per_round_tpi"])) if d["per_round_tpi"] else None
    taus = [r["tau"] for r in per_round]
    summary = {"excluded_degenerate_rounds": DEGENERATE_LOG, "episodes": len(eps), "rounds": sum(len(v) for v in eps.values()),
               "episode_end": [json.loads(line) for line in (Path(args.results) / "episodes.jsonl").open()]
               if (Path(args.results) / "episodes.jsonl").exists() else [],
               "phases": phases, "reason_tau_round_mean": float(np.mean(taus)) if taus else None,
               "reason_tau_round_ci95": bootstrap_ci(taus), "retrieval": ret, "per_round": per_round}
    txt = json.dumps(summary, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(txt)
    brief = {k: v for k, v in summary.items() if k not in ("per_round",)}
    for d in brief["retrieval"].values():
        d.pop("per_round_tpi", None)
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
