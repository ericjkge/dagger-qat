"""Per-iteration analysis of DFlash2 on recorded PEAR episodes (needs spec_steps.log from the patched server).

spec_steps.log line: "<rid> <n> <a>"  = at output position n (tokens already emitted by this request) DFlash2 proposed 7
drafts and a of them were accepted (plus one bonus token). Joined with the request's recorded output_ids.

Outputs:
  * acceptance by phase (reason / answer) and by relative position within the reasoning (deciles)
  * acceptance by content of the preceding context (>= 2 digits in the last 6 tokens: numeric/coordinate reasoning vs
    prose) -> sub-phase signal; labeled before the step so the label does not depend on the outcome
  * acceptance vs round index and vs context length (long-history degradation, cf. DLite)
  * complementarity with retrieval: at each DFlash2 step, the in-context n-gram drafter's match length m on the
    realized continuation (unbiased acceptance estimate for a deterministic drafter). Reports
       - P(m > a), mean max(a, m) + 1 vs mean a + 1: the per-step oracle-switch upper bound for a DFlash2+retrieval
         hybrid at equal verify cost (an upper bound: a real router must predict which drafter to trust)
       - the same with "prev round only" and "shuffled other-episode round" corpora (temporal-structure control)
Usage: python analyze_steps.py <results_dir> --tokenizer <tokenizer.json> [--out file.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
DIGIT = re.compile(r"[0-9]")


def load_steps(path: Path):
    steps = defaultdict(list)
    for line in path.open():
        rid, n, a = line.split()
        steps[rid].append((int(n), int(a)))
    return steps


def match_len(d, seq, i):
    m = 0
    while m < len(d) and i + m < len(seq) and d[m] == seq[i + m]:
        m += 1
    return m


def boot(xs, n=1000, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    ms = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(n)]
    return [round(float(np.percentile(ms, 2.5)), 4), round(float(np.percentile(ms, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    res = Path(args.results)
    steps = load_steps(res / "spec_steps.log")
    eps = {}
    for p in sorted(res.glob("*.jsonl")):
        if p.name != "episodes.jsonl":
            rows = [json.loads(x) for x in p.open()]
            rows = [r for r in rows if not degenerate(r)]
            if rows:
                eps[p.stem] = rows
    names = list(eps)
    rng = random.Random(0)
    # per-round full generated ids (reason + answer) for corpora
    gen = {n: [sum((q["meta"]["output_ids"] for q in r["requests"]), []) for r in rows] for n, rows in eps.items()}

    recs = []  # one per DFlash2 step
    missing = 0
    for n in names:
        ctx = NgramIndex()
        for ri, r in enumerate(eps[n]):
            prev = NgramIndex(gen[n][ri - 1]) if ri > 0 else None
            others = [o for o in names if o != n]
            o = rng.choice(others) if others else None
            shuf = NgramIndex(gen[o][min(ri, len(gen[o]) - 1)]) if o else None
            offset = 0  # position within the round's generated ids
            round_ids = gen[n][ri]
            fed = 0  # round tokens already appended to ctx
            reason_len = next((q["meta"]["completion_tokens"] for q in r["requests"] if q["phase"] == "reason"), None)
            for q in r["requests"]:
                m = q["meta"]
                ids = m["output_ids"]
                st = steps.get(m["id"])
                if not st:
                    missing += 1
                    offset += len(ids)
                    continue
                for pos, a in st:
                    g = offset + pos  # position in round_ids of the first drafted token
                    while fed < g:
                        ctx.append(round_ids[fed])
                        fed += 1
                    hist = round_ids[max(0, g - 8):g]
                    if len(hist) < 2 and ri > 0:
                        hist = (gen[n][ri - 1] + round_ids[:g])[-8:]
                    mc = match_len(ctx.propose(hist, K), round_ids, g) if hist else 0
                    mp = match_len(prev.propose(hist, K), round_ids, g) if (prev and hist) else 0
                    ms = match_len(shuf.propose(hist, K), round_ids, g) if (shuf and hist) else 0
                    # label from the CONTEXT before the step (the last 6 emitted tokens), never from the accepted span
                    # (labeling by the outcome would bias long accepted spans toward containing digits)
                    span = tok.decode(round_ids[max(0, g - 6):g], skip_special_tokens=False)
                    recs.append({"ep": n, "round": ri, "phase": q["phase"], "g": g, "a": a, "m_ctx": mc,
                                 "m_prev": mp, "m_shuf": ms,
                                 "relpos": (g / reason_len) if (q["phase"] == "reason" and reason_len) else None,
                                 "numeric": len(DIGIT.findall(span)) >= 2,
                                 "ctx_tokens": m["prompt_tokens"] + pos})
                offset += len(ids)
            while fed < len(round_ids):
                ctx.append(round_ids[fed])
                fed += 1

    A = lambda rs: np.array([x["a"] for x in rs], float)  # noqa: E731
    out = {"steps": len(recs), "requests_missing_steps": missing, "excluded_degenerate_rounds": DEGENERATE_LOG}
    for ph in ("reason", "answer"):
        rs = [x for x in recs if x["phase"] == ph]
        if not rs:
            continue
        a = A(rs)
        d = {"steps": len(rs), "tokens_per_step": float(a.mean() + 1), "ci95": [c + 1 for c in boot(a)]}
        for key in ("m_ctx", "m_prev", "m_shuf"):
            mm = np.array([x[key] for x in rs], float)
            d[key] = {"retrieval_alone_tokens_per_step": float(mm.mean() + 1),
                      "p_retrieval_beats_dflash": float((mm > a).mean()),
                      "oracle_switch_tokens_per_step": float(np.maximum(a, mm).mean() + 1),
                      "oracle_gain_pct": float((np.maximum(a, mm).mean() + 1) / (a.mean() + 1) * 100 - 100),
                      "oracle_gain_ci95_pct": [round(100 * ((x + 1) / (a.mean() + 1)) - 100, 2)
                                               for x in boot(np.maximum(a, mm))]}
        out[ph] = d
    rs = [x for x in recs if x["phase"] == "reason"]
    out["reason_by_decile"] = {}
    for k in range(10):
        sel = [x for x in rs if x["relpos"] is not None and k / 10 <= x["relpos"] < (k + 1) / 10]
        if sel:
            out["reason_by_decile"][k] = round(float(A(sel).mean() + 1), 3)
    for flag in (True, False):
        sel = [x for x in rs if x["numeric"] == flag]
        out[f"reason_numeric_{flag}"] = {"steps": len(sel), "tokens_per_step": float(A(sel).mean() + 1),
                                         "ci95": [c + 1 for c in boot(A(sel))]}
    by_round = defaultdict(list)
    for x in rs:
        by_round[min(x["round"], 15)].append(x["a"])
    out["reason_by_round"] = {k: [len(v), round(float(np.mean(v) + 1), 3)] for k, v in sorted(by_round.items())}
    bins = [0, 8000, 16000, 32000, 64000, 128000, 1e9]
    out["reason_by_ctx"] = {}
    for lo, hi in zip(bins, bins[1:]):
        sel = [x for x in rs if lo <= x["ctx_tokens"] < hi]
        if sel:
            out["reason_by_ctx"][f"{int(lo)}-{int(hi) if hi < 1e9 else 'inf'}"] = [len(sel),
                                                                                round(float(A(sel).mean() + 1), 3)]
    # per-episode tokens/step (for CIs across episodes rather than steps)
    per_ep = defaultdict(list)
    for x in rs:
        per_ep[x["ep"]].append(x["a"])
    out["reason_per_episode"] = {k: round(float(np.mean(v) + 1), 3) for k, v in per_ep.items()}
    txt = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
