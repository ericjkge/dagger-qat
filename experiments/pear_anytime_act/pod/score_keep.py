"""Selective retention of earlier rounds' thinking: which parts of old reasoning does PEAR's next action depend on?

For sampled rounds r >= 2 of each recorded episode, teacher-force round r's recorded tokens under:
  full     : all earlier thinking kept (as recorded)
  strip    : earlier thinking removed (answer/action only), as the template renders preserve_thinking=false
  grounded : earlier thinking reduced to its grounded-claim sentences (metric quantities, outcomes of earlier moves,
             object positions / occlusion, self-calibration); everything else dropped
  random   : the same number of characters of earlier thinking, as randomly chosen whole sentences (seeded)
  last     : only the immediately preceding round's thinking kept in full
Records per-token logprobs and top-1 agreement for round r (thinking 't' and answer/action 'a' tokens)."""
import argparse, json, random, re, sys, urllib.request, zlib
from pathlib import Path
sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import IM_END, Template  # noqa: E402

SENT = "\u0000STRIPPED\u0000"
SPLIT = re.compile(r"(?<=[.!?\n])\s+")
GROUNDED = re.compile(
    r"(\d+(\.\d+)?\s*(cm|px|pixels?)\b"                                     # metric / image quantities
    r"|\b(moved|stopped|reached|blocked|overshot|undershot|didn.t reach|only moved|collid|contact|hit)\b"  # move outcomes
    r"|\b(was at|located at|is at|position of|occluded|hidden|out of view|no longer visible)\b"  # positions / occlusion
    r"|\b(scale|per cm|cm per|px per)\b)", re.I)


def sentences(think):
    return [s for s in SPLIT.split(think) if s.strip()]


def think_of(g):
    return g.partition("</think>")[0]


def answer_of(g):
    return g.split("</think>", 1)[-1].lstrip("\n")


def score(url, text, images):
    body = {"text": text, "image_data": images, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 1, "return_text_in_logprobs": True}
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        m = json.loads(r.read())["meta_info"]
    lp, top = m["input_token_logprobs"], m["input_top_logprobs"]
    segs, cur, state, prev = [], None, "other", ""
    for i, (l, tid, t) in enumerate(lp):
        if prev == "<|im_start|>" and t == "assistant":
            state, cur = "hdr", []; segs.append(cur)
        elif t == "<|im_start|>":
            state = "other"
        elif state == "hdr" and t == "<think>":
            state = "think"
        elif state == "think" and t == "</think>":
            state = "answer"
        elif state in ("think", "answer") and l is not None:
            if t == "<|im_end|>":
                state = "other"
            else:
                cur.append((state[0], round(l, 5), int(bool(top[i]) and top[i][0][1] == tid)))
        prev = t
    return segs[-1], m["prompt_tokens"]


def variant(rows, r, cond, rnd):
    """Earlier rounds' generated text under a retention condition; returns (rows', kept_chars, total_chars)."""
    out = []; kept = tot = 0
    for i, x in enumerate(rows):
        if i >= r:
            out.append(x); continue
        th, ans = think_of(x["generated"]), answer_of(x["generated"])
        S = sentences(th); tot += len(th)
        if cond == "full":
            keep = S
        elif cond == "strip":
            keep = []
        elif cond == "grounded":
            keep = [s for s in S if GROUNDED.search(s)]
        elif cond == "last":
            keep = S if i == r - 1 else []
        else:
            raise ValueError(cond)
        kept += sum(len(s) for s in keep)
        out.append((i, keep, ans, S))
    return out, kept, tot


think_of_full = {}


def render(out, cond_budget, rnd):
    """Turns retained sentence lists into rows; for 'random', fill cond_budget chars with random whole sentences."""
    rows = []
    if cond_budget is not None:
        pool = [(i, k) for item in out if isinstance(item, tuple) for i, k in [(item[0], j) for j in range(len(item[3]))]]
        rnd.shuffle(pool); chosen = set(); used = 0
        lens = {item[0]: [len(s) for s in item[3]] for item in out if isinstance(item, tuple)}
        for i, j in pool:
            if used >= cond_budget:
                break
            chosen.add((i, j)); used += lens[i][j]
    for item in out:
        if not isinstance(item, tuple):
            rows.append(item); continue
        i, keep, ans, S = item
        if cond_budget is None and keep is S:  # nothing dropped: the recorded text, byte for byte
            rows.append({"generated": think_of_full[i], "_orig": i}); continue
        if cond_budget is not None:
            keep = [S[j] for j in range(len(S)) if (i, j) in chosen]
        if keep:
            rows.append({"generated": "\n".join(keep) + "\n</think>\n\n" + ans, "_orig": i})
        else:
            rows.append({"generated": SENT + ans, "_orig": i})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--results", default="/workspace/results/dflash2_main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    tmpl = Template("/workspace/models/pear", "xhigh")
    res = Path(a.results)
    for ep in sorted(p.stem for p in res.glob("libero_*.jsonl")):
        rows = [r for r in map(json.loads, (res / f"{ep}.jsonl").open()) if r.get("generated")]
        think_of_full.clear(); think_of_full.update({i: x["generated"] for i, x in enumerate(rows)})
        rounds = sorted(set(list(range(2, len(rows), a.every)) + [len(rows) - 1]) - {0, 1})
        for r in rounds:
            rec = {"episode": ep, "round": r}
            g_out, g_kept, tot = variant(rows, r, "grounded", None)
            for cond in ("full", "strip", "grounded", "random", "last"):
                rnd = random.Random(zlib.crc32(f"{ep}:{r}".encode()))
                if cond == "random":
                    out = [(it[0], [], it[2], it[3]) if isinstance(it, tuple) else it for it in variant(rows, r, "full", None)[0]]
                    rr = render(out, g_kept, rnd); kept = g_kept
                else:
                    out, kept, _ = variant(rows, r, cond, None)
                    rr = render(out, None, rnd)
                # merge rendered earlier rounds back onto the original row dicts (keep frames, state, etc.)
                merged = [dict(rows[x["_orig"]], generated=x["generated"]) if "_orig" in x else x for x in rr]
                text, images = context(merged, r, res / "frames" / ep, tmpl)
                text = text.replace("<think>\n" + SENT, "")
                assert SENT not in text
                toks, n = score(a.url, text + rows[r]["generated"] + IM_END, images)
                rec[cond] = {"prompt_tokens": n, "kept_chars": kept, "tokens": toks}
            rec["old_think_chars"] = tot
            L = {c: len(rec[c]["tokens"]) for c in ("full", "strip", "grounded", "random", "last")}
            assert len(set(L.values())) == 1, (ep, r, L)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            nll = lambda t: -sum(x[1] for x in t if x[0] == "a") / max(1, sum(1 for x in t if x[0] == "a"))
            print(ep, r, {c: (rec[c]["prompt_tokens"], round(nll(rec[c]["tokens"]), 4)) for c in L}, flush=True)
    print("KEEP_DONE")


if __name__ == "__main__":
    main()
