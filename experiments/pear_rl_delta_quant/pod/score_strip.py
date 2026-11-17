"""Functional check for context pruning: does dropping earlier rounds' thinking change PEAR's next round?

For sampled rounds r >= 1 of each recorded episode, teacher-force round r's recorded tokens (thinking + action) under
  full : the recorded context (preserve_thinking, all earlier thinking kept)
  strip: earlier assistant turns rendered as the template does with preserve_thinking=false (answer/action only)
and record per-token logprobs and top-1 agreement for round r only. Same server (BF16 PEAR) for both."""
import argparse, json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import IM_END, Template  # noqa: E402

SENT = "\u0000STRIPPED\u0000"


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
        rounds = sorted(set(list(range(1, len(rows), a.every)) + [len(rows) - 1]) - {0})
        for r in rounds:
            out = {"episode": ep, "round": r}
            for cond in ("full", "strip"):
                rr = rows if cond == "full" else [dict(x, generated=SENT + x["generated"].split("</think>", 1)[-1].lstrip("\n"))
                                                  if i < r else x for i, x in enumerate(rows)]
                text, images = context(rr, r, res / "frames" / ep, tmpl)
                text = text.replace("<think>\n" + SENT, "")
                assert SENT not in text
                toks, n = score(a.url, text + rows[r]["generated"] + IM_END, images)
                out[cond] = {"prompt_tokens": n, "tokens": toks}
            assert len(out["full"]["tokens"]) == len(out["strip"]["tokens"]), (ep, r)
            with open(a.out, "a") as f:
                f.write(json.dumps(out) + "\n")
            f_, s_ = out["full"], out["strip"]
            nll = lambda t: -sum(x[1] for x in t) / max(1, len(t))
            print(ep, r, f_["prompt_tokens"], s_["prompt_tokens"], round(nll(f_["tokens"]), 4), round(nll(s_["tokens"]), 4), flush=True)
    print("STRIP_DONE")


if __name__ == "__main__":
    main()
