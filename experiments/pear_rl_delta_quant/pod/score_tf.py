"""Teacher-forced scoring of the recorded BF16-PEAR episodes under whatever arm the server runs.

One prefill per episode over its full final context (preserve_thinking keeps every earlier turn verbatim), with input
logprobs. Each generated token is classed as thinking (inside <think>..</think>) or answer (after </think>: the
action). Output: per-episode token logprobs of the model-generated tokens, plus top-1 agreement.
Run with the sim venv python from /workspace (needs replay_bench/run_episodes + pear package)."""
import argparse, json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import IM_END, Template  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--results", default="/workspace/results/dflash2_main")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmpl = Template("/workspace/models/pear", "xhigh")
    res = Path(a.results)
    eps = sorted(p.stem for p in res.glob("libero_*.jsonl"))
    for ep in eps:
        rows = [json.loads(x) for x in (res / f"{ep}.jsonl").open()]
        rows = [r for r in rows if r.get("generated")]
        last = len(rows) - 1
        text, images = context(rows, last, res / "frames" / ep, tmpl)
        text += rows[last]["generated"] + IM_END
        body = {"text": text, "image_data": images, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 1,
                "return_text_in_logprobs": True}
        req = urllib.request.Request(a.url + "/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            m = json.loads(r.read())["meta_info"]
        lp, top = m["input_token_logprobs"], m["input_top_logprobs"]
        toks, state, prev = [], "user", ""
        for i, (l, tid, t) in enumerate(lp):
            if prev == "<|im_start|>" and t == "assistant":
                state = "asst_header"
            elif t == "<|im_start|>":
                state = "other"
            elif state == "asst_header" and t == "<think>":
                state = "think"
            elif state == "think" and t == "</think>":
                state = "answer"
            elif state in ("think", "answer") and l is not None:
                if t == "<|im_end|>":
                    state = "other"
                else:
                    t1 = top[i][0][1] if top[i] else None
                    toks.append((state[0], round(l, 5), int(t1 == tid)))
            prev = t
        rec = {"arm": a.arm, "episode": ep, "prompt_tokens": m["prompt_tokens"], "rounds": len(rows),
               "n_think": sum(x[0] == "t" for x in toks), "n_answer": sum(x[0] == "a" for x in toks),
               "tokens": toks}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(ep, rec["prompt_tokens"], rec["n_think"], rec["n_answer"],
              round(-sum(x[1] for x in toks) / max(1, len(toks)), 4), flush=True)
    print("SCORE_DONE", a.arm)


if __name__ == "__main__":
    main()
