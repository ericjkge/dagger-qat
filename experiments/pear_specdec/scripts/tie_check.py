"""Exactness follow-up: where greedy AR and greedy DFlash2 outputs first differ, is the target's top-1 vs top-2 margin a
bf16 near-tie (numerical, kernel-shape dependent) or a real preference (which would indicate a verification bug)?

For each divergence (context, common prefix length L): prompt = recorded context + common prefix tokens (as text);
ask the running server for 1 token with top-2 logprobs (that token comes from the PREFILL path: no speculation
involved). Report the logprob margin and which of the two divergent tokens are the top-2.
Run with the sim venv python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_episodes import Template  # noqa: E402
from replay_bench import context  # noqa: E402
import urllib.request  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--results", default="/workspace/results/dflash2_main")
    ap.add_argument("--bench", default="/workspace/results/bench")
    ap.add_argument("--configs", default="dflash2_k7,dflash2_k3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/workspace/models/pear/tokenizer.json")
    tmpl = Template("/workspace/models/pear", "xhigh")
    load = lambda c: {(r["episode"], r["round"]): r["output_ids"]  # noqa: E731
                      for r in map(json.loads, open(Path(args.bench) / f"exact_{c}.jsonl"))}
    ar = load("ar")
    res = []
    for c in args.configs.split(","):
        for (ep, rnd), ids in load(c).items():
            ref = ar[(ep, rnd)]
            L = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), None)
            if L is None:
                res.append({"config": c, "context": f"{ep}:{rnd}", "identical": True})
                continue
            rows = [json.loads(x) for x in open(Path(args.results) / f"{ep}.jsonl")]
            text, images = context(rows, rnd, Path(args.results) / "frames" / ep, tmpl)
            prefix = tok.decode(ref[:L], skip_special_tokens=False)
            body = {"text": text + prefix, "image_data": images,
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                    "return_logprob": True, "top_logprobs_num": 5, "logprob_start_len": -1}
            req = urllib.request.Request(args.url + "/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            m = json.loads(urllib.request.urlopen(req, timeout=600).read())["meta_info"]
            top = m["output_top_logprobs"][0]  # [(logprob, id, text), ...]
            rec = {"config": c, "context": f"{ep}:{rnd}", "diverge_at": L, "ar_token": ref[L], "spec_token": ids[L],
                   "top": [(round(t[0], 4), t[1]) for t in top[:5]],
                   "margin_top1_top2": round(top[0][0] - top[1][0], 4),
                   "both_in_top2": {ref[L], ids[L]} <= {top[0][1], top[1][1]}}
            res.append(rec)
            print(rec, flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
