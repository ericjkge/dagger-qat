"""Generic-summary baseline (S4): PEAR summarizes each recorded round's thinking, capped at the token count of that
round's geometry paragraphs (S3), so S3 and S4 keep equal budgets. Thinking disabled, greedy."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer
from common import split_gen, geo_only, post
RES = Path("/workspace/results/dflash2_main"); OUT = Path("/workspace/results/pilot/summaries.jsonl")
tok = AutoTokenizer.from_pretrained("/workspace/models/pear")
jobs = []
for p in sorted(RES.glob("libero_*.jsonl")):
    for row in map(json.loads, p.open()):
        g = row.get("generated")
        if not g or "</think>" not in g:
            continue
        th, _ = split_gen(g)
        n = len(tok(geo_only(th), add_special_tokens=False).input_ids)
        jobs.append((row["episode"], row["round"], th, n))
def run(j):
    ep, r, th, n = j
    if n == 0:
        return {"episode": ep, "round": r, "budget": 0, "summary": ""}
    prompt = ("<|im_start|>user\nBelow is your own reasoning from one step of a robot manipulation episode. Rewrite it as a "
              f"compact note of at most {n} tokens that keeps everything you would need in later steps (facts, measurements, "
              "conclusions, plans). Output only the note.\n\n<reasoning>\n" + th + "\n</reasoning><|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    out = post("http://127.0.0.1:30000", {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": n}})
    return {"episode": ep, "round": r, "budget": n, "summary": out["text"].strip(), "tokens": out["meta_info"]["completion_tokens"]}
with ThreadPoolExecutor(8) as ex:
    for rec in ex.map(run, jobs):
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
print("SUMMARIES_DONE", len(jobs))
