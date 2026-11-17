"""Paired latency benchmark: regenerate the reasoning of recorded rounds from their EXACT recorded contexts
(system prompt, earlier turns verbatim, recorded frames) under whatever spec config the server runs.

Same contexts across configs -> paired comparison of decode throughput and per-round latency. Each context is generated
`--reps` times (temperature 1.0, top-p 0.95, stop at </think>), so each config sees the same prompts; outputs differ
(sampling), which is why we compare decode throughput (tokens / decode time), not just wall time.

Run with the sim venv python (needs jinja2, PIL and the pear package).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_episodes import (IM_END, REMINDER, SYSTEM_PROMPT_TEMPLATE, THINK_END, Template,  # noqa: E402
                          generate, png_b64)
from pear.agent_loop.protocol import JsonBlockProtocol  # noqa: E402
from pear.envs.libero.env import LiberoEnv  # noqa: E402


def user_blocks(state):
    text = JsonBlockProtocol.turn_text(state, REMINDER)
    blocks = [{"type": "text", "text": text}]
    for name in LiberoEnv.CAMERAS:
        blocks += [{"type": "text", "text": f"[{name}]"}, {"type": "image", "image": None}]
    return blocks


def context(rows, r, frames: Path, tmpl: Template):
    """Prompt text and images for round r of a recorded episode."""
    system = JsonBlockProtocol.system_prompt(SYSTEM_PROMPT_TEMPLATE, LiberoEnv.ENVIRONMENT_PROMPT,
                                             rows[0]["instruction"])
    text = tmpl.first(system, user_blocks(rows[0]["state"]))
    for i in range(1, r + 1):
        text += rows[i - 1]["generated"] + IM_END + "\n" + tmpl.next_user(user_blocks(rows[i]["state"]))
    images = [png_b64(Image.open(frames / f"round{i:03d}_{cam}.png")) for i in range(r + 1) for cam in LiberoEnv.CAMERAS]
    return text, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model-dir", default="/workspace/models/pear")
    ap.add_argument("--results", required=True)
    ap.add_argument("--select", required=True, help="comma list episode:round")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", type=int, default=0, help="1: temperature 0, fixed max-new, save output ids (exactness)")
    ap.add_argument("--max-new", type=int, default=8192)
    args = ap.parse_args()
    tok = json.loads(Path(args.model_dir, "tokenizer.json").read_text())
    think_end_id = next(t["id"] for t in tok["added_tokens"] if t["content"] == THINK_END)
    tmpl = Template(args.model_dir, "xhigh")
    res = Path(args.results)
    # warm-up (CUDA graphs / JIT) on a tiny prompt
    generate(args.url, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", [], {"max_new_tokens": 16})
    for spec in args.select.split(","):
        ep, r = spec.rsplit(":", 1)
        rows = [json.loads(x) for x in (res / f"{ep}.jsonl").open()]
        text, images = context(rows, int(r), res / "frames" / ep, tmpl)
        for rep in range(args.reps):
            sp = ({"temperature": 0.0, "max_new_tokens": args.max_new} if args.greedy else
                  {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": args.max_new})
            out, wall = generate(args.url, text, images, {**sp, "stop_token_ids": [think_end_id],
                                                          "skip_special_tokens": False, "no_stop_trim": True})
            m = out["meta_info"]
            rec = {"config": args.config, "episode": ep, "round": int(r), "rep": rep, "wall_s": wall,
                   "prompt_tokens": m["prompt_tokens"], "cached_tokens": m.get("cached_tokens"),
                   "completion_tokens": m["completion_tokens"], "verify_ct": m.get("spec_verify_ct"),
                   "e2e_latency": m.get("e2e_latency"), "prefill_latency": m.get("prefill_latency"),
                   "first_token_latency": m.get("first_token_latency"),
                   "decode_throughput": m.get("decode_throughput"), "time": time.time(),
                   "hist": m.get("spec_correct_drafts_histogram")}
            if args.greedy:
                rec["output_ids"] = [x[1] for x in (m.get("output_token_logprobs") or [])]
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
