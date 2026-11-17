"""Single-node PEAR LIBERO episode runner against an SGLang server (native /generate API).

Reproduces PEAR's conversation (pear/agent_loop: EnvironmentTool + PearAgentLoop) without Ray/verl:
  * system prompt = JsonBlockProtocol.system_prompt(system_prompt.md, LiberoEnv.environment_prompt(result), instruction)
  * user turn     = json.dumps(state) + "\\n\\n" + REMINDER, then "[agentview]" <image> "[wrist]" <image>
  * chat template = the model's (reasoning_effort xhigh default, preserve_thinking true); later turns are appended
    as text exactly like verl's token-level concatenation: <generated text><|im_end|>\\n<user turn><|im_start|>assistant\\n<think>\\n
  * sampling      = temperature 1.0, top-p 0.95 (PEAR deploy config); 256k response-token budget per episode
  * action        = last ```json block after </think>, checked against LiberoEnv.SCHEMA; one MotionController call

Each round is generated with TWO requests when --split (default): (A) up to and including </think> (stop on that
token id), (B) the rest. The per-request server counters (completion_tokens, spec_verify_ct) then give the
speculative acceptance of the reasoning phase and of the answer/action phase separately. Splitting does not change
the sampled distribution (generation simply resumes from the same prefix; prefix caching makes B's prefill a hit).

Writes one JSON line per round to <out>/<episode>.jsonl as it goes, plus the frames as PNG (for offline replay).
Run with the sim venv's python and simenv.sh sourced.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import jinja2

sys.path.insert(0, "/workspace/pear")
from pear.agent_loop.protocol import JsonBlockProtocol, InvalidAction  # noqa: E402
from pear.envs.libero.env import LiberoEnv  # noqa: E402

# constants of pear/envs/tool.py (EnvironmentTool), copied: importing it needs verl
SYSTEM_PROMPT_TEMPLATE = Path("/workspace/pear/pear/envs/system_prompt.md").read_text()
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."

THINK_END = "</think>"
IM_END = "<|im_end|>"
JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def user_content(result) -> list[dict]:
    text = JsonBlockProtocol.turn_text(result["state"], REMINDER)
    blocks = [{"type": "text", "text": text}]
    for name, image in zip(LiberoEnv.CAMERAS, result["images"]):
        blocks.append({"type": "text", "text": f"[{name}]"})
        blocks.append({"type": "image", "image": None})
    return blocks


class Template:
    def __init__(self, model_dir: str, effort: str):
        src = Path(model_dir, "chat_template.jinja").read_text()
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        self.t = env.from_string(src)
        self.effort = effort

    def first(self, system: str, user: list[dict]) -> str:
        return self.t.render(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                             add_generation_prompt=True, reasoning_effort=self.effort)

    def next_user(self, user: list[dict]) -> str:
        # the template's rendering of one user turn + generation prompt (system/assistant parts sliced away)
        full = self.t.render(messages=[{"role": "user", "content": user}], add_generation_prompt=True,
                             reasoning_effort=self.effort)
        start = full.index("<|im_start|>user")
        return full[start:]


def generate(url: str, text: str, images: list[str], params: dict) -> tuple[dict, float]:
    body = {"text": text, "image_data": images or None, "sampling_params": params, "return_logprob": True,
            "logprob_start_len": -1}
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t


def slim_meta(meta: dict) -> dict:
    keep = {k: v for k, v in meta.items() if k not in ("output_token_logprobs", "input_token_logprobs",
                                                         "output_top_logprobs", "input_top_logprobs")}
    lp = meta.get("output_token_logprobs") or []
    keep["output_ids"] = [x[1] for x in lp]
    keep["output_logprobs"] = [None if x[0] is None else round(x[0], 4) for x in lp]
    return keep


def run_episode(args, suite: str, task_id: int, trial_id: int, tmpl: Template, think_end_id: int) -> dict:
    name = f"{suite}_t{task_id}_r{trial_id}"
    out_dir = Path(args.out)
    frame_dir = out_dir / "frames" / name
    frame_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.jsonl"
    if log_path.exists():
        log_path.unlink()

    env = LiberoEnv(suite, task_id, trial_id)
    history: list[list[float]] = []
    t0 = time.perf_counter()
    result = env.rollout(history, None)
    env_s = time.perf_counter() - t0
    system = JsonBlockProtocol.system_prompt(SYSTEM_PROMPT_TEMPLATE,
                                             LiberoEnv.environment_prompt(result), result["instruction"])
    text = tmpl.first(system, user_content(result))
    images: list[str] = []
    budget_left = args.response_budget
    sp = {"temperature": 1.0, "top_p": 0.95, "skip_special_tokens": False, "no_stop_trim": True}
    end_reason, success, rnd = None, False, 0
    ctx_used = 0  # prompt + completion tokens of the last request (the next prompt is this plus one user turn)

    def cap():  # max_new_tokens of a round's first request: the episode budget, within the context window
        return max(1, min(budget_left, args.context_len - ctx_used - args.turn_margin))

    while True:
        for i, im in enumerate(result["images"]):
            im.save(frame_dir / f"round{rnd:03d}_{LiberoEnv.CAMERAS[i]}.png")
            images.append(png_b64(im))
        rec = {"episode": name, "round": rnd, "suite": suite, "task_id": task_id, "trial_id": trial_id,
               "instruction": result["instruction"], "state": result["state"], "env_s_before": round(env_s, 3),
               "prompt_chars": len(text), "n_images": len(images)}
        reqs = []
        if args.split:
            out_a, wall_a = generate(args.url, text, images,
                                     {**sp, "max_new_tokens": cap(), "stop_token_ids": [think_end_id]})
            gen_a = out_a["text"]
            reqs.append({"phase": "reason", "wall_s": round(wall_a, 4), "meta": slim_meta(out_a["meta_info"])})
            used = out_a["meta_info"]["completion_tokens"]
            ctx_used = out_a["meta_info"]["prompt_tokens"] + used
            gen_b, wall_b = "", 0.0
            if THINK_END in gen_a and budget_left - used > 0:
                out_b, wall_b = generate(args.url, text + gen_a, images, {**sp, "max_new_tokens": max(1, min(budget_left - used, args.context_len - ctx_used - 16))})
                gen_b = out_b["text"]
                reqs.append({"phase": "answer", "wall_s": round(wall_b, 4), "meta": slim_meta(out_b["meta_info"])})
                used += out_b["meta_info"]["completion_tokens"]
                ctx_used = out_b["meta_info"]["prompt_tokens"] + out_b["meta_info"]["completion_tokens"]
            gen = gen_a + gen_b
        else:
            out, wall = generate(args.url, text, images, {**sp, "max_new_tokens": cap()})
            gen = out["text"]
            reqs.append({"phase": "all", "wall_s": round(wall, 4), "meta": slim_meta(out["meta_info"])})
            used = out["meta_info"]["completion_tokens"]
            ctx_used = out["meta_info"]["prompt_tokens"] + used
        budget_left -= used
        gen = gen.removesuffix(IM_END)
        rec.update(generated=gen, requests=reqs, completion_tokens=used)

        answer = gen.rpartition(THINK_END)[2]
        blocks = JSON_BLOCK.findall(answer)
        action = None
        try:
            if not blocks:
                raise InvalidAction("no ```json block")
            action = json.loads(blocks[-1])
            JsonBlockProtocol.check_action(action, LiberoEnv.SCHEMA)
        except (InvalidAction, json.JSONDecodeError) as exc:
            rec.update(action=action, parse_error=str(exc))
            end_reason = "parse"
        if end_reason is None:
            t0 = time.perf_counter()
            result = env.rollout(history, action)
            env_s = time.perf_counter() - t0
            history.extend(result["actions"])
            success = bool(result["done"]) and sum(float(r) for r in result["rewards"]) > 0
            rec.update(action=action, outcome=result["outcome"], executed=result["executed"], done=result["done"],
                       success=success, env_s=round(env_s, 3), steps_total=len(history))
            if result["done"] or result["images"] is None:
                end_reason = "env"
            elif budget_left <= 0 or ctx_used >= args.context_len - args.turn_margin:
                end_reason = "budget"
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{name}] round {rnd}: {used} tok, {sum(r['wall_s'] for r in reqs):.1f}s, "
              f"verify={[r['meta'].get('spec_verify_ct') for r in reqs]} outcome={rec.get('outcome')} "
              f"{rec.get('parse_error', '')}", flush=True)
        if end_reason is not None or rnd + 1 >= args.max_rounds:
            end_reason = end_reason or "max_rounds"
            break
        text = text + gen + IM_END + "\n" + tmpl.next_user(user_content(result))
        rnd += 1

    summary = {"episode": name, "rounds": rnd + 1, "end_reason": end_reason, "success": success,
               "steps": len(history), "tokens": args.response_budget - budget_left}
    with open(out_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model-dir", default="/workspace/models/pear")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", required=True, help="comma list suite:task:trial")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=1000)
    ap.add_argument("--response-budget", type=int, default=256000)
    ap.add_argument("--context-len", type=int, default=262144)
    ap.add_argument("--turn-margin", type=int, default=4096)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tok = json.loads(Path(args.model_dir, "tokenizer.json").read_text())
    think_end_id = next(t["id"] for t in tok["added_tokens"] if t["content"] == THINK_END)
    tmpl = Template(args.model_dir, args.effort)
    with open(Path(args.out) / "config.json", "w") as f:
        json.dump({**vars(args), "think_end_id": think_end_id, "time": time.time()}, f, indent=1)
    for spec in args.episodes.split(","):
        suite, task, trial = spec.split(":")
        try:
            print(run_episode(args, suite, int(task), int(trial), tmpl, think_end_id), flush=True)
        except Exception as exc:  # keep going; the error is recorded
            import traceback
            traceback.print_exc()
            with open(Path(args.out) / "episodes.jsonl", "a") as f:
                f.write(json.dumps({"episode": f"{suite}_t{task}_r{trial}", "error": repr(exc)}) + "\n")


if __name__ == "__main__":
    main()
