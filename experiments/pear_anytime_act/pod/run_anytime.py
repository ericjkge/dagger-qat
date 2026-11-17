"""Closed-loop LIBERO episodes with PEAR where thinking is generated in segments and the provisional action is probed
between segments. --rule decides when to stop thinking:
  full   : never stop early (baseline; probes are still logged so exit rules can be replayed offline)
  exact  : stop when two consecutive greedy provisional actions are identical (Dynasor/DEER-style consistency)
  tol    : stop when two consecutive greedy provisional actions agree within the feedback tolerance (ours)
On exit the action is sampled after the forced "</think>" exactly as PEAR would produce it, and the round is recorded
in the context with that shortened thinking (the model sees its own truncated reasoning in later rounds).
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

sys.path.insert(0, "/workspace")
from run_episodes import (IM_END, JSON_BLOCK, THINK_END, SYSTEM_PROMPT_TEMPLATE, Template, generate, png_b64,  # noqa: E402
                          user_content)
from pear.agent_loop.protocol import InvalidAction, JsonBlockProtocol  # noqa: E402
from pear.envs.libero.env import LiberoEnv  # noqa: E402

FORCE = "\n</think>\n\n```json\n"
ARM = ("x", "y", "z", "roll", "pitch", "yaw")


def parse(txt):
    try:
        return json.loads(txt.split("```")[0].strip())
    except Exception:
        return None


def vec(a):
    try:
        return [float(a["arm"][k]) for k in ARM] + [float(a["gripper"]["width"])]
    except Exception:
        return None


def agree(a, b, rule, tol_abs, tol_rel, tol_rot):
    va, vb = vec(a), vec(b)
    if va is None or vb is None:
        return False
    if rule == "exact":
        return va == vb
    dt = math.dist(va[:3], vb[:3]); mag = max(math.hypot(*va[:3]), math.hypot(*vb[:3]))
    if dt > max(tol_abs, tol_rel * mag):
        return False
    if max(abs(va[i] - vb[i]) for i in (3, 4, 5)) > tol_rot:
        return False
    ga, gb = va[6], vb[6]  # gripper: same command class (open / hold / close)
    cls = lambda g: 0 if abs(g) < 1.0 else (1 if g > 0 else -1)
    return cls(ga) == cls(gb)


def post_abort(url, rid):
    import urllib.request
    try:
        req = urllib.request.Request(url + "/abort_request", data=json.dumps({"rid": rid}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass


def irreversible(act):
    """Gripper command change, or a descent (toward contact) without lateral motion."""
    v = vec(act)
    if v is None:
        return True
    return abs(v[6]) >= 1.0 or (v[2] <= -3.0 and math.hypot(v[0], v[1]) < 3.0)


def consensus_round(a, text, images, sp, cap):
    wait_this_round = random.random() < a.gate_p
    """k parallel full chains; when m have finished, cancel the rest and return the medoid of the m finished actions
    (translation distance; gripper command class must match the majority). Returns (gen, used_tokens, info)."""
    import uuid, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rids = [uuid.uuid4().hex for _ in range(a.k)]
    def one(rid):
        o, _ = generate(a.url, text, images, {**sp, "max_new_tokens": cap}, rid=rid)
        return o
    done = []
    ex = ThreadPoolExecutor(a.k)
    futs = {ex.submit(one, rid): rid for rid in rids}
    for f in as_completed(futs):
        try:
            o = f.result()
        except Exception:
            continue
        g = o["text"].removesuffix(IM_END)
        b = JSON_BLOCK.findall(g.rpartition(THINK_END)[2])
        act = None
        try:
            act = json.loads(b[-1]) if b else None
        except Exception:
            act = None
        done.append({"gen": g, "tok": o["meta_info"]["completion_tokens"], "act": act,
                     "finish": o["meta_info"]["finish_reason"]})
        nv = sum(1 for d in done if vec(d["act"]))
        if nv >= a.m:
            break
        if a.gate == 2 and nv >= 1 and not wait_this_round:
            break  # control: random gate at the same rate
        if a.gate == 1 and nv >= 1 and not irreversible(next(d for d in done if vec(d["act"]))["act"]):
            break  # reversible move proposed by the first finisher: act now
    for f, rid in futs.items():
        if not f.done():
            post_abort(a.url, rid)
    ex.shutdown(wait=True, cancel_futures=True)
    valid = [d for d in done if vec(d["act"])]
    if not valid:
        d = done[0] if done else {"gen": "", "tok": 0}
        return d["gen"], d["tok"], {"m_tok": [x["tok"] for x in done], "picked": 0}
    cls = lambda d: (lambda g: 0 if abs(g) < 1.0 else (1 if g > 0 else -1))(vec(d["act"])[6])
    maj = max(set(cls(d) for d in valid), key=lambda c: sum(cls(d) == c for d in valid))
    cand = [d for d in valid if cls(d) == maj]
    best = min(cand, key=lambda d: sum(math.dist(vec(d["act"])[:3], vec(e["act"])[:3]) for e in valid))
    # latency proxy: decode steps until the m-th finisher; compute: tokens of all chains until then
    m_tok = sorted(x["tok"] for x in done)
    return best["gen"], best["tok"], {"m_tok": m_tok, "latency_tok": m_tok[-1],
                                      "acts": [x["act"] for x in valid], "picked_tok": best["tok"]}


def run_episode(a, suite, task_id, trial_id, tmpl, think_end_id):
    name = f"{suite}_t{task_id}_r{trial_id}"
    out_dir = Path(a.out); log_path = out_dir / f"{name}.jsonl"
    skip = Path(f"/workspace/skip_suites_{a.rule}")
    if skip.exists() and suite in skip.read_text().split():
        print({"episode": name, "skip": "suite delegated", "success": None}, flush=True)
        return None
    done_p = out_dir / "episodes.jsonl"
    if done_p.exists() and any(json.loads(l).get("episode") == name and "error" not in json.loads(l) for l in done_p.open()):
        print({"episode": name, "skip": "already done", "success": None}, flush=True)
        return None
    log_path.unlink(missing_ok=True)
    env = LiberoEnv(suite, task_id, trial_id)
    history = []
    result = env.rollout(history, None)
    system = JsonBlockProtocol.system_prompt(SYSTEM_PROMPT_TEMPLATE, LiberoEnv.environment_prompt(result), result["instruction"])
    text = tmpl.first(system, user_content(result))
    images = []
    budget_left = a.response_budget
    sp = {"temperature": 1.0, "top_p": 0.95, "skip_special_tokens": False, "no_stop_trim": True}
    end_reason, success, rnd, ctx_used = None, False, 0, 0
    t_ep = time.time()
    while True:
        images += [png_b64(im) for im in result["images"]]
        rec = {"episode": name, "round": rnd, "state": result["state"], "probes": []}
        think, used, exited, prev, t0 = "", 0, False, None, time.time()
        probe_tok = 0
        if a.rule == "consensus":
            cap = max(1, min(32000, budget_left, a.context_len - ctx_used - a.turn_margin))
            gen, used, info = consensus_round(a, text, images, sp, cap)
            rec["consensus"] = info
            budget_left -= used
            ctx_used += used
            rec.update(generated=gen, completion_tokens=used, probe_tokens=0, exited=False,
                       wall_s=round(time.time() - t0, 2))
        # --- thinking in segments ---
        while a.rule != "consensus":
            cap = max(1, min(a.gap, budget_left - used, a.context_len - ctx_used - a.turn_margin))
            o, _ = generate(a.url, text + think, images, {**sp, "max_new_tokens": cap, "stop_token_ids": [think_end_id]})
            think += o["text"]; used += o["meta_info"]["completion_tokens"]
            ctx_used = o["meta_info"]["prompt_tokens"] + o["meta_info"]["completion_tokens"]
            if THINK_END in o["text"] or o["meta_info"]["finish_reason"]["type"] == "stop" or used >= budget_left:
                break
            if "\n\n" not in think[-4000:]:
                continue
            # extend to the end of the current paragraph so the probe cuts at a boundary
            if not think.endswith("\n\n"):
                o, _ = generate(a.url, text + think, images, {**sp, "max_new_tokens": 400, "stop": ["\n\n"],
                                                             "stop_token_ids": [think_end_id]})
                think += o["text"]; used += o["meta_info"]["completion_tokens"]
                if THINK_END in o["text"]:
                    break
            if a.rule == "none":
                continue
            pr, _ = generate(a.url, text + think.rstrip("\n") + FORCE, images,
                             {"max_new_tokens": 96, "stop": ["```"], "temperature": 0.0, "skip_special_tokens": False})
            cur = parse(pr["text"]); probe_tok += pr["meta_info"]["completion_tokens"]
            rec["probes"].append({"tok": used, "greedy": cur})
            if a.rule in ("exact", "tol") and prev is not None and cur is not None and \
                    agree(prev, cur, a.rule, a.tol_abs, a.tol_rel, a.tol_rot) and used >= a.min_tok:
                exited = True
                break
            prev = cur
        # --- answer ---
        if a.rule == "consensus":
            pass
        elif exited:
            think = think.rstrip("\n")
            o, _ = generate(a.url, text + think + FORCE, images, {**sp, "max_new_tokens": 128, "stop": ["```"]})
            gen = think + FORCE + o["text"].split("```")[0].rstrip() + "\n```"
            used += o["meta_info"]["completion_tokens"]
        else:
            gen = think
            if THINK_END in think and budget_left - used > 0:
                o, _ = generate(a.url, text + think, images, {**sp, "max_new_tokens": max(1, min(2048, budget_left - used))})
                gen += o["text"]; used += o["meta_info"]["completion_tokens"]
                ctx_used = o["meta_info"]["prompt_tokens"] + o["meta_info"]["completion_tokens"]
        if a.rule != "consensus":
            budget_left -= used
            gen = gen.removesuffix(IM_END)
            rec.update(generated=gen, completion_tokens=used, probe_tokens=probe_tok, exited=exited,
                       wall_s=round(time.time() - t0, 2))
        answer = gen.rpartition(THINK_END)[2]
        blocks = JSON_BLOCK.findall(answer)
        action = None
        try:
            if not blocks:
                raise InvalidAction("no ```json block")
            action = json.loads(blocks[-1])
            JsonBlockProtocol.check_action(action, LiberoEnv.SCHEMA)
        except (InvalidAction, json.JSONDecodeError) as exc:
            rec.update(action=action, parse_error=str(exc)); end_reason = "parse"
        if end_reason is None:
            result = env.rollout(history, action)
            history.extend(result["actions"])
            success = bool(result["done"]) and sum(float(r) for r in result["rewards"]) > 0
            rec.update(action=action, outcome=result["outcome"], executed=result["executed"], done=result["done"],
                       success=success)
            if result["done"] or result["images"] is None:
                end_reason = "env"
            elif budget_left <= 0 or ctx_used >= a.context_len - a.turn_margin:
                end_reason = "budget"
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{name}] r{rnd} {used} tok exited={exited} probes={len(rec['probes'])} {rec.get('outcome')}", flush=True)
        if end_reason is not None or rnd + 1 >= a.max_rounds:
            end_reason = end_reason or "max_rounds"
            break
        text = text + gen + IM_END + "\n" + tmpl.next_user(user_content(result))
        rnd += 1
    summ = {"episode": name, "rule": a.rule, "rounds": rnd + 1, "end_reason": end_reason, "success": success,
            "steps": len(history), "tokens": a.response_budget - budget_left, "wall_s": round(time.time() - t_ep, 1)}
    with open(out_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", required=True, help="suite:task:trial")
    ap.add_argument("--rule", default="full", choices=["none", "full", "exact", "tol", "consensus"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--gate-p", type=float, default=0.37)
    ap.add_argument("--gate", type=int, default=0, help="1: act on the first finisher unless its action is irreversible")
    ap.add_argument("--gap", type=int, default=150)
    ap.add_argument("--min-tok", type=int, default=0)
    ap.add_argument("--tol-abs", type=float, default=1.0)
    ap.add_argument("--tol-rel", type=float, default=0.15)
    ap.add_argument("--tol-rot", type=float, default=0.1)
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--response-budget", type=int, default=160000)
    ap.add_argument("--context-len", type=int, default=262144)
    ap.add_argument("--turn-margin", type=int, default=4096)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tok = json.loads(Path("/workspace/models/pear/tokenizer.json").read_text())
    think_end_id = next(t["id"] for t in tok["added_tokens"] if t["content"] == THINK_END)
    tmpl = Template("/workspace/models/pear", "xhigh")
    suite, task, trial = a.episode.split(":")
    try:
        print(run_episode(a, suite, int(task), int(trial), tmpl, think_end_id), flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc()
        with open(Path(a.out) / "episodes.jsonl", "a") as f:
            f.write(json.dumps({"episode": f"{suite}_t{task}_r{trial}", "rule": a.rule, "error": repr(exc)}) + "\n")


if __name__ == "__main__":
    main()
