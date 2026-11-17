"""Teacher-forced scoring of round r's recorded tokens under compressed histories (same rounds as strip_A.jsonl):
  S2 card : old thinking dropped + oracle calibration card in the current user turn
  S3 geo  : old thinking reduced to its geometry paragraphs
  S4 summ : old thinking replaced by PEAR's own summary at the S3 token budget
  S5 geo+card"""
import json, sys
from pathlib import Path
from replay_bench import context
from run_episodes import IM_END, Template
from score_strip import score
from common import split_gen, geo_only

RES = Path("/workspace/results/dflash2_main"); P = Path("/workspace/results/pilot")
tmpl = Template("/workspace/models/pear", "xhigh")
cards = {(c["ep"], c["round"]): c for c in map(json.loads, (P / "cards.jsonl").open())}
summ = {(s["episode"], s["round"]): s["summary"] for s in map(json.loads, (P / "summaries.jsonl").open())}
T = json.loads((P / "card_transform.json").read_text())  # robosuite (row,col) -> displayed image (down,right)


def desc(v):
    d, rgt = T["m"][0][0] * v[0] + T["m"][0][1] * v[1], T["m"][1][0] * v[0] + T["m"][1][1] * v[1]
    parts = [f"{abs(d):.1f} px {'down' if d > 0 else 'up'}" if abs(d) >= 0.5 else None,
             f"{abs(rgt):.1f} px {'right' if rgt > 0 else 'left'}" if abs(rgt) >= 0.5 else None]
    return ", ".join(p for p in parts if p) or "less than 0.5 px"


def card_text(c):
    a = c["dA"]; w = c["dW"]  # already per +1 state unit
    return ("[Simulator calibration for the current pose, exact] Agentview image: moving the arm +1 along x moves the "
            f"gripper {desc(a['x'])}; +1 along y: {desc(a['y'])}; +1 along z: {desc(a['z'])}. Wrist image: objects at table "
            f"height are {c['wrist_depth_cm']:.0f} cm below the camera; moving the arm +1 along x moves them {desc(w['x'])}; "
            f"+1 along y: {desc(w['y'])}.\n\n")


def build(rows, r, mode, ep):
    rr = []
    for i, x in enumerate(rows):
        if i >= r:
            rr.append(x); continue
        th, ans = split_gen(x["generated"])
        if mode in ("card", "strip"):
            rr.append(dict(x, generated="\u0000S\u0000" + ans.lstrip("\n")))
        elif mode in ("geo", "geo+card"):
            rr.append(dict(x, generated=geo_only(th) + "\n</think>" + ans))
        elif mode == "summ":
            rr.append(dict(x, generated=summ.get((ep, i), "") + "\n</think>" + ans))
    text, images = context(rr, r, RES / "frames" / ep, tmpl)
    text = text.replace("<think>\n\u0000S\u0000", "")
    if mode in ("card", "geo+card"):
        i = text.rindex("<|im_start|>user\n") + len("<|im_start|>user\n")
        text = text[:i] + card_text(cards[(ep, r)]) + text[i:]
    return text + rows[r]["generated"] + IM_END, images


def main():
    out = P / "ctx_scores.jsonl"; out.unlink(missing_ok=True)
    for s in map(json.loads, (P / "strip_A.jsonl").open()):
        ep, r = s["episode"], s["round"]
        rows = [x for x in map(json.loads, (RES / f"{ep}.jsonl").open()) if x.get("generated")]
        rec = {"episode": ep, "round": r}
        for mode in ("card", "geo", "summ", "geo+card"):
            if any("</think>" not in x["generated"] for x in rows[:r]) or (ep, r) not in cards:
                rec["skip"] = True; break
            text, images = build(rows, r, mode, ep)
            toks, n = score("http://127.0.0.1:30000", text, images)
            rec[mode] = {"prompt_tokens": n, "tokens": toks}
        if r == 1 and ep == s["episode"]:
            rec["card_example"] = card_text(cards[(ep, r)]) if (ep, r) in cards else None
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(ep, r, {m: rec[m]["prompt_tokens"] for m in ("card", "geo", "summ", "geo+card") if m in rec}, flush=True)
    print("CTX_DONE")


if __name__ == "__main__":
    main()
