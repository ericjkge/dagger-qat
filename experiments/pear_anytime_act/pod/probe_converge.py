"""E1: does PEAR's provisional action converge long before its thinking ends?

For every recorded round (14 LIBERO episodes, 171 rounds), cut the recorded thinking at paragraph boundaries spaced
>= --gap tokens apart (at most --max-cuts per round, plus the full thinking), force the thinking to end there
("\n</think>\n\n```json\n") and decode the provisional action: one greedy decode and --samples samples at PEAR's
sampling settings (T=1.0, top_p=0.95). Output: one JSON line per (round, cut) with token position and parsed actions.
"""
import argparse, json, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import Template  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

FORCE = "\n</think>\n\n```json\n"


def post(url, body):
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def parse(txt):
    s = txt.split("```")[0].strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def cuts_for(think, tok, gap, max_cuts):
    ids = tok.encode(think, add_special_tokens=False).ids
    n = len(ids)
    # paragraph boundaries in characters -> token positions (prefix token count)
    pos, last, out = 0, -10**9, []
    paras = think.split("\n\n")
    for p in paras[:-1]:
        pos += len(p) + 2
        t = len(tok.encode(think[:pos], add_special_tokens=False).ids)
        if t - last >= gap:
            out.append((pos, t)); last = t
    if len(out) > max_cuts:
        idx = [round(i * (len(out) - 1) / (max_cuts - 1)) for i in range(max_cuts)]
        out = [out[i] for i in sorted(set(idx))]
    out.append((len(think), n))  # full thinking
    return out, n


def probe_round(a, tmpl, tok, ep, rows, r, lock, fout):
    res = Path(a.results)
    text, images = context(rows, r, res / "frames" / ep, tmpl)
    gen = rows[r]["generated"]
    if "</think>" not in gen:
        return
    think = gen.partition("</think>")[0]
    cuts, n = cuts_for(think, tok, a.gap, a.max_cuts)
    for ci, (cpos, ctok) in enumerate(cuts):
        prompt = text + think[:cpos].rstrip("\n") + FORCE
        outs = []
        for s in range(1 + a.samples):
            sp = {"max_new_tokens": 96, "stop": ["```"], "temperature": 0.0 if s == 0 else 1.0, "top_p": 1.0 if s == 0 else 0.95,
                  "skip_special_tokens": False}
            o = post(a.url, {"text": prompt, "image_data": images, "sampling_params": sp})
            outs.append(o["text"])
        rec = {"ep": ep, "round": r, "cut": ci, "ncuts": len(cuts), "tok": ctok, "think_tok": n,
               "final": ci == len(cuts) - 1, "recorded": rows[r].get("action"),
               "greedy": parse(outs[0]), "samples": [parse(x) for x in outs[1:]], "raw0": outs[0][:200],
               "prompt_tokens": o["meta_info"]["prompt_tokens"]}
        with lock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
    print(f"{ep} r{r} cuts={len(cuts)} think={n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--results", default="/workspace/results/dflash2_main")
    ap.add_argument("--out", default="/workspace/results/probe/probe.jsonl")
    ap.add_argument("--gap", type=int, default=150)
    ap.add_argument("--max-cuts", type=int, default=30)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    tmpl = Template("/workspace/models/pear", "xhigh")
    tok = Tokenizer.from_file("/workspace/models/pear/tokenizer.json")
    res = Path(a.results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    jobs = []
    for ep in sorted(p.stem for p in res.glob("libero_*.jsonl")):
        rows = [r for r in map(json.loads, (res / f"{ep}.jsonl").open()) if r.get("generated")]
        for r in range(len(rows)):
            jobs.append((ep, rows, r))
    # longest rounds first so the tail is short
    jobs.sort(key=lambda j: -len(j[1][j[2]]["generated"]))
    with open(a.out, "w") as fout, ThreadPoolExecutor(a.threads) as ex:
        for f in [ex.submit(probe_round, a, tmpl, tok, ep, rows, r, lock, fout) for ep, rows, r in jobs]:
            try:
                f.result()
            except Exception as e:  # keep going
                print("ERR", repr(e), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
