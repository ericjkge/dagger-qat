"""Decode latency of k parallel chains vs one, same prompt (a recorded round's context), on an otherwise idle server.
Reports seconds per decoded token on the critical path for k = 1, 2, 4."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import Template  # noqa: E402


def gen(url, text, images, n_tok):
    body = {"text": text, "image_data": images, "sampling_params": {"max_new_tokens": n_tok, "temperature": 1.0,
                                                                     "ignore_eos": True}}
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        o = json.loads(r.read())
    return time.time() - t, o["meta_info"]


def main():
    url = "http://127.0.0.1:30000"; res = Path("/workspace/results/dflash2_main")
    tmpl = Template("/workspace/models/pear", "xhigh")
    out = []
    for ep, r in (("libero_spatial_t0_r0", 3), ("libero_10_t0_r0", 10)):
        rows = [json.loads(l) for l in (res / f"{ep}.jsonl").open()]
        text, images = context(rows, r, res / "frames" / ep, tmpl)
        gen(url, text, images, 8)  # warm the prefix cache
        for k in (1, 2, 4, 1):
            with ThreadPoolExecutor(k) as ex:
                rs = list(ex.map(lambda _: gen(url, text, images, 1024), range(k)))
            wall = max(x[0] for x in rs)
            out.append({"ep": ep, "round": r, "k": k, "wall_s": round(wall, 2), "s_per_tok": round(wall / 1024, 5)})
            print(out[-1], flush=True)
    Path("/workspace/results/batch_latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
