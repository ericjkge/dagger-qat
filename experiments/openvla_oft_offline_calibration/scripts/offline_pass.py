"""Run all offline arms through OpenVLA-OFT's 32 decoder layers, streaming the weights once.

Arms (all start from the identical layer-0 inputs written by offline_prepare.py):
  ref         FP16 weights, eval frames                         (reference for every comparison)
  bf16        BF16 weights, eval frames of 10 states            (numerical floor: FP16 vs BF16)
  rtn         W4 g128 round-to-nearest, eval frames             (no calibration data)
  gptq_clean  W4 g128 GPTQ on 32 clean calibration frames       (the baseline's recipe)
  gptq_pert   W4 g128 GPTQ on the same 32 states, perturbed     (LIBERO-Plus calibration variants)
  gptq_oracle W4 g128 GPTQ on 32 perturbed eval frames of 8 eval states (test-distribution oracle)

Outputs in --run: <arm>.stream files, metrics.pkl (per-layer drift and per-linear errors), and
actions.npz (normalized and un-normalized action chunks per arm and frame).
"""

import argparse
import json
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import torch

from dqat.gptq import GPTQConfig
from dqat.offline.model import LINEARS, NUM_LAYERS, REPO, REVISION, build_layer, layer_names, upstream
from dqat.offline.remote import RemoteSafetensors
from dqat.offline.stream import (
    Arm, LayerErrorRecorder, Streams, batches, gptq_layers, linear_of, make_pool, rtn_weight, run_layer,
)

DEVICE = "mps"
GPTQ_ARMS = ("gptq_clean", "gptq_pert", "gptq_oracle")


def define_arms(frames: list[dict], seed: int = 0) -> tuple[list[Arm], dict]:
    rng = np.random.default_rng(seed)
    ev = [f for f in frames if f["role"] == "eval"]
    eval_ids = [f["id"] for f in ev]
    calib = [f for f in frames if f["role"] == "calib"]
    calib_clean = [f["id"] for f in calib if f["condition"] == "clean"]
    calib_pert = [f["id"] for f in calib if f["condition"] != "clean"]
    states_by_task = {}
    for f in ev:
        states_by_task.setdefault(f["task_id"], [])
        if f["state"] not in states_by_task[f["task_id"]]:
            states_by_task[f["task_id"]].append(f["state"])
    tasks = sorted(states_by_task)
    floor_states = [states_by_task[t][0] for t in tasks]  # one state per task
    oracle_tasks = sorted(rng.choice(tasks, 8, replace=False).tolist())
    oracle_states = [states_by_task[t][int(rng.integers(len(states_by_task[t])))] for t in oracle_tasks]
    perts = ["camera", "light", "background", "noise", "language"]
    oracle_calib = []
    for k, s in enumerate(oracle_states):
        dropped = perts[k % 5]  # 4 of 5 perturbations per state -> 32 frames, conditions balanced
        oracle_calib += [f["id"] for f in ev if f["state"] == s and f["condition"] not in ("clean", dropped)]
    arms = [
        Arm("ref", "fp", eval_ids),
        Arm("bf16", "fp", [f["id"] for f in ev if f["state"] in floor_states], dtype="bfloat16"),
        Arm("rtn", "rtn", eval_ids, layer_error=True),
        Arm("gptq_clean", "gptq", eval_ids + calib_clean, calib=calib_clean, layer_error=True),
        Arm("gptq_pert", "gptq", eval_ids + calib_pert, calib=calib_pert, layer_error=True),
        Arm("gptq_oracle", "gptq", eval_ids, calib=oracle_calib, layer_error=True),
    ]
    for a in arms:
        if a.method == "gptq" and len(a.calib) != 32:
            raise RuntimeError(f"{a.name}: {len(a.calib)} calibration frames")
    return arms, dict(oracle_states=oracle_states, floor_states=floor_states)


class Prefetch:
    def __init__(self, remote: RemoteSafetensors):
        self.remote, self.thread, self.result = remote, None, None

    def start(self, index: int):
        def run():
            self.result = (index, self.remote.get(layer_names(index)))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def take(self, index: int) -> dict:
        self.thread.join()
        got, tensors = self.result
        if got != index:
            raise RuntimeError("prefetch out of order")
        return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/offline/run"))
    parser.add_argument("--ckpt", type=Path, default=Path("data/offline/ckpt"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="streams/results dir (default: --run)")
    parser.add_argument("--smoke", type=int, default=0, help="test: keep 2 eval states, run this many layers")
    args = parser.parse_args()
    out_dir = args.out or args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    num_layers = args.smoke or NUM_LAYERS
    torch.manual_seed(0)
    manifest = json.loads((args.run / "manifest.json").read_text())
    frames = manifest["frames"]
    lengths = {f["id"]: f["length"] for f in frames}
    offsets = {f["id"]: f["offset"] for f in frames}
    starts = {f["id"]: f["action_start"] for f in frames}
    inputs = np.memmap(args.run / "inputs.f16", dtype=np.float16, mode="r", shape=(manifest["total_tokens"], 4096))
    arm_list, info = define_arms(frames)
    if args.smoke:
        keep = {f["state"] for f in frames if f["role"] == "eval"}
        keep = set(sorted(keep)[:2])
        state_of = {f["id"]: f["state"] for f in frames}
        for a in arm_list:
            a.frames = [f for f in a.frames if state_of[f] in keep or f in a.calib]
    (out_dir / "arms.json").write_text(json.dumps(dict(info, arms=[a.__dict__ for a in arm_list]), indent=1))
    arms = {a.name: a for a in arm_list}
    streams = {a.name: Streams(out_dir, a, lengths, inputs, offsets) for a in arm_list}
    metrics_path = out_dir / "metrics.pkl"
    metrics = pickle.loads(metrics_path.read_bytes()) if metrics_path.exists() else dict(drift={}, linear={})
    _, cfg_mod, *_ = upstream()
    text_config = cfg_mod.OpenVLAConfig.from_pretrained(args.ckpt).text_config
    remote = RemoteSafetensors(REPO, REVISION, args.ckpt / "model.safetensors.index.json")
    first = min(int(s.done.min()) for s in streams.values())
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    pool = make_pool(args.workers)
    prefetch = Prefetch(remote)
    prefetch.start(first)
    config = GPTQConfig()
    for i in range(first, num_layers):
        t0 = time.time()
        tensors = prefetch.take(i)
        if i + 1 < num_layers:
            prefetch.start(i + 1)
        prefix = f"language_model.model.layers.{i}."
        weights = {n: tensors[prefix + n + ".weight"] for n in LINEARS}  # checkpoint BF16
        ref_layer = build_layer(text_config, i, tensors, DEVICE, torch.float16)

        def advance(name, layer, recorder=None):
            todo = [f for f in arms[name].frames if streams[name].layer_of(f) == i]
            for fids in batches(todo, lengths, args.batch):
                if recorder:
                    recorder.batch = [(f, starts[f]) for f in fids]
                h = run_layer(layer, streams[name].read(fids, DEVICE))
                streams[name].write(fids, h, i)
                yield

        def fp_side():
            yield from advance("ref", ref_layer)
            bf_layer = build_layer(text_config, i, tensors, DEVICE, torch.bfloat16)
            yield from advance("bf16", bf_layer)
            del bf_layer
            rtn_layer = build_layer(text_config, i, tensors, DEVICE, torch.float16)
            for n in LINEARS:
                linear_of(rtn_layer, n).weight.data.copy_(rtn_weight(weights[n], config).to(DEVICE, torch.float16))
            rec = LayerErrorRecorder(rtn_layer, ref_layer)
            yield from advance("rtn", rtn_layer, rec)
            rec.remove()
            metrics["linear"][("rtn", i)] = rec.values

        work = fp_side()
        filler = lambda: next(work, StopIteration) is not StopIteration  # noqa: E731

        saved = out_dir / "gptq_layer.pt"
        g_layers = {n: build_layer(text_config, i, tensors, DEVICE, torch.float16) for n in GPTQ_ARMS}
        state = torch.load(saved) if saved.exists() else None
        if state is not None and state["layer"] == i:
            for n, layer in g_layers.items():
                for lin in LINEARS:
                    linear_of(layer, lin).weight.data.copy_(state[n][lin].to(DEVICE))
            log(f"layer {i}: loaded saved GPTQ weights")
        else:
            gptq_layers(g_layers, streams, arms, i, weights, pool, lengths, DEVICE, args.batch, args.threads,
                        filler, log)
            torch.save(dict(layer=i, **{n: {lin: linear_of(l, lin).weight.data.cpu() for lin in LINEARS}
                                        for n, l in g_layers.items()}), saved)
        while filler():
            pass
        for n, layer in g_layers.items():
            rec = LayerErrorRecorder(layer, ref_layer)
            for _ in advance(n, layer, rec):
                pass
            rec.remove()
            metrics["linear"][(n, i)] = rec.values
        del g_layers, ref_layer
        torch.mps.empty_cache()

        # Hidden-state drift vs the reference after this layer (action tokens / every 16th token).
        ref = streams["ref"]
        for name, s in streams.items():
            if name == "ref":
                continue
            out = {}
            for f in arms[name].frames:
                if f not in ref.local:
                    continue
                a = ref.read([f], "cpu")[0].float()
                b = s.read([f], "cpu")[0].float()
                st = starts[f]
                rel = lambda sl: float((b[sl] - a[sl]).norm() / a[sl].norm())  # noqa: E731
                out[f] = (rel(slice(st, st + 56)), rel(slice(0, None, 16)))
            metrics["drift"][(name, i)] = out
        tmp = metrics_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(metrics))
        tmp.replace(metrics_path)
        log(f"layer {i} done in {time.time() - t0:.0f}s; downloaded {remote.bytes_read / 1e9:.1f} GB")
    pool.shutdown()

    from dqat.offline.model import OFTEnds

    ends = OFTEnds(args.ckpt, remote)
    out = {}
    del prefetch
    torch.mps.empty_cache()
    for name, s in streams.items():
        if int(s.done.min()) != num_layers:
            raise RuntimeError(f"{name} did not finish")
        norm, act = [], []
        for f in arms[name].frames:
            n, a = ends.head(s.read([f], DEVICE)[0], starts[f])
            norm.append(n)
            act.append(a)
        out[f"{name}_ids"] = np.array(arms[name].frames)
        out[f"{name}_norm"] = np.stack(norm)
        out[f"{name}_act"] = np.stack(act)
    np.savez(out_dir / "actions.npz", **out)
    log("actions written")


if __name__ == "__main__":
    main()
