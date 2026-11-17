"""Layer-by-layer execution of several model variants ("arms") over stored hidden-state streams.

Every arm keeps its own float16/bfloat16 stream of hidden states for its frames on disk. For each
decoder layer the FP weights are fetched once; each arm then builds its own version of the layer
(FP, RTN, or GPTQ calibrated on that arm's own calibration frames, true-sequential exactly as
`dqat.gptq.quantize_block`) and advances its frames by one layer.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dqat.gptq import GPTQConfig, HessianAccumulator, group_grid
from dqat.offline.model import LINEARS, run_layer

GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("self_attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)
TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class StopForward(Exception):
    pass


@dataclass
class Arm:
    name: str
    method: str  # "fp" | "rtn" | "gptq"
    frames: list[int]  # frame ids carried by this arm (eval + calibration)
    calib: list[int] = field(default_factory=list)  # subset of frames used for GPTQ statistics
    dtype: str = "float16"
    layer_error: bool = False  # record per-linear relative output error on action tokens


class Streams:
    """One file per arm: frames stored contiguously, 2-byte values, plus per-frame layer progress."""

    def __init__(self, root: Path, arm: Arm, lengths: dict[int, int], inputs: np.memmap, offsets: dict[int, int]):
        self.arm = arm
        self.lengths = lengths
        self.local = {}
        pos = 0
        for fid in arm.frames:
            self.local[fid] = pos
            pos += lengths[fid]
        path = root / f"{arm.name}.stream"
        progress = root / f"{arm.name}.progress.npy"
        fresh = not path.exists()
        self.data = np.memmap(path, dtype=np.uint16, mode="w+" if fresh else "r+", shape=(pos, inputs.shape[1]))
        if fresh:
            for fid in arm.frames:
                h = torch.from_numpy(np.asarray(inputs[offsets[fid] : offsets[fid] + lengths[fid]]))
                self.data[self.local[fid] : self.local[fid] + lengths[fid]] = _to_bits(h, arm.dtype)
            self.data.flush()
            np.save(progress, np.zeros(len(arm.frames), dtype=np.int64))
        self.progress_path = progress
        self.done = np.load(progress)

    def read(self, fids: list[int], device) -> torch.Tensor:
        rows = [np.asarray(self.data[self.local[f] : self.local[f] + self.lengths[f]]) for f in fids]
        bits = torch.from_numpy(np.stack(rows).view(np.int16))
        return bits.view(TORCH_DTYPES[self.arm.dtype]).to(device)

    def write(self, fids: list[int], h: torch.Tensor, layer: int) -> None:
        if not torch.isfinite(h).all():
            raise RuntimeError(f"{self.arm.name}: nonfinite hidden states at layer {layer}")
        bits = _to_bits(h.cpu(), self.arm.dtype)
        for i, f in enumerate(fids):
            self.data[self.local[f] : self.local[f] + self.lengths[f]] = bits[i]
        self.data.flush()
        for f in fids:
            self.done[self.arm.frames.index(f)] = layer + 1
        np.save(self.progress_path, self.done)

    def layer_of(self, fid: int) -> int:
        return int(self.done[self.arm.frames.index(fid)])


def _to_bits(h: torch.Tensor, dtype: str) -> np.ndarray:
    return h.to(TORCH_DTYPES[dtype]).contiguous().view(torch.int16).numpy().view(np.uint16)


def batches(fids: list[int], lengths: dict[int, int], size: int) -> list[list[int]]:
    by_len: dict[int, list[int]] = {}
    for f in fids:
        by_len.setdefault(lengths[f], []).append(f)
    out = []
    for group in by_len.values():
        out += [group[i : i + size] for i in range(0, len(group), size)]
    return out


def rtn_weight(weight: torch.Tensor, config: GPTQConfig) -> torch.Tensor:
    """GPTQ's grid (asymmetric per-row groups) with plain rounding, i.e. no error compensation."""
    w = weight.float()
    out = torch.empty_like(w)
    q = 2**config.bits - 1
    for start in range(0, w.shape[1], config.group_size):
        block = w[:, start : start + config.group_size]
        scale, zero = group_grid(block, config.bits)
        codes = (torch.round(block / scale[:, None]) + zero[:, None]).clamp(0, q)
        out[:, start : start + config.group_size] = scale[:, None] * (codes - zero[:, None])
    return out


def _solve(weight: torch.Tensor, hessian: torch.Tensor, threads: int) -> torch.Tensor:
    """Worker: the baseline's GPTQ (dqat.gptq.quantize_linear) on CPU in float32."""
    from dqat.gptq import quantize_linear

    torch.set_num_threads(threads)
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data = weight.float().clone()
    quantize_linear(linear, hessian, GPTQConfig())
    return linear.weight.data.clone()


def linear_of(layer, name):
    module = layer
    for part in name.split("."):
        module = getattr(module, part)
    return module


@torch.no_grad()
def gptq_layers(layers: dict, streams: dict, arms: dict, layer_index: int, weights: dict, pool, lengths,
                device, batch: int, threads: int, filler, log) -> None:
    """True-sequential GPTQ (QKV -> O -> up/gate -> down, as `dqat.gptq.quantize_block`) for several arms.

    Inputs of each group are recomputed from the stored layer input through the partially quantized
    layer, stopping at the group's first linear. CPU solves run in `pool`; while they run, `filler()`
    advances other arms on the GPU (returns False when it has nothing left).
    """
    for group in GROUPS:
        jobs = {}
        for name, layer in layers.items():
            acc = HessianAccumulator(linear_of(layer, group[0]).in_features, device)

            def hook(module, args, acc=acc):
                acc.add(args[0])
                raise StopForward

            handle = linear_of(layer, group[0]).register_forward_pre_hook(hook)
            try:
                for fids in batches(arms[name].calib, lengths, batch):
                    if any(streams[name].layer_of(f) != layer_index for f in fids):
                        raise RuntimeError(f"{name}: calibration frames are not at layer {layer_index}")
                    try:
                        run_layer(layer, streams[name].read(fids, device))
                    except StopForward:
                        pass
            finally:
                handle.remove()
            if acc.observations != len(arms[name].calib):
                raise RuntimeError(f"{name}: Hessian saw {acc.observations} frames")
            hessian = acc.matrix.cpu()
            del acc
            for linear in group:
                jobs[(name, linear)] = pool.submit(_solve, weights[linear], hessian, threads)
        pending = set(jobs.values())
        while pending:
            pending = {j for j in pending if not j.done()}
            if pending and not filler():
                wait(pending, return_when=FIRST_COMPLETED)
        for (name, linear), job in jobs.items():
            target = linear_of(layers[name], linear)
            target.weight.data.copy_(job.result().to(target.weight.device, target.weight.dtype))
        log(f"layer {layer_index}: GPTQ {group[0].split('.')[-1]} group done ({len(layers)} arms)")


class LayerErrorRecorder:
    """Relative output error of each linear vs the same linear with FP weights, on the same input.

    Measured per frame on the action tokens ("act") and on every 16th token ("all").
    """

    def __init__(self, layer, fp_layer):
        self.handles, self.values, self.batch = [], {name: {} for name in LINEARS}, None
        for name in LINEARS:
            fp = linear_of(fp_layer, name)

            def hook(module, args, output, name=name, fp=fp):
                x = args[0]
                for i, (fid, start) in enumerate(self.batch):
                    idx = {"act": slice(start, start + 56), "all": slice(0, x.shape[1], 16)}
                    errs = []
                    for sl in idx.values():
                        ref = torch.nn.functional.linear(x[i, sl].to(fp.weight.dtype), fp.weight).float()
                        diff = output[i, sl].float() - ref
                        errs.append(float(diff.pow(2).sum() / ref.pow(2).sum().clamp_min(1e-12)))
                    self.values[name][fid] = errs

            self.handles.append(linear_of(layer, name).register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def make_pool(workers: int) -> ProcessPoolExecutor:
    import multiprocessing as mp

    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)
