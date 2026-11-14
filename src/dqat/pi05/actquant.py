"""Fake activation quantization of π0.5 action-expert linear inputs (weights stay BF16).

Scope: inputs of the 7 projections in each of the 18 Gemma-300M action-expert layers. Inputs
shared by several projections are one *group* and are quantized once, identically:

    L{l}.qkv    input of q_proj/k_proj/v_proj (post-AdaRMS-norm hidden state), width 1024
    L{l}.o      input of o_proj (attention output), width 2048 (8 heads x 256)
    L{l}.gateup input of gate_proj/up_proj (post-AdaRMS-norm hidden state), width 1024
    L{l}.down   input of down_proj (gelu(gate) * up), width 4096

The VLM prefix (SigLIP + Gemma-2B) and its KV cache are never touched. The expert is called once per
flow-matching (denoising) step; `step` counts calls within one `sample_actions` (0 = t=1, pure noise).

Quantizer: dynamic per-token asymmetric min-max integer grid with `bits` bits (default 4):
    s = (max - min) / (2^b - 1),  z = round(-min / s),  q = clamp(round(h / s) + z, 0, 2^b - 1),
    ĥ = (q - z) * s
computed in float32, cast back to the activation dtype. A `rounder` callback may replace the
nearest-rounding codes with any codes ON THE SAME GRID (same s, z, bit width) for Step-4 experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

LINEAR_TO_GROUP = {
    "self_attn.q_proj": "qkv",
    "self_attn.k_proj": "qkv",
    "self_attn.v_proj": "qkv",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gateup",
    "mlp.up_proj": "gateup",
    "mlp.down_proj": "down",
}
GROUP_KINDS = ("qkv", "o", "gateup", "down")


def grid_params(h: torch.Tensor, bits: int):
    """Per-token (last-dim) asymmetric grid. Returns float32 scale and zero point broadcastable to h."""
    x = h.float()
    lo = x.amin(dim=-1, keepdim=True).clamp(max=0.0)
    hi = x.amax(dim=-1, keepdim=True).clamp(min=0.0)
    qmax = 2**bits - 1
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    zero = torch.round(-lo / scale)
    return scale, zero, qmax


def nearest_codes(h: torch.Tensor, scale, zero, qmax):
    return torch.clamp(torch.round(h.float() / scale) + zero, 0, qmax)


def dequant(codes, scale, zero):
    return (codes - zero) * scale


_SIGNS: dict = {}


def hadamard(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Orthonormal randomized Walsh-Hadamard transform over the last dim (power of 2): R = H·diag(s)/sqrt(n).

    QuaRot-style online rotation: W h = (W Rᵀ)(R h), so a deployment folds Rᵀ into the weight and applies
    R to the activation (O(n log n) per token). Fixed random signs per width (seed 0)."""
    n = x.shape[-1]
    if n not in _SIGNS:
        g = torch.Generator().manual_seed(0)
        _SIGNS[n] = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    signs = _SIGNS[n].to(x.device)
    y = x.float()
    if not inverse:
        y = y * signs
    shape = y.shape
    y = y.reshape(-1, n)
    k = 1
    while k < n:
        y = y.view(-1, n // (2 * k), 2, k)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        k *= 2
    y = y.reshape(shape) / n**0.5
    if inverse:
        y = y * signs
    return y


def fake_quant(h: torch.Tensor, bits: int = 4) -> torch.Tensor:
    scale, zero, qmax = grid_params(h, bits)
    return dequant(nearest_codes(h, scale, zero, qmax), scale, zero).to(h.dtype)


@dataclass
class QuantConfig:
    bits: int = 4
    groups: set[str] | None = None  # None = all groups; else e.g. {"L3.down", "L10.qkv"}
    steps: set[int] | None = None  # None = all denoising steps
    kinds: set[str] | None = None  # restrict to group kinds, e.g. {"down"}
    rotate: set[str] | None = None  # group kinds quantized in a randomized-Hadamard basis (QuaRot-style)

    def active(self, group: str, step: int) -> bool:
        if self.steps is not None and step not in self.steps:
            return False
        if self.kinds is not None and group.split(".")[1] not in self.kinds:
            return False
        return self.groups is None or group in self.groups


@dataclass
class ActivationController:
    """Installs forward pre-hooks on the action-expert linears. One instance per model.

    Modes (set before each `sample_actions` call; `begin_call` resets per-call state):
      quant      : QuantConfig or None (None = BF16 reference)
      rounder    : optional fn(group, step, h_fp32, scale, zero, qmax, nearest) -> codes
      probe      : optional "all" or set of (group, step); a zero leaf (requires grad) is ADDED to that
                   input, so leaf.grad is the gradient wrt the input; stored in `leaves`
      record     : optional set of (group, step) whose FP input and quantized input are stored
      inject     : optional {(group, step): delta} ADDED to the input (float32 delta)
    """

    model: torch.nn.Module
    quant: QuantConfig | None = None
    rounder: object = None
    probe: tuple[str, int] | None = None
    record: set | None = None
    inject: dict | None = None
    step: int = 0
    stats: dict = field(default_factory=dict)
    leaves: dict = field(default_factory=dict)
    recorded: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)
    _handles: list = field(default_factory=list)

    def install(self):
        expert = self.model.paligemma_with_expert.gemma_expert.model
        for layer_idx, layer in enumerate(expert.layers):
            for name, group_kind in LINEAR_TO_GROUP.items():
                module = layer.get_submodule(name)
                group = f"L{layer_idx}.{group_kind}"
                self._handles.append(module.register_forward_pre_hook(self._make_hook(group)))
        # Count expert calls: wrap denoise_step (an instance attribute shadows the bound method;
        # sample_actions calls self.denoise_step, so this sees every flow step).
        original = self.model.denoise_step

        def counted(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                self.step += 1
                self._cache.clear()

        self.model.denoise_step = counted
        self._original_denoise = original
        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.model.denoise_step = self._original_denoise

    def begin_call(self):
        self.step = 0
        self.stats = {}
        self.leaves = {}
        self.recorded = {}
        self._cache = {}

    def _transform(self, group: str, h: torch.Tensor) -> torch.Tensor:
        key = (group, self.step)
        out = h
        if self.inject is not None and key in self.inject:
            # additive perturbation (first-order directional probes compose correctly across groups)
            wide = torch.promote_types(h.dtype, torch.float32)
            out = (h.to(wide) + self.inject[key].to(device=h.device, dtype=wide)).to(h.dtype)
        elif self.quant is not None and self.quant.active(group, self.step):
            rot = self.quant.rotate is not None and group.split(".")[1] in self.quant.rotate
            x = hadamard(h) if rot else h
            scale, zero, qmax = grid_params(x, self.quant.bits)
            codes = nearest_codes(x, scale, zero, qmax)
            if self.rounder is not None:
                codes = self.rounder(group, self.step, x.float(), scale, zero, qmax, codes)
            xq = dequant(codes, scale, zero)
            out = (hadamard(xq, inverse=True) if rot else xq).to(h.dtype)
            # accumulate on device (no host sync per layer); converted in activation_mse/group_stats
            err = (out.float() - h.float()).pow(2).sum()
            s = self.stats.setdefault(group, [0.0, 0.0, 0])
            s[0] = s[0] + err
            s[1] = s[1] + h.float().pow(2).sum()
            s[2] += h.numel()
        if self.record is not None and (self.record == "all" or key in self.record):
            self.recorded[key] = (h.detach().float(), out.detach().float())  # kept on device
        if self.probe is not None and (self.probe == "all" or key in self.probe):
            # Additive zero leaf: d(output)/d(leaf) == d(output)/d(input) with the upstream graph intact,
            # so one backward pass yields M for every (group, step) at once.
            wide = torch.promote_types(h.dtype, torch.float32)
            leaf = torch.zeros_like(out, dtype=wide, requires_grad=True)
            self.leaves[key] = leaf
            out = (out.to(wide) + leaf).to(h.dtype)
        return out

    def _make_hook(self, group: str):
        def hook(module, args):
            (h,) = args[:1]
            key = (group, self.step)
            # q/k/v (and gate/up) share their input: transform once, reuse the identical tensor.
            cached = self._cache.get(key)
            if cached is not None and cached[0] is h:
                return (cached[1],) + tuple(args[1:])
            out = self._transform(group, h)
            self._cache[key] = (h, out)
            return (out,) + tuple(args[1:])

        return hook

    def activation_mse(self):
        """Aggregate over all quantized groups/steps of the last call: (MSE, relative MSE)."""
        if not self.stats:
            return 0.0, 0.0
        self.stats = {g: [float(v[0]), float(v[1]), v[2]] for g, v in self.stats.items()}
        se = sum(v[0] for v in self.stats.values())
        ss = sum(v[1] for v in self.stats.values())
        n = sum(v[2] for v in self.stats.values())
        return se / n, se / max(ss, 1e-30)
