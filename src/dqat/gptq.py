"""Llama-only GPTQ with sequential Hessian error compensation and BF16 QDQ.

Adapted from IST-DASLab/gptq at 2d65066eeb06a5c9ff5184d8cebdf33662c67faf
(gptq.py, quant.py, llama.py), under Apache-2.0; see licenses/gptq.txt.
Changes: variable-length multimodal capture, explicit validation, shared input
statistics for Q/K/V and up/gate, and portable packed calibration artifacts.
The model still runs BF16 kernels. Artifact packing is not an inference speedup.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from dqat.awq import _output, _to, llama_linears

REFERENCE_REVISION = "2d65066eeb06a5c9ff5184d8cebdf33662c67faf"


@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damp_percent: float = 0.01

    def __post_init__(self):
        if self.bits != 4:
            raise ValueError("This GPTQ artifact format supports four-bit weights")
        if self.group_size < 2 or self.group_size % 2 or self.block_size != self.group_size:
            raise ValueError("GPTQ requires even groups and one group per compensation block")
        if not math.isfinite(self.damp_percent) or self.damp_percent <= 0:
            raise ValueError("GPTQ requires positive finite Hessian damping")


@contextmanager
def full_precision_statistics():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class HessianAccumulator:
    """Reference normalization: H = 2 / N_observations * sum X^T X."""

    def __init__(self, width, device):
        self.matrix = torch.zeros(width, width, dtype=torch.float32, device=device)
        self.observations = 0
        self.tokens = 0

    @torch.no_grad()
    def add(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.matrix.shape[0] or inputs.numel() == 0:
            raise ValueError("Expected nonempty [observations, tokens, channels] calibration inputs")
        if not torch.isfinite(inputs).all():
            raise ValueError("Nonfinite GPTQ calibration inputs")
        count = inputs.shape[0]
        self.matrix.mul_(self.observations / (self.observations + count))
        self.observations += count
        x = inputs.detach().reshape(-1, inputs.shape[-1]).t().float()
        self.tokens += x.shape[1]
        x = math.sqrt(2 / self.observations) * x
        self.matrix.add_(x.matmul(x.t()))


def group_grid(weights, bits=4):
    """Reference asymmetric per-row min/max grid, including zero in the range."""
    minimum = torch.minimum(weights.amin(1), torch.zeros_like(weights[:, 0]))
    maximum = torch.maximum(weights.amax(1), torch.zeros_like(weights[:, 0]))
    empty = (minimum == 0) & (maximum == 0)
    minimum[empty], maximum[empty] = -1, 1
    scale = (maximum - minimum) / (2**bits - 1)
    zero = torch.round(-minimum / scale)
    return scale, zero


def dequantize(artifact, config, device, dtype):
    rows, columns = artifact["shape"]
    if columns % config.group_size:
        raise ValueError("GPTQ artifact width is incompatible with the group size")
    packed, scales, zeros = (artifact[name] for name in ("packed_codes", "scales", "zeros"))
    if (
        packed.shape != (rows, columns // 2)
        or packed.dtype != torch.uint8
        or scales.shape != (rows, columns // config.group_size)
        or zeros.shape != scales.shape
        or zeros.dtype != torch.uint8
        or not torch.isfinite(scales).all()
        or not torch.all(scales > 0)
        or torch.any(zeros > 15)
    ):
        raise ValueError("Malformed GPTQ artifact grid or packed weights")
    packed = packed.to(device)
    codes = torch.empty(rows, columns, dtype=torch.uint8, device=device)
    codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
    grouped = codes.reshape(rows, -1, config.group_size).float()
    weights = (grouped - zeros.to(device).float().unsqueeze(-1)) * scales.to(device).float().unsqueeze(-1)
    return weights.reshape(rows, columns).to(dtype)


@torch.no_grad()
def quantize_linear(linear, hessian, config):
    """Natural column order, dynamic asymmetric groups, 1% damping by default."""
    weight = linear.weight.detach().float().clone()
    rows, columns = weight.shape
    if columns % config.group_size or hessian.shape != (columns, columns):
        raise ValueError("GPTQ weight, group and Hessian dimensions do not match")
    if not torch.isfinite(weight).all() or not torch.isfinite(hessian).all():
        raise ValueError("GPTQ requires finite weights and Hessian")
    hessian = hessian.detach().to(weight.device, torch.float32).clone()
    if torch.any(hessian.diag() < 0):
        raise ValueError("GPTQ Hessian diagonal must be nonnegative")
    dead = hessian.diag() == 0
    hessian[dead, dead] = 1
    weight[:, dead] = 0
    damping = config.damp_percent * hessian.diag().mean()
    hessian.diagonal().add_(damping)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    inverse_factor = torch.linalg.cholesky(inverse, upper=True)
    del inverse, hessian
    codes = torch.empty(rows, columns, dtype=torch.uint8, device=weight.device)
    scales, zeros = [], []
    for start in range(0, columns, config.block_size):
        end = start + config.block_size
        current = weight[:, start:end].clone()
        errors = torch.zeros_like(current)
        factor = inverse_factor[start:end, start:end]
        scale, zero = group_grid(weight[:, start:end], config.bits)
        scales.append(scale.cpu())
        zeros.append(zero.to(torch.uint8).cpu())
        for column in range(config.block_size):
            original = current[:, column]
            quantized = (torch.round(original / scale) + zero).clamp(0, 15)
            codes[:, start + column] = quantized.to(torch.uint8)
            reconstructed = scale * (quantized - zero)
            error = (original - reconstructed) / factor[column, column]
            current[:, column:] -= error.unsqueeze(1).matmul(factor[column, column:].unsqueeze(0))
            errors[:, column] = error
        weight[:, end:] -= errors.matmul(inverse_factor[start:end, end:])
    artifact = {
        "shape": [rows, columns],
        "packed_codes": (codes[:, ::2] | (codes[:, 1::2] << 4)).cpu(),
        "scales": torch.stack(scales, dim=1),
        "zeros": torch.stack(zeros, dim=1),
        "dead_columns": int(dead.sum()),
        "damping": float(damping),
    }
    linear.weight.copy_(dequantize(artifact, config, linear.weight.device, linear.weight.dtype))
    if not torch.isfinite(linear.weight).all():
        raise RuntimeError("GPTQ produced nonfinite weights")
    return artifact


@torch.no_grad()
def quantize_block(block, samples, config):
    """True-sequential QKV -> O -> up/gate -> down; propagate quantized outputs."""
    linears = llama_linears(block)
    device = next(block.parameters()).device
    groups = [
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("self_attn.o_proj",),
        ("mlp.up_proj", "mlp.gate_proj"),
        ("mlp.down_proj",),
    ]
    artifacts = {}
    for names in groups:
        # These Llama projections consume exactly the same tensor within a group.
        statistics = HessianAccumulator(linears[names[0]].in_features, device)

        def capture(module, args, statistics=statistics):
            statistics.add(args[0])

        handle = linears[names[0]].register_forward_pre_hook(capture)
        try:
            for x, kwargs in samples:
                block(x.to(device), **_to(kwargs, device))
        finally:
            handle.remove()
        if statistics.observations == 0:
            raise ValueError("GPTQ needs calibration observations")
        for name in names:
            artifact = quantize_linear(linears[name], statistics.matrix, config)
            artifact.update(observations=statistics.observations, tokens=statistics.tokens)
            artifacts[name] = artifact
        del statistics, capture, handle
    outputs = [(_output(block(x.to(device), **_to(kw, device))).cpu(), kw) for x, kw in samples]
    return outputs, artifacts


@torch.no_grad()
def calibrate(policy, observations, config):
    layers = policy.model.language_model.model.layers
    device = next(layers[0].parameters()).device
    samples = []

    class Captured(Exception):
        pass

    def capture(module, args, kwargs):
        if kwargs.get("past_key_value") is not None:
            raise RuntimeError("GPTQ calibration requires cache-free decoder inputs")
        samples.append((_to(args[0], "cpu"), _to(kwargs, "cpu")))
        raise Captured

    handle = layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for observation, instruction in observations:
            before = len(samples)
            try:
                policy.predict_prepared(observation, instruction)
            except Captured:
                pass
            if len(samples) != before + 1:
                raise RuntimeError("Failed to capture exactly one multimodal decoder input")
    finally:
        handle.remove()
    if not samples:
        raise ValueError("GPTQ needs calibration observations")
    policy.model.cpu()
    torch.cuda.empty_cache()
    artifacts = []
    with full_precision_statistics():
        for index, layer in enumerate(layers):
            print(f"GPTQ layer {index + 1}/{len(layers)}", flush=True)
            layer.to(device)
            samples, artifact = quantize_block(layer, samples, config)
            artifacts.append(artifact)
            layer.cpu()
            torch.cuda.empty_cache()
    policy.model.to(device)
    return artifacts


@torch.no_grad()
def apply_artifacts(model, artifacts, config):
    layers = model.language_model.model.layers
    if len(layers) != len(artifacts):
        raise ValueError("GPTQ artifact layer count does not match the checkpoint")
    for layer, artifact in zip(layers, artifacts):
        linears = llama_linears(layer)
        if set(linears) != set(artifact):
            raise ValueError("GPTQ artifact does not match the decoder-only projection scope")
        for name, linear in linears.items():
            if list(linear.weight.shape) != artifact[name]["shape"]:
                raise ValueError("GPTQ artifact weight shape does not match the checkpoint")
            linear.weight.copy_(dequantize(artifact[name], config, linear.weight.device, linear.weight.dtype))
