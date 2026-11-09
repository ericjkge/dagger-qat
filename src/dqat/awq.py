"""Llama-only AWQ: activation-aware scale search, clipping, then fake W4 quantization.

Adapted from MIT HAN Lab llm-awq, revision d6e797a42b9ef7778de8ee2352116e0f48a78d61.
See licenses/llm-awq.txt. No packed kernels or memory/speed claims. Only the seven
linear projections in each language decoder layer are quantized. Vision, multimodal
projector, embeddings, lm_head, proprio projector and action head stay unquantized.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AWQConfig:
    bits: int = 4
    group_size: int = 128
    scale_grid: int = 20
    clip_grid: int = 20
    clip_tokens: int = 512

    def __post_init__(self):
        if not 2 <= self.bits <= 8 or self.group_size < 1:
            raise ValueError("AWQ requires 2–8 bits and a positive group size")
        if min(self.scale_grid, self.clip_grid, self.clip_tokens) < 1:
            raise ValueError("AWQ search sizes must be positive")


@torch.no_grad()
def pseudo_quantize(weight, config):
    """Per-output-channel, asymmetric group quantize/dequantize with integer zeros."""
    if weight.shape[-1] % config.group_size:
        raise ValueError(f"Input width {weight.shape[-1]} is not divisible by {config.group_size}")
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize nonfinite weights")
    grouped = weight.float().reshape(-1, config.group_size)
    maximum, minimum = grouped.amax(1, keepdim=True), grouped.amin(1, keepdim=True)
    qmax = 2**config.bits - 1
    scale = (maximum - minimum).clamp(min=1e-5) / qmax
    zero = (-torch.round(minimum / scale)).clamp(0, qmax)
    quantized = (torch.round(grouped / scale) + zero).clamp(0, qmax)
    return ((quantized - zero) * scale).reshape(weight.shape).to(weight.dtype)


def _to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _to(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_to(item, device) for item in value)
    return value


def _output(value):
    return value[0] if isinstance(value, tuple) else value


@torch.no_grad()
def search_scale(module, linears, inputs, kwargs, config):
    """Choose alpha in [0, 1) by full module output MSE; preserve original weights."""
    device = next(module.parameters()).device
    originals = [linear.weight.detach().clone() for linear in linears]
    reference = [_output(module(x.to(device), **_to(kw, device))).cpu() for x, kw in zip(inputs, kwargs)]
    total_tokens = sum(x.numel() // x.shape[-1] for x in inputs)
    mean_abs = sum(x.float().reshape(-1, x.shape[-1]).abs().sum(0) for x in inputs) / total_tokens
    best_error, best_scales = float("inf"), None
    try:
        for index in range(config.scale_grid):
            scales = mean_abs.pow(index / config.scale_grid).clamp(min=1e-4).to(device)
            scales /= (scales.max() * scales.min()).sqrt()
            for linear, weight in zip(linears, originals):
                linear.weight.copy_(pseudo_quantize(weight.float() * scales, config) / scales)
            error, elements = 0.0, 0
            for x, kw, expected in zip(inputs, kwargs, reference):
                actual = _output(module(x.to(device), **_to(kw, device)))
                error += (actual.float() - expected.to(device).float()).square().sum().item()
                elements += actual.numel()
            error /= elements
            if error < best_error:
                best_error, best_scales = error, scales.detach().cpu()
    finally:
        for linear, weight in zip(linears, originals):
            linear.weight.copy_(weight)
    if best_scales is None:
        raise RuntimeError("AWQ scale search produced no finite candidate")
    return best_scales


@torch.no_grad()
def apply_scale(block, previous_name, linear_names, scales):
    previous = block.get_submodule(previous_name)
    scales = scales.to(previous.weight.device, previous.weight.dtype)
    if isinstance(previous, nn.Linear):
        if previous.out_features != scales.numel():
            raise ValueError("AWQ output scaling requires matching dimensions (no GQA)")
        previous.weight.div_(scales[:, None])
    else:
        previous.weight.div_(scales)
    if getattr(previous, "bias", None) is not None:
        previous.bias.div_(scales)
    for name in linear_names:
        block.get_submodule(name).weight.mul_(scales[None, :])


@torch.no_grad()
def search_clip(weight, inputs, config):
    """Per-group symmetric clipping candidates scored by grouped linear output MSE."""
    x = torch.cat([value.reshape(-1, value.shape[-1]) for value in inputs])
    indices = torch.linspace(0, len(x) - 1, min(config.clip_tokens, len(x))).long()
    x = (
        x[indices]
        .float()
        .to(weight.device)
        .reshape(-1, weight.shape[1] // config.group_size, config.group_size)
    )
    groups = weight.float().reshape(weight.shape[0], -1, config.group_size)
    best_bounds = []
    for start in range(0, len(groups), 64):
        w = groups[start : start + 64]
        original = torch.einsum("tgi,ogi->otg", x, w)
        bound = w.abs().amax(-1, keepdim=True)
        best, min_error = bound.clone(), torch.full_like(bound, float("inf"))
        for index in range(max(1, config.clip_grid // 2)):
            candidate = bound * (1 - index / config.clip_grid)
            quantized = pseudo_quantize(w.clamp(-candidate, candidate), config)
            predicted = torch.einsum("tgi,ogi->otg", x, quantized)
            error = (predicted - original).square().mean(1).unsqueeze(-1)
            better = error < min_error
            best = torch.where(better, candidate, best)
            min_error = torch.minimum(error, min_error)
        best_bounds.append(best.cpu())
    return torch.cat(best_bounds)


def llama_linears(block):
    expected = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    found = {name: module for name, module in block.named_modules() if isinstance(module, nn.Linear)}
    if set(found) != expected:
        raise ValueError(f"Unsupported decoder projections: {sorted(found)}")
    return found


@torch.no_grad()
def quantize_block(block, samples, config):
    device = next(block.parameters()).device
    linears = llama_linears(block)
    features = {
        name: [] for name in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")
    }
    handles = []
    for name in features:

        def capture(module, args, output, name=name):
            features[name].append(args[0].detach().cpu())

        handles.append(linears[name].register_forward_hook(capture))
    try:
        # As in llm-awq, propagate the ORIGINAL outputs to the next layer.
        outputs = [_output(block(x.to(device), **_to(kw, device))).cpu() for x, kw in samples]
    finally:
        for handle in handles:
            handle.remove()
    features["self_attn.k_proj"] = features["self_attn.v_proj"] = features["self_attn.q_proj"]
    features["mlp.up_proj"] = features["mlp.gate_proj"]
    mapping = [
        ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "self_attn"),
        ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], "mlp"),
        ("mlp.up_proj", ["mlp.down_proj"], "mlp.down_proj"),
    ]
    if linears["self_attn.v_proj"].weight.shape == linears["self_attn.o_proj"].weight.shape:
        mapping.insert(1, ("self_attn.v_proj", ["self_attn.o_proj"], "self_attn.o_proj"))
    scales = []
    for previous, names, inspected in mapping:
        kwargs = (
            [{key: value for key, value in kw.items() if key != "use_cache"} for _, kw in samples]
            if inspected == "self_attn"
            else [{} for _ in samples]
        )
        scale = search_scale(
            block.get_submodule(inspected),
            [linears[name] for name in names],
            features[names[0]],
            kwargs,
            config,
        )
        scales.append((previous, names, scale))
    for previous, names, scale in scales:
        apply_scale(block, previous, names, scale)
        for name in names:
            features[name] = [x / scale.to(x.dtype) for x in features[name]]
    clips = []
    for name, linear in linears.items():
        if name not in ("self_attn.q_proj", "self_attn.k_proj"):
            bound = search_clip(linear.weight, features[name], config)
            grouped = linear.weight.reshape(bound.shape[0], bound.shape[1], -1)
            bound_device = bound.to(device, grouped.dtype)
            grouped.copy_(grouped.clamp(-bound_device, bound_device))
            clips.append((name, bound))
        linear.weight.copy_(pseudo_quantize(linear.weight, config))
    return list(zip(outputs, [kw for _, kw in samples])), {"scales": scales, "clips": clips}


@torch.no_grad()
def calibrate(policy, observations, config):
    layers = policy.model.language_model.model.layers
    device = next(layers[0].parameters()).device
    samples = []

    class Captured(Exception):
        pass

    def capture(module, args, kwargs):
        if kwargs.get("past_key_value") is not None:
            raise RuntimeError("AWQ calibration requires cache-free decoder inputs")
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
        raise ValueError("AWQ needs calibration observations")
    policy.model.cpu()
    torch.cuda.empty_cache()
    artifacts = []
    for index, layer in enumerate(layers):
        print(f"AWQ layer {index + 1}/{len(layers)}", flush=True)
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
        raise ValueError("AWQ artifact layer count does not match the checkpoint")
    for layer, artifact in zip(layers, artifacts):
        for previous, names, scales in artifact["scales"]:
            apply_scale(layer, previous, names, scales)
        for name, bound in artifact["clips"]:
            weight = layer.get_submodule(name).weight
            bound = bound.to(weight.device, weight.dtype)
            grouped = weight.reshape(bound.shape[0], bound.shape[1], -1)
            grouped.copy_(grouped.clamp(-bound, bound))
        for linear in llama_linears(layer).values():
            linear.weight.copy_(pseudo_quantize(linear.weight, config))
