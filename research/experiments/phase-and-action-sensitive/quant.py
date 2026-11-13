"""Quantizable layers and the symmetric RTN group quantizer shared by scoring and evaluation."""

import re

import torch

GROUP = 128
BITS = (2, 3, 4, 8)
# Vision backbone transformer-block linears (DINOv2 + SigLIP) and Llama decoder linears.
TARGET = re.compile(
    r"^vision_backbone\.(featurizer|fused_featurizer)\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))$"
    r"|^language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$"
)


def target_layers(model):
    layers = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and TARGET.match(n)}
    if not layers:
        raise RuntimeError("No quantizable layers matched")
    return layers


def quantize(w, bits):
    """Symmetric round-to-nearest, groups of GROUP along the input dim, 2^b - 1 levels.
    Returns the dequantized weight in float32. The last group is zero-padded if needed."""
    out, inp = w.shape
    pad = (-inp) % GROUP
    x = torch.nn.functional.pad(w.float(), (0, pad)).view(out, -1, GROUP)
    levels = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / levels
    q = torch.clamp(torch.round(x / scale), -levels, levels) * scale
    return q.view(out, -1)[:, :inp]


@torch.no_grad()
def apply_bits(model, bits_by_layer):
    layers = target_layers(model)
    if set(bits_by_layer) != set(layers):
        raise ValueError("Allocation does not cover exactly the quantizable layers")
    for name, bits in bits_by_layer.items():
        w = layers[name].weight
        w.copy_(quantize(w, bits).to(w.dtype))
