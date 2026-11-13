"""Quantizable layers and the symmetric RTN group quantizer shared by scoring and evaluation."""

import re

import torch

GROUP = 128
BITS = (2, 3, 4, 8)
# Vision backbone transformer-block linears (DINOv2 + SigLIP) and Llama decoder linears.
TARGET = re.compile(
    r"^vision_backbone\.(featurizer|fused_featurizer)\.blocks\.(\d+)\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))$"
    r"|^language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$"
)


def target_layers(model):
    """Quantizable linears. OpenVLA takes features from the second-to-last ViT
    block, so each ViT's last block cannot affect the action and is excluded."""
    vb = model.vision_backbone
    last = {"featurizer": len(vb.featurizer.blocks) - 1, "fused_featurizer": len(vb.fused_featurizer.blocks) - 1}
    layers = {}
    for n, m in model.named_modules():
        match = TARGET.match(n)
        if isinstance(m, torch.nn.Linear) and match and not (match[1] and int(match[2]) == last[match[1]]):
            layers[n] = m
    return layers


def quantize(w, bits):
    """Symmetric round-to-nearest, groups of GROUP along the input dim, 2^b - 1 levels.
    Returns the dequantized weight in w's dtype, exactly as evaluation stores it.
    The last group is zero-padded if needed."""
    out, inp = w.shape
    pad = (-inp) % GROUP
    x = torch.nn.functional.pad(w.float(), (0, pad)).view(out, -1, GROUP)
    levels = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / levels
    q = torch.clamp(torch.round(x / scale), -levels, levels) * scale
    return q.view(out, -1)[:, :inp].to(w.dtype)


@torch.no_grad()
def apply_bits(model, bits_by_layer):
    layers = target_layers(model)
    if set(bits_by_layer) != set(layers):
        raise ValueError("Allocation does not cover exactly the quantizable layers")
    for name, bits in bits_by_layer.items():
        w = layers[name].weight
        w.copy_(quantize(w, bits))
