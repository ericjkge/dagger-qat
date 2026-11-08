"""Which Linears get quantized, SmoothQuant scale groups, and the fake-quantized Linear."""

from __future__ import annotations

import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from vlacal.quant.grids import quantize_act_per_token, quantize_act_static

_VLM = r"\.paligemma\.model\.language_model\.layers\.\d+\."
_EXPERT = r"\.gemma_expert\.model\.layers\.\d+\."
_VISION = r"\.vision_tower\.vision_model\.encoder\.layers\.\d+\."
LAYOUTS = {
    # QuantVLA (CVPR 2026) for π0.5: every LLM Linear + the action expert's MLP; attention
    # projections of the expert, the vision tower, projector and action/time heads stay FP.
    # QuantVLA reports 180 quantized layers for π0.5 (18 x 7 + 18 x 3).
    "quantvla": [
        _VLM + r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$",
        _EXPERT + r"mlp\.(gate|up|down)_proj$",
    ],
    # Every transformer-block Linear (still excludes projector, action/time heads, adaRMS).
    "all_blocks": [
        _VLM + r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$",
        _EXPERT + r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$",
        _VISION + r"(self_attn\.(q|k|v|out)_proj|mlp\.fc(1|2))$",
    ],
}
EXPECTED_COUNT = {"quantvla": 180}


def select_layers(model: nn.Module, layout: str) -> list[str]:
    patterns = [re.compile(p) for p in LAYOUTS[layout]]
    return [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(p.search(name) for p in patterns)
    ]


def smooth_groups(names: list[str]) -> list[list[str]]:
    """Layers that share one input after a norm, so they must share one smoothing vector.

    Same choice as official SmoothQuant for Llama-like blocks: smooth norm -> {q,k,v} and
    norm -> {gate, up} (and SigLIP's norm -> fc1). o_proj / down_proj / fc2 are not smoothed
    because their scale could not be folded into a preceding norm.
    """
    groups = defaultdict(list)
    for name in names:
        parent, _, leaf = name.rpartition(".")
        if leaf in ("q_proj", "k_proj", "v_proj"):
            groups[parent + ":qkv"].append(name)
        elif leaf in ("gate_proj", "up_proj"):
            groups[parent + ":gate_up"].append(name)
        elif leaf == "fc1":
            groups[parent + ":fc1"].append(name)
    return list(groups.values())


def smoothquant_scales(
    act_absmax: torch.Tensor, weights: list[torch.Tensor], alpha: float
) -> torch.Tensor:
    """s_j = max|X_j|^alpha / max|W_:,j|^(1-alpha), as official `smooth_ln_fcs` (float32)."""
    w_absmax = torch.stack([w.float().abs().amax(dim=0) for w in weights]).amax(dim=0)
    w_absmax = w_absmax.clamp(min=1e-5)
    return (act_absmax.float().pow(alpha) / w_absmax.pow(1 - alpha)).clamp(min=1e-5)


class QLinear(nn.Module):
    """Linear with fake-quantized weight and (optionally) smoothed + fake-quantized input.

    y = Q_w(W * s) @ Q_a(x / s) + b. With s folded into the preceding norm, this is what a
    SmoothQuant deployment computes. Quantization math runs in float32; the matmul runs in
    the layer's original dtype (bf16 in π0.5), like the FP model.
    """

    def __init__(
        self,
        linear: nn.Linear,
        weight: torch.Tensor,
        *,
        smooth: torch.Tensor | None = None,
        act_bits: int | None = None,
        act_mode: str | None = None,  # None | "static" | "dynamic"
        act_absmax: torch.Tensor | None = None,  # static: calibrated range of x / s
    ):
        super().__init__()
        if act_mode not in (None, "static", "dynamic"):
            raise ValueError(act_mode)
        if act_mode == "static" and act_absmax is None:
            raise ValueError("static activation quantization needs a calibrated range")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        dtype = linear.weight.dtype
        self.weight = nn.Parameter(weight.to(dtype), requires_grad=False)
        self.bias = linear.bias
        self.register_buffer("smooth", None if smooth is None else smooth.float())
        self.register_buffer("act_absmax", None if act_absmax is None else act_absmax.float())
        self.act_bits, self.act_mode = act_bits, act_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth is not None or self.act_mode is not None:
            xf = x.float()
            if self.smooth is not None:
                xf = xf / self.smooth
            if self.act_mode == "static":
                xf = quantize_act_static(xf, self.act_absmax, self.act_bits)
            elif self.act_mode == "dynamic":
                xf = quantize_act_per_token(xf, self.act_bits)
            x = xf.to(x.dtype)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, smooth={self.smooth is not None}, "
            f"act={self.act_mode}{'' if self.act_mode is None else self.act_bits}"
        )
