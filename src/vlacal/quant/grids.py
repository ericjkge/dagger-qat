"""Quantization grids (quantize -> dequantize in float).

Weights use GPTQ's min-max grid for every method (RTN, GPTQ, SmoothQuant), so arms differ
only in algorithm, never in grid. Activations use SmoothQuant's symmetric abs-max grid,
with either a static (calibrated) per-tensor scale or a dynamic per-token scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WeightGrid:
    """Per-output-channel (optionally per-group) min-max grid, as in GPTQ's `Quantizer`.

    sym=True follows GPTQ's convention: scale = 2*absmax / (2^b - 1), zero = 2^(b-1).
    """

    bits: int = 4
    sym: bool = False
    group_size: int = -1  # -1: one scale per output row

    @property
    def maxq(self) -> int:
        return 2**self.bits - 1

    def params(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale and zero for each row of w [rows, cols] (float32). Shapes [rows, 1]."""
        zeros = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
        xmin = torch.minimum(w.min(dim=1).values, zeros)
        xmax = torch.maximum(w.max(dim=1).values, zeros)
        if self.sym:
            xmax = torch.maximum(xmin.abs(), xmax)
            xmin = torch.where(xmin < 0, -xmax, xmin)
        both_zero = (xmin == 0) & (xmax == 0)
        xmin = torch.where(both_zero, -torch.ones_like(xmin), xmin)
        xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
        scale = (xmax - xmin) / self.maxq
        if self.sym:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)
        return scale[:, None], zero[:, None]

    def quantize(self, w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
        return scale * (q - zero)

    def rtn(self, w: torch.Tensor) -> torch.Tensor:
        """Round-to-nearest of a full weight matrix [out, in]; returns float32."""
        w = w.float()
        if self.group_size == -1:
            return self.quantize(w, *self.params(w))
        out = torch.empty_like(w)
        for c in range(0, w.shape[1], self.group_size):
            block = w[:, c : c + self.group_size]
            out[:, c : c + self.group_size] = self.quantize(block, *self.params(block))
        return out


def act_qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def quantize_act_static(x: torch.Tensor, absmax: torch.Tensor | float, bits: int) -> torch.Tensor:
    """Symmetric per-tensor quantization with a fixed (calibrated) range. Clips outside it."""
    qmax = act_qmax(bits)
    scale = torch.as_tensor(absmax, dtype=x.dtype, device=x.device).clamp(min=1e-5) / qmax
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_act_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric dynamic per-token quantization (SmoothQuant's per_token absmax). Never clips."""
    qmax = act_qmax(bits)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / qmax
    return torch.round(x / scale) * scale
