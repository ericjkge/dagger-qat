# Test oracle copied from mit-han-lab/smoothquant @ c61476d728e42ae0d8a35e7e78494edcac3237b5
# (smoothquant/smooth.py `smooth_ln_fcs_llama_like`, smoothquant/fake_quant.py).
# MIT License, Copyright (c) 2022 MIT HAN Lab. Changes: RMSNorm isinstance assert dropped
# (any module with a .weight works); smooth_ln_fcs_llama_like also returns its scales.
# Otherwise verbatim.
# ruff: noqa
import torch
import torch.nn as nn


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)
    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t
