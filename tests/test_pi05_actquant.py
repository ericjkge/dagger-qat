"""CPU tests for the π0.5 activation fake-quantizer hooks, on a tiny model with the same module layout."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "actquant", Path(__file__).resolve().parents[1] / "src/dqat/pi05/actquant.py")
aq = importlib.util.module_from_spec(spec)
sys.modules["actquant"] = aq
spec.loader.exec_module(aq)


class Attn(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(w, w), nn.Linear(w, w), nn.Linear(w, w)
        self.o_proj = nn.Linear(w, w)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        a = torch.softmax(q @ k.transpose(-1, -2) / 4, -1) @ v
        return self.o_proj(a)


class Mlp(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(w, 2 * w), nn.Linear(w, 2 * w), nn.Linear(2 * w, w)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.self_attn, self.mlp = Attn(w), Mlp(w)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class Tiny(nn.Module):
    def __init__(self, w=16, depth=2):
        super().__init__()
        expert = SimpleNamespace(model=SimpleNamespace(layers=nn.ModuleList([Layer(w) for _ in range(depth)])))
        self.layers = expert.model.layers
        self.paligemma_with_expert = SimpleNamespace(gemma_expert=expert)

    def denoise_step(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def sample(self, x, steps=3):
        for _ in range(steps):
            x = x + 0.1 * self.denoise_step(x)
        return x


def test_grid_is_4bit_and_error_bounded():
    h = torch.randn(5, 64) * torch.linspace(0.1, 5, 64)
    scale, zero, qmax = aq.grid_params(h, 4)
    codes = aq.nearest_codes(h, scale, zero, qmax)
    assert codes.min() >= 0 and codes.max() <= 15
    for row in codes:
        assert len(torch.unique(row)) <= 16
    err = (aq.dequant(codes, scale, zero) - h).abs()
    assert torch.all(err <= scale / 2 + 1e-6 + scale * 0.5)  # zero-point rounding adds <= s/2 at the edge


def test_hooks_quantize_shared_inputs_once_and_count_steps():
    torch.manual_seed(0)
    m = Tiny()
    x = torch.randn(1, 4, 16)
    ref = m.sample(x)
    ctrl = aq.ActivationController(m).install()
    ctrl.quant = None
    ctrl.begin_call()
    assert torch.allclose(m.sample(x), ref)  # no-op when quant is None
    assert ctrl.step == 3
    seen = []
    m.layers[0].self_attn.k_proj.register_forward_pre_hook(lambda mod, a: seen.append(a[0].clone()))
    m.layers[0].self_attn.q_proj.register_forward_pre_hook(lambda mod, a: seen.append(a[0].clone()))
    ctrl.quant = aq.QuantConfig(bits=4)
    ctrl.begin_call()
    out = m.sample(x)
    assert not torch.allclose(out, ref)
    assert set(ctrl.stats) == {f"L{i}.{k}" for i in range(2) for k in aq.GROUP_KINDS}
    # q and k saw the same (quantized) tensor at each step
    assert torch.equal(seen[0], seen[1])
    ctrl.remove()


def test_step_and_group_selection():
    m = Tiny()
    x = torch.randn(1, 4, 16)
    ctrl = aq.ActivationController(m).install()
    ctrl.quant = aq.QuantConfig(bits=4, groups={"L1.down"}, steps={2})
    ctrl.begin_call()
    m.sample(x)
    assert set(ctrl.stats) == {"L1.down"}
    assert ctrl.stats["L1.down"][2] == 4 * 32  # one step, 4 tokens x 32 features


def test_additive_probe_gradient_matches_finite_difference():
    torch.manual_seed(1)
    m = Tiny().double()
    x = torch.randn(1, 4, 16, dtype=torch.float64)
    ctrl = aq.ActivationController(m).install()
    ctrl.probe, ctrl.record = "all", "all"
    ctrl.begin_call()
    y = m.sample(x)
    w = torch.randn_like(y)
    key = ("L0.gateup", 1)
    (g,) = torch.autograd.grad((w * y).sum(), [ctrl.leaves[key]])
    delta = torch.randn_like(g) * 1e-6
    ctrl.probe, ctrl.record = None, None
    ctrl.inject = {key: delta}
    ctrl.begin_call()
    y2 = m.sample(x)
    fd = ((w * (y2 - y)).sum()).item()
    lin = (g * delta).sum().item()
    assert abs(fd - lin) <= 1e-3 * abs(lin) + 1e-12


def test_rounder_keeps_grid():
    m = Tiny()
    x = torch.randn(1, 4, 16)
    ctrl = aq.ActivationController(m).install()
    calls = []

    def rounder(group, step, h, scale, zero, qmax, nearest):
        calls.append(group)
        return torch.clamp(torch.floor(h / scale) + zero, 0, qmax)  # round-down: still on the grid

    ctrl.quant, ctrl.rounder = aq.QuantConfig(bits=4, groups={"L0.qkv"}, steps={0}), rounder
    ctrl.begin_call()
    m.sample(x)
    assert calls == ["L0.qkv"]


def test_hadamard_orthonormal_roundtrip():
    from dqat.pi05.actquant import hadamard

    x = torch.randn(3, 5, 1024, dtype=torch.float64).float()
    y = hadamard(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-4)
    assert torch.allclose(hadamard(y, inverse=True), x, atol=1e-4)
    # spreads a single outlier evenly
    e = torch.zeros(1, 4096); e[0, 7] = 1.0
    assert torch.allclose(hadamard(e).abs(), torch.full((1, 4096), 4096**-0.5), atol=1e-6)
