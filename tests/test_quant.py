"""Quantization core: cross-checks against official GPTQ / SmoothQuant code, and the
calibration -> quantize -> undo pipeline on a fake π0.5."""

import copy

import numpy as np
import pytest
import torch
from fake_pi05 import BIG, FakePolicy, make_obs, noise_for
from reference import gptq_reference as gref
from reference import smoothquant_reference as sqref

from vlacal.quant import QuantConfig, quantize_model
from vlacal.quant.capture import Collector, group_of
from vlacal.quant.gptq import gptq_quantize
from vlacal.quant.grids import (
    WeightGrid,
    quantize_act_per_token,
    quantize_act_static,
)
from vlacal.quant.layers import QLinear, select_layers, smooth_groups, smoothquant_scales

torch.manual_seed(0)


# ---------------------------------------------------------------- grids vs official


@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("bits", [3, 4, 8])
def test_weight_grid_matches_official_quantizer(sym, bits):
    w = torch.randn(16, 40) * torch.linspace(0.1, 3, 40)
    ref = gref.Quantizer()
    ref.configure(bits, perchannel=True, sym=sym)
    ref.find_params(w, weight=True)
    expected = gref.quantize(w, ref.scale, ref.zero, ref.maxq)
    grid = WeightGrid(bits, sym)
    assert torch.equal(grid.rtn(w), expected)


def test_weight_grid_groups_quantize_each_group_independently():
    w = torch.randn(4, 64)
    w[:, 32:] *= 100
    grid = WeightGrid(4, group_size=32)
    q = grid.rtn(w)
    assert torch.equal(q[:, :32], WeightGrid(4).rtn(w[:, :32]))
    assert torch.equal(q[:, 32:], WeightGrid(4).rtn(w[:, 32:]))


def test_per_token_matches_official_smoothquant():
    x = torch.randn(3, 7, 20) * 5
    assert torch.equal(
        quantize_act_per_token(x.clone(), 8), sqref.quantize_activation_per_token_absmax(x.clone())
    )


def test_static_act_clips_outside_calibrated_range_and_dynamic_does_not():
    x = torch.tensor([[0.5, 1.0, 10.0]])
    s = quantize_act_static(x, 1.0, 8)
    assert s[0, 2] == pytest.approx(1.0) and s[0, 1] == pytest.approx(1.0)
    assert quantize_act_per_token(x, 8)[0, 2] == pytest.approx(10.0)


# ---------------------------------------------------------------- GPTQ vs official


@pytest.mark.parametrize("group", [-1, 16])
@pytest.mark.parametrize("sym", [False, True])
def test_gptq_matches_official(group, sym):
    torch.manual_seed(1)
    lin = torch.nn.Linear(48, 10, bias=False)
    x = torch.randn(6, 30, 48) @ torch.randn(48, 48) * 0.3  # correlated inputs
    ref = gref.GPTQ(lin)
    ref.quantizer = gref.Quantizer()
    ref.quantizer.configure(4, perchannel=True, sym=sym)
    for b in range(x.shape[0]):
        ref.add_batch(x[b : b + 1], None)
    H = ref.H.clone()
    q_ref = ref.fasterquant(blocksize=16, percdamp=0.01, groupsize=group)
    q, loss = gptq_quantize(lin.weight, H, WeightGrid(4, sym, group), blocksize=16)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=1e-6)
    assert loss == pytest.approx(ref.loss, rel=1e-4)


def test_gptq_beats_rtn_on_its_own_hessian():
    torch.manual_seed(2)
    w = torch.randn(32, 64)
    x = torch.randn(500, 64) @ torch.randn(64, 64)
    H = 2 * x.T @ x / len(x)
    grid = WeightGrid(3)
    q, _ = gptq_quantize(w, H, grid)

    def err(wq):
        return float(((x @ (w - wq).T) ** 2).sum())

    assert err(q) < 0.7 * err(grid.rtn(w))


def test_gptq_is_invariant_to_hessian_scale():
    w, x = torch.randn(8, 32), torch.randn(100, 32)
    H = x.T @ x
    q1, _ = gptq_quantize(w, H, WeightGrid(4))
    q2, _ = gptq_quantize(w, 37.0 * H, WeightGrid(4))
    torch.testing.assert_close(q1, q2)


# ---------------------------------------------------------------- SmoothQuant vs official


def test_smoothquant_scales_match_official():
    torch.manual_seed(3)
    fcs = [torch.nn.Linear(24, 9, bias=False), torch.nn.Linear(24, 5, bias=False)]
    act = torch.rand(24) * 20
    ln = torch.nn.Module()
    ln.weight = torch.nn.Parameter(torch.ones(24))
    ours = smoothquant_scales(act, [f.weight for f in fcs], alpha=0.5)
    fcs_ref = copy.deepcopy(fcs)
    theirs = sqref.smooth_ln_fcs_llama_like(ln, fcs_ref, act, alpha=0.5)
    torch.testing.assert_close(ours, theirs)
    for f, fr in zip(fcs, fcs_ref, strict=True):  # same smoothed weights
        torch.testing.assert_close(f.weight * ours[None, :], fr.weight)


def test_qlinear_smoothing_alone_is_exact_up_to_rounding():
    lin = torch.nn.Linear(16, 4)
    s = torch.rand(16) + 0.5
    q = QLinear(lin, lin.weight.detach() * s[None, :], smooth=s)
    x = torch.randn(5, 16)
    torch.testing.assert_close(q(x), lin(x), rtol=1e-5, atol=1e-5)


def test_qlinear_without_quantization_is_bit_exact():
    lin = torch.nn.Linear(16, 4).to(torch.bfloat16)
    q = QLinear(lin, lin.weight.detach())
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    assert torch.equal(q(x), lin(x)) and q(x).dtype == torch.bfloat16


# ---------------------------------------------------------------- layout and groups


def _openpi_names():
    vlm = "paligemma_with_expert.paligemma.model.language_model.layers.{}."
    exp = "paligemma_with_expert.gemma_expert.model.layers.{}."
    attn = [f"self_attn.{p}_proj" for p in "qkvo"]
    mlp = [f"mlp.{p}_proj" for p in ("gate", "up", "down")]
    names = [vlm.format(i) + n for i in range(18) for n in attn + mlp]
    names += [exp.format(i) + n for i in range(18) for n in attn + mlp + ["input_layernorm.dense"]]
    return names + ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]


def test_quantvla_layout_selects_180_layers_of_real_pi05_names():
    model = torch.nn.Module()
    for name in _openpi_names():
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            if not hasattr(parent, p):
                setattr(parent, p, torch.nn.Module())
            parent = getattr(parent, p)
        setattr(parent, leaf, torch.nn.Linear(2, 2))
    sel = select_layers(model, "quantvla")
    assert len(sel) == 180
    assert not any("self_attn" in n and "gemma_expert" in n for n in sel)
    assert not any(n.endswith("dense") or n.startswith(("action", "time")) for n in sel)


def test_smooth_groups_share_inputs():
    names = select_layers(FakePolicy()._model, "quantvla")
    groups = smooth_groups(names)
    leaves = sorted(tuple(sorted(n.rsplit(".", 1)[1] for n in g)) for g in groups)
    assert ("k_proj", "q_proj", "v_proj") in leaves and ("gate_proj", "up_proj") in leaves
    assert not any("o_proj" in n or "down_proj" in n for g in groups for n in g)


def test_group_of_rejects_unknown_layers():
    with pytest.raises(ValueError):
        group_of("some_new_head")


# ---------------------------------------------------------------- pipeline on fake π0.5


def _runner(policy, frames):
    def run(collector):
        for i, obs in enumerate(frames):
            policy.infer(obs, noise=noise_for(i))

    return run


def test_collector_excludes_padding_and_builds_hessians_in_budgeted_passes():
    pol = FakePolicy()
    names = select_layers(pol._model, "all_blocks")
    passes = []
    run = _runner(pol, make_obs(4))

    def counting(col):
        passes.append(1)
        run(col)

    stats = Collector(pol._model, names).run(
        counting, names, hessian_budget_gb=1e-6, log=lambda m: None
    )
    assert len(passes) > 2  # tiny budget forces one Hessian group per pass
    for name, st in stats.items():
        assert st.chan_absmax.max() < BIG / 10, f"padding leaked into {name}"
        assert st.hessian is not None and torch.allclose(st.hessian, st.hessian.T)
    vlm_q = next(n for n in names if "language_model.layers.0.self_attn.q_proj" in n)
    assert stats[vlm_q].tokens == 4 * (2 * 4 + 3)  # 4 frames x (2 real cameras x 4 + 3 prompt)


def test_weighted_hessian_matches_manual_weighting():
    pol = FakePolicy()
    name = select_layers(pol._model, "quantvla")[0]
    frames = make_obs(2)

    def weighted(col):
        for i, obs in enumerate(frames):
            col.sample_weight = [1.0, 3.0][i]
            pol.infer(obs, noise=noise_for(i))

    hw = Collector(pol._model, [name]).run(weighted, [name], log=lambda m: None)[name].hessian

    def single(i):
        def run(col):
            pol.infer(frames[i], noise=noise_for(i))

        return Collector(pol._model, [name]).run(run, [name], log=lambda m: None)[name]

    a, b = single(0), single(1)  # each normalized by its own token count
    manual = (a.hessian * a.tokens + 3 * b.hessian * b.tokens) / (a.tokens + 3 * b.tokens)
    torch.testing.assert_close(hw, manual)


CONFIGS = {
    "smoothquant_w4a8_static": QuantConfig(),
    "w4a8_dynamic": QuantConfig(a_mode="dynamic", smooth_alpha=None),
    "gptq_w4": QuantConfig(w_method="gptq", a_bits=None, a_mode=None, smooth_alpha=None),
    "gptq_smooth_w4a8": QuantConfig(w_method="gptq", w_group_size=8),
    "fp_noop": QuantConfig(w_bits=None, a_bits=None, a_mode=None, smooth_alpha=None),
}


@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_quantize_then_undo_restores_exact_outputs(cfg_name):
    pol = FakePolicy()
    evals = make_obs(3, seed=9)
    before = [pol.infer(o, noise=noise_for(i))["actions"] for i, o in enumerate(evals)]
    h = quantize_model(pol._model, CONFIGS[cfg_name], _runner(pol, make_obs(4)), log=lambda m: None)
    assert h.report["n_layers"] == 2 * 7 + 2 * 3
    during = [pol.infer(o, noise=noise_for(i))["actions"] for i, o in enumerate(evals)]
    h.undo()
    after = [pol.infer(o, noise=noise_for(i))["actions"] for i, o in enumerate(evals)]
    for b, d, a in zip(before, during, after, strict=True):
        assert np.array_equal(b, a)  # undo is exact
        if cfg_name == "fp_noop":
            np.testing.assert_allclose(d, b, rtol=1e-6, atol=1e-6)
        else:
            assert not np.array_equal(d, b)  # quantization did something


def test_calibration_data_changes_static_ranges_but_not_dynamic_rtn():
    """The experimental variable: static scales / GPTQ depend on calibration data,
    dynamic-activation RTN does not."""

    def quantized_actions(cfg, calib):
        pol = FakePolicy()
        quantize_model(pol._model, cfg, _runner(pol, calib), log=lambda m: None)
        return pol.infer(make_obs(1, seed=5)[0], noise=noise_for(0))["actions"]

    clean, shifted = make_obs(4, seed=1), make_obs(4, seed=1, scale=3.0, shift=2.0)
    static = QuantConfig(a_bits=4)
    assert not np.array_equal(quantized_actions(static, clean), quantized_actions(static, shifted))
    gptq = QuantConfig(w_method="gptq", a_bits=None, a_mode=None, smooth_alpha=None)
    assert not np.array_equal(quantized_actions(gptq, clean), quantized_actions(gptq, shifted))
    dyn = QuantConfig(a_mode="dynamic", smooth_alpha=None)
    assert not dyn.needs_calibration
    pol = FakePolicy()
    quantize_model(pol._model, dyn, None, log=lambda m: None)  # no data needed


def test_config_validation_and_json_roundtrip():
    with pytest.raises(ValueError):
        QuantConfig(a_bits=8, a_mode=None)
    with pytest.raises(ValueError):
        QuantConfig(w_method="awq")
    cfg = QuantConfig(w_method="gptq", w_group_size=128)
    import json

    assert QuantConfig.from_json(json.dumps(cfg.to_dict())) == cfg
