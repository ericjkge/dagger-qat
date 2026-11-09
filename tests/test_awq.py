"""Numerical checks of AWQ using the same Llama fork used by the real policy."""

import copy

import pytest
import torch
from torch import nn

from dqat.awq import (
    AWQConfig,
    apply_artifacts,
    apply_scale,
    pseudo_quantize,
    quantize_block,
    search_clip,
    search_scale,
)


def reference_quantize(weight, bits, group_size):
    # Independent scalar/group reference for MIT HAN Lab's asymmetric grid.
    result = torch.empty_like(weight)
    for row in range(weight.shape[0]):
        for start in range(0, weight.shape[1], group_size):
            group = weight[row, start : start + group_size]
            scale = max(float(group.max() - group.min()), 1e-5) / (2**bits - 1)
            zero = min(max(round(-float(group.min()) / scale), 0), 2**bits - 1)
            result[row, start : start + group_size] = (
                torch.clamp(torch.round(group / scale) + zero, 0, 2**bits - 1) - zero
            ) * scale
    return result


def test_asymmetric_grid_matches_independent_reference():
    torch.manual_seed(4)
    weights = torch.randn(12, 32)
    weights[0] = 0
    config = AWQConfig(group_size=8)
    torch.testing.assert_close(pseudo_quantize(weights, config), reference_quantize(weights, 4, 8))
    assert torch.count_nonzero(pseudo_quantize(weights, config)[0]) == 0


def test_invalid_shapes_and_nonfinite_weights_fail():
    with pytest.raises(ValueError):
        pseudo_quantize(torch.ones(3, 15), AWQConfig(group_size=8))
    with pytest.raises(ValueError):
        pseudo_quantize(torch.full((2, 8), float("nan")), AWQConfig(group_size=8))


def make_block():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaDecoderLayer(config, layer_idx=0).eval()


def samples():
    return [
        (
            torch.randn(1, n, 32),
            dict(
                position_ids=torch.arange(n).unsqueeze(0),
                attention_mask=torch.zeros(1, 1, n, n),
                use_cache=False,
            ),
        )
        for n in (5, 7)
    ]


@pytest.mark.parametrize(
    "previous,names",
    [
        ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        ("self_attn.v_proj", ["self_attn.o_proj"]),
        ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ("mlp.up_proj", ["mlp.down_proj"]),
    ],
)
def test_fused_scaling_preserves_decoder_output(previous, names):
    torch.manual_seed(0)
    block = make_block()
    x, kwargs = samples()[0]
    expected = block(x, **kwargs)[0]
    width = block.get_submodule(names[0]).in_features
    apply_scale(block, previous, names, torch.linspace(0.3, 2.0, width))
    torch.testing.assert_close(block(x, **kwargs)[0], expected, rtol=2e-5, atol=1e-6)


def test_scale_search_preserves_weights_and_beats_rtn_candidate():
    torch.manual_seed(3)
    linear = nn.Linear(32, 16, bias=False)
    x = torch.randn(2, 17, 32)
    x[..., :2] *= 25
    original = linear.weight.detach().clone()
    config = AWQConfig(group_size=8)
    scales = search_scale(linear, [linear], [x], [{}], config)
    torch.testing.assert_close(linear.weight, original, rtol=0, atol=0)
    expected = nn.functional.linear(x, original)
    awq = nn.functional.linear(x, pseudo_quantize(original * scales, config) / scales)
    rtn = nn.functional.linear(x, pseudo_quantize(original, config))
    assert (awq - expected).square().mean() <= (rtn - expected).square().mean() + 1e-6


def test_clip_handles_small_samples_and_improves_grouped_error():
    torch.manual_seed(9)
    weights, x = torch.randn(13, 32), torch.randn(3, 32)
    config = AWQConfig(group_size=8)
    bound = search_clip(weights, [x], config)
    groups = weights.reshape(13, 4, 8)
    assert bound.shape == (13, 4, 1)
    assert torch.all(bound <= groups.abs().amax(-1, keepdim=True))
    reference = torch.einsum("tgi,ogi->otg", x.reshape(3, 4, 8), groups)

    def error(w):
        predicted = torch.einsum("tgi,ogi->otg", x.reshape(3, 4, 8), w)
        return (predicted - reference).square().mean(1)

    assert torch.all(
        error(pseudo_quantize(groups.clamp(-bound, bound), config))
        <= error(pseudo_quantize(groups, config)) + 1e-5
    )


def test_variable_length_multimodal_masks_and_artifact_roundtrip():
    torch.manual_seed(11)
    block = make_block()
    untouched = copy.deepcopy(block)
    original_q = block.self_attn.q_proj.weight.detach().clone()
    inputs = samples()
    with torch.no_grad():
        reference = [block(x, **kw)[0] for x, kw in inputs]
    config = AWQConfig(group_size=8, scale_grid=4, clip_grid=4, clip_tokens=8)
    propagated, artifact = quantize_block(block, inputs, config)
    for (actual, _), expected in zip(propagated, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([untouched])
    model.vision_backbone = nn.Linear(32, 32)
    vision_weight = model.vision_backbone.weight.detach().clone()
    apply_artifacts(model, [artifact], config)
    for name, parameter in block.named_parameters():
        torch.testing.assert_close(parameter, untouched.get_parameter(name), rtol=0, atol=0)
    assert not torch.equal(block.self_attn.q_proj.weight, original_q)
    torch.testing.assert_close(model.vision_backbone.weight, vision_weight, rtol=0, atol=0)


def test_capture_runs_real_decoder_and_removes_hooks():
    from transformers import LlamaConfig, LlamaForCausalLM

    from dqat.awq import calibrate

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        use_cache=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"

    class Policy:
        def __init__(self):
            self.model = nn.Module()
            self.model.language_model = LlamaForCausalLM(config).eval()

        def predict_prepared(self, observation, instruction):
            return self.model.language_model(inputs_embeds=observation)

    policy = Policy()
    frames = [(torch.randn(1, length, 32), "task") for length in (5, 8)]
    artifacts = calibrate(policy, frames, AWQConfig(group_size=8, scale_grid=2, clip_grid=2))
    assert len(artifacts) == 2
    assert not policy.model.language_model.model.layers[0]._forward_pre_hooks
    assert torch.isfinite(policy.predict_prepared(frames[0][0], "task").logits).all()
