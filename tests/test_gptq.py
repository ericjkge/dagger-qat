import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dqat.gptq import (
    GPTQConfig,
    HessianAccumulator,
    apply_artifacts,
    calibrate,
    dequantize,
    full_precision_statistics,
    quantize_block,
    quantize_linear,
)


def test_hessian_matches_gram_for_variable_length_observations():
    torch.manual_seed(8)
    first, second = torch.randn(2, 5, 16), torch.randn(1, 9, 16)
    stats = HessianAccumulator(16, "cpu")
    stats.add(first)
    stats.add(second)
    x = torch.cat((first.reshape(-1, 16), second.reshape(-1, 16)))
    torch.testing.assert_close(stats.matrix, 2 / 3 * x.t().matmul(x))
    assert stats.observations == 3 and stats.tokens == 19


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_quantization_matches_original_gptq_fixture(dtype_name):
    dtype = getattr(torch, dtype_name)
    with np.load(Path(__file__).parent / "fixtures/gptq-reference.npz") as reference:
        linear = nn.Linear(16, 6, bias=False).to(dtype)
        linear.weight.data.copy_(torch.from_numpy(reference["weights"]).to(dtype))
        stats = HessianAccumulator(16, "cpu")
        for key in ("first", "second"):
            stats.add(torch.from_numpy(reference[key]).to(dtype))
        hessian = stats.matrix.clone()
        config = GPTQConfig(group_size=8, block_size=8)
        artifact = quantize_linear(linear, stats.matrix, config)
        torch.testing.assert_close(
            linear.weight.float(),
            torch.from_numpy(reference["expected_" + dtype_name]),
            rtol=2e-5 if dtype == torch.float32 else 0,
            atol=2e-6 if dtype == torch.float32 else 0,
        )
        torch.testing.assert_close(stats.matrix, hessian, rtol=0, atol=0)
    assert artifact["packed_codes"].numel() * 2 == linear.weight.numel()
    assert artifact["dead_columns"] == 1
    torch.testing.assert_close(dequantize(artifact, config, "cpu", dtype), linear.weight, rtol=0, atol=0)
    assert torch.count_nonzero(linear.weight[:, -1]) == 0


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


def make_samples():
    return [
        (
            torch.randn(1, length, 32),
            dict(
                position_ids=torch.arange(length).unsqueeze(0),
                attention_mask=torch.triu(torch.full((1, 1, length, length), -10000.0), diagonal=1),
                use_cache=False,
            ),
        )
        for length in (5, 7)
    ]


def test_sequential_block_propagation_and_packed_artifact_replay(tmp_path):
    torch.manual_seed(14)
    block = make_block()
    untouched = copy.deepcopy(block)
    samples = make_samples()
    config = GPTQConfig(group_size=8, block_size=8)
    propagated, artifact = quantize_block(block, samples, config)
    assert len(artifact) == 7
    assert all(part["observations"] == 2 and part["tokens"] == 12 for part in artifact.values())
    with torch.no_grad():
        for (actual, _), (inputs, kwargs) in zip(propagated, samples):
            torch.testing.assert_close(actual, block(inputs, **kwargs)[0], rtol=0, atol=0)
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([untouched])
    model.vision_backbone = nn.Linear(32, 32)
    model.action_head = nn.Linear(32, 7)
    protected = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("language_model.") or "layernorm" in name
    }
    cache = tmp_path / "gptq.pt"
    torch.save([artifact], cache)
    apply_artifacts(model, torch.load(cache, weights_only=True), config)
    for name, parameter in block.named_parameters():
        torch.testing.assert_close(parameter, untouched.get_parameter(name), rtol=0, atol=0)
    for name, before in protected.items():
        torch.testing.assert_close(model.state_dict()[name], before, rtol=0, atol=0)
    for module in block.modules():
        assert not module._forward_pre_hooks


def test_multimodal_capture_uses_real_decoder_and_cleans_hooks():
    from transformers import LlamaConfig, LlamaForCausalLM

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

    torch.manual_seed(17)
    policy = Policy()
    observations = [(torch.randn(1, length, 32), "task") for length in (5, 8)]
    artifacts = calibrate(policy, observations, GPTQConfig(group_size=8, block_size=8))
    assert len(artifacts) == 2
    assert not policy.model.language_model.model.layers[0]._forward_pre_hooks
    assert torch.isfinite(policy.predict_prepared(observations[0][0], "task").logits).all()


def test_invalid_inputs_and_artifacts_fail_explicitly():
    with pytest.raises(ValueError, match="four-bit"):
        GPTQConfig(bits=3)
    with pytest.raises(ValueError, match="damping"):
        GPTQConfig(damp_percent=0)
    stats = HessianAccumulator(8, "cpu")
    with pytest.raises(ValueError, match="Nonfinite"):
        stats.add(torch.full((1, 3, 8), float("nan")))
    linear = nn.Linear(8, 4, bias=False)
    config = GPTQConfig(group_size=8, block_size=8)
    with pytest.raises(ValueError, match="dimensions"):
        quantize_linear(linear, torch.eye(7), config)
    artifact = quantize_linear(linear, torch.eye(8), config)
    artifact["scales"][0, 0] = 0
    with pytest.raises(ValueError, match="Malformed"):
        dequantize(artifact, config, "cpu", torch.float32)


def test_statistics_precision_flags_restore_after_failure():
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError, match="test"):
        with full_precision_statistics():
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            raise RuntimeError("test")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
