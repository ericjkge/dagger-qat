"""Offline layer streaming reproduces the HF Llama forward and the baseline GPTQ exactly (tiny model, CPU)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaModel  # noqa: E402

from dqat.gptq import GPTQConfig, quantize_block  # noqa: E402
from dqat.offline.model import LINEARS, run_layer  # noqa: E402
from dqat.offline.stream import Arm, Streams, gptq_layers, linear_of, rtn_weight  # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=64, attn_implementation="sdpa")
    return LlamaModel(cfg).eval()


def test_run_layer_matches_llama_forward():
    model = tiny()
    x = torch.randn(2, 37, 256)
    with torch.no_grad():
        want = model(inputs_embeds=x, output_hidden_states=True).hidden_states[-2]  # input to final layer
        h = x
        for layer in model.layers[:-1]:
            h = run_layer(layer, h)
    torch.testing.assert_close(h, want, rtol=1e-5, atol=1e-5)


def test_rtn_is_gptq_grid_without_compensation():
    w = torch.randn(8, 256)
    q = rtn_weight(w, GPTQConfig())
    for g in range(2):
        block = q[:, g * 128 : (g + 1) * 128]
        assert all(len(torch.unique(row)) <= 16 for row in block)
    assert (q - w).abs().max() < (w.abs().max() * 2 / 15)


def test_gptq_layers_matches_quantize_block(tmp_path):
    model = tiny(1)
    layer = model.layers[0]
    frames = list(range(5))
    lengths = {f: 20 + 3 * (f % 2) for f in frames}
    offsets, pos = {}, 0
    for f in frames:
        offsets[f], pos = pos, pos + lengths[f]
    data = (torch.randn(pos, 256) * 0.5).half().numpy()
    inputs = np.memmap(tmp_path / "in.f16", dtype=np.float16, mode="w+", shape=(pos, 256))
    inputs[:] = data
    arm = Arm("g", "gptq", frames, calib=frames[:4])
    s = Streams(tmp_path, arm, lengths, inputs, offsets)
    weights = {n: linear_of(layer, n).weight.data.clone().bfloat16() for n in LINEARS}
    mine, ref = tiny(1).layers[0].half(), tiny(1).layers[0].half()
    for n in LINEARS:
        linear_of(mine, n).weight.data.copy_(weights[n])
        linear_of(ref, n).weight.data.copy_(weights[n])
    import dqat.gptq as gptq

    original = gptq.quantize_linear
    seen = {"mine": {}, "ref": {}}

    def recording(who):
        def fn(linear, hessian, config):
            seen[who][round(float(linear.weight.float().sum()), 3)] = hessian.float().clone()
            return original(linear, hessian, config)
        return fn

    gptq.quantize_linear = recording("mine")
    try:
        with ThreadPoolExecutor(2) as pool:
            gptq_layers({"g": mine}, {"g": s}, {"g": arm}, 0, weights, pool, lengths, "cpu", 4, 1,
                        lambda: False, lambda m: None)
        samples = []
        for f in arm.calib:
            x = torch.from_numpy(np.asarray(inputs[offsets[f] : offsets[f] + lengths[f]]))[None]
            pos_ids = torch.arange(x.shape[1])
            samples.append((x, dict(attention_mask=None, position_ids=pos_ids[None], past_key_value=None,
                                    output_attentions=False, use_cache=False, cache_position=pos_ids)))
        gptq.quantize_linear = recording("ref")
        with torch.no_grad():
            quantize_block(ref, samples, GPTQConfig())
    finally:
        gptq.quantize_linear = original
    # Same seven solver inputs (GPTQ itself is chaotic in fp16, so compare what it is given).
    assert len(seen["mine"]) == len(seen["ref"]) == 7
    for key, h in seen["ref"].items():
        assert float((seen["mine"][key] - h).norm() / h.norm()) < 0.02, key
    # Q/K/V see the unquantized block input, so they must agree to rounding ties.
    for n in LINEARS[:3]:
        a, b = linear_of(mine, n).weight.float(), linear_of(ref, n).weight.float()
        assert (a != b).float().mean() < 1e-3, n
