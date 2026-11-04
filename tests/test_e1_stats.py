"""Unit tests for the model-independent parts of scripts/e1_activation_shift.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "e1", Path(__file__).parents[1] / "scripts" / "e1_activation_shift.py"
)
e1 = importlib.util.module_from_spec(_spec)
sys.modules["e1"] = e1  # dataclasses need the module registered
_spec.loader.exec_module(e1)
S = e1.S


def test_group_of_real_openpi_names():
    p = "paligemma_with_expert.paligemma"
    assert e1.group_of(f"{p}.model.vision_tower.vision_model.encoder.layers.0.mlp.fc1") == "vision"
    assert e1.group_of(f"{p}.model.multi_modal_projector.linear") == "projector"
    assert e1.group_of(f"{p}.model.language_model.layers.3.mlp.down_proj") == "vlm"
    expert = "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj"
    assert e1.group_of(expert) == "expert"
    assert e1.group_of("action_out_proj") == "action_io"
    assert e1.group_of(f"{p}.lm_head") is None
    assert e1.group_of("paligemma_with_expert.gemma_expert.lm_head") is None


def test_int8_error_zero_on_grid_and_matches_bruteforce():
    s = torch.tensor(0.5)
    x = torch.arange(-127, 128, dtype=torch.float32) * s  # exactly representable
    assert e1.int8_sq_error(x, s).item() == 0.0
    x = torch.randn(1000) * 3
    s = x.abs().max() / 127
    q = np.clip(np.round(x.numpy() / s.item()), -127, 127) * s.item()
    assert e1.int8_sq_error(x, s).item() == pytest.approx(((x.numpy() - q) ** 2).sum(), rel=1e-5)


def test_static_scale_clips_values_outside_calibrated_range():
    x = torch.tensor([[1.0, 4.0]])
    # calibrated max 2 -> 4.0 is clipped to 2.0: error 4, plus 1.0 rounds exactly
    assert e1.int8_sq_error(x, torch.tensor(2.0 / 127)).item() == pytest.approx(4.0, rel=1e-4)


def test_layer_stats_counts():
    x = torch.tensor([[1.0, -3.0], [0.5, 2.0]])
    T, C = torch.tensor(2.5), torch.tensor([0.8, 2.5])
    st, cm = e1.layer_stats(x, T, C)
    assert torch.equal(cm, torch.tensor([1.0, 3.0]))
    assert st[S["n"]] == 4 and st[S["sumsq"]] == pytest.approx(1 + 9 + 0.25 + 4)
    assert st[S["n_over_T"]] == 1 and st[S["excess_sq_T"]] == pytest.approx(0.25)
    assert st[S["n_over_C"]] == 2  # 1.0 > 0.8 and 3.0 > 2.5
    # dynamic per-token is never worse than dynamic per-tensor, which never clips
    assert st[S["err_dyn_token"]] <= st[S["err_dyn_tensor"]] + 1e-12
    assert st[S["err_static"]] > st[S["err_dyn_tensor"]]  # static clips the -3.0


def test_layer_stats_calibration_pass_only_counts():
    st, _ = e1.layer_stats(torch.randn(5, 3))
    assert st[S["n"]] == 15 and st[S["err_static"]] == 0


def test_perturb():
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert e1.perturb("clean", img, rng) is img
    assert (e1.perturb("dark", img, rng) == 100).all()
    assert (e1.perturb("bright", img, rng) == 255).all()  # clipped
    warm = e1.perturb("warm", img, rng)
    assert warm[..., 0].max() == 230 and warm[..., 2].max() == 160
    n = e1.perturb("noise25", img, np.random.default_rng(1)).astype(float) - 200
    assert 15 < n.std() < 35 and e1.perturb("noise25", img, rng).dtype == np.uint8
    with pytest.raises(ValueError):
        e1.perturb("fog", img, rng)


def test_overlap_and_shift_metrics():
    a = np.arange(200, dtype=float)
    assert e1.topk_jaccard(a, a) == 1.0
    assert e1.topk_jaccard(a, a[::-1].copy()) == 0.0
    assert e1.chan_log2_shift(a * 2 + 1, a + 0.5) == pytest.approx(1.0, abs=1e-4)


def test_paired_bootstrap_ci_brackets_mean():
    d = np.random.default_rng(0).normal(1.0, 0.1, 200)
    m, lo, hi = e1.paired_bootstrap_ci(d)
    assert lo < m < hi and lo > 0.9 and hi < 1.1


# ---------------------------------------------------------------- probe + analysis on a fake model
# Mirrors the call structure of openpi's PI0Pytorch.sample_actions: embed_prefix calls
# paligemma_with_expert.embed_image once per camera, then the VLM runs on the prefix once,
# then the action expert runs once per denoising step.

BIG = 1e6  # planted in padding; must never show up in the statistics
nn = torch.nn


class _FakePWE(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        self.paligemma.model.vision_tower = nn.Module()
        self.paligemma.model.vision_tower.fc1 = nn.Linear(4, 4)
        self.paligemma.model.multi_modal_projector = nn.Module()
        self.paligemma.model.multi_modal_projector.linear = nn.Linear(4, 6)
        self.paligemma.model.language_model = nn.Module()
        self.paligemma.model.language_model.down_proj = nn.Linear(6, 6)
        self.paligemma.lm_head = nn.Linear(6, 3)  # never called
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Module()
        self.gemma_expert.model.q_proj = nn.Linear(6, 6)
        for p in self.parameters():
            nn.init.eye_(p) if p.ndim == 2 else nn.init.zeros_(p)

    def embed_image(self, img):  # img [B, P, 4]
        return self.paligemma.model.multi_modal_projector.linear(
            self.paligemma.model.vision_tower.fc1(img)
        )


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _FakePWE()
        self.action_out_proj = nn.Linear(6, 2)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = [self.paligemma_with_expert.embed_image(im) for im in images]
        pads = [
            m[:, None].expand(im.shape[0], im.shape[1])
            for im, m in zip(images, img_masks, strict=True)
        ]
        return torch.cat(embs + [lang_tokens], 1), torch.cat(pads + [lang_masks], 1), None

    def sample(self, images, img_masks, lang_tokens, lang_masks, steps=3):
        prefix, _, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        self.paligemma_with_expert.paligemma.model.language_model.down_proj(prefix)
        x = torch.ones(1, 2, 6)
        for _ in range(steps):
            x = self.paligemma_with_expert.gemma_expert.model.q_proj(x)
        return self.action_out_proj(x)


class _FakePolicy:
    def __init__(self):
        self._model = _FakeModel()

    def infer(self, obs, noise=None):
        def img(a):
            return torch.from_numpy(
                a[:2, :2].reshape(1, 4, 3)[..., :1].repeat(4, -1).astype("float32")
            )

        images = [
            img(obs["observation/image"]),
            img(obs["observation/wrist_image"]),
            torch.full((1, 4, 4), BIG),
        ]  # masked third camera, zero-filled in openpi
        masks = [torch.tensor([True]), torch.tensor([True]), torch.tensor([False])]
        lang = torch.cat([torch.ones(1, 2, 6), torch.full((1, 3, 6), BIG)], 1)  # 3 padded tokens
        lang_mask = torch.tensor([[True, True, False, False, False]])
        a = self._model.sample(images, masks, lang, lang_mask)
        return {"actions": np.tile(a[0, :1, :1].detach().numpy(), (e1.HORIZON, 7))}


def _frames(n, split):
    rng = np.random.default_rng(len(split))
    return [
        e1.Frame(
            i,
            split,
            "task",
            i,
            0,
            rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
            rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
            np.zeros(8, np.float32),
            np.zeros((e1.HORIZON, 7), np.float32),
        )
        for i in range(n)
    ]


def test_probe_excludes_padding_and_analysis_runs():
    pol = _FakePolicy()
    with pytest.raises(RuntimeError, match="unexpected"):
        bad = _FakeModel()
        bad.stray = nn.Linear(2, 2)
        e1.Probe(bad, device="cpu")
    probe = e1.Probe(pol._model, device="cpu")
    assert sorted(probe.groups) == ["action_io", "expert", "projector", "vision", "vlm"]

    cal = e1.run_condition(pol, probe, _frames(3, "calib"), "clean")
    for i, g in enumerate(probe.groups):
        assert cal["chan_max"][i].max() < 1000, f"padding leaked into {g}"
    vlm = probe.groups.index("vlm")
    # per frame: 2 cameras x 4 patches + 2 real prompt tokens = 10 tokens x 6 channels
    assert cal["per_frame"][0, vlm, S["n"]] == 10 * 6
    expert = probe.groups.index("expert")
    assert cal["per_frame"][0, expert, S["n"]] == 3 * 2 * 6  # 3 denoising steps x 2 tokens

    probe.set_calibration([torch.from_numpy(c) for c in cal["chan_max"]])
    fev = _frames(4, "eval")
    res = {c: e1.run_condition(pol, probe, fev, c) for c in ("clean", "dark", "bright")}
    s = e1.analyze(probe, fev, res, cal["chan_max"], np.zeros(7))
    v = s["conditions"]["bright"]["groups"]["vision"]
    assert v["range_ratio"] > s["conditions"]["dark"]["groups"]["vision"]["range_ratio"]
    assert len(v["paired_diff_err_static"]) == 3
    e1.print_summary(s)
    import json

    json.dumps(s)  # must be serializable
