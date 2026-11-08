"""A tiny stand-in for openpi's PI0Pytorch with the same module names and call structure.

- embed_prefix calls paligemma_with_expert.embed_image once per camera slot (3 slots, the
  third zero-filled and masked), then appends prompt tokens (some of them padding),
- the PaliGemma LM runs once over the prefix,
- the action expert runs once per denoising step.
Padding is filled with BIG so tests can check it never reaches calibration statistics.
"""

import numpy as np
import torch
from torch import nn

BIG = 1e3
D_VIS, D_VLM, D_EXP, MLP = 8, 12, 6, 16
N_LAYERS, PATCHES, PROMPT, PAD, STEPS = 2, 4, 3, 2, 3


def _attn(d, out_name="o_proj"):
    m = nn.Module()
    for n in ("q_proj", "k_proj", "v_proj", out_name):
        setattr(m, n, nn.Linear(d, d, bias=out_name == "out_proj"))
    return m


def _gemma_layer(d):
    layer = nn.Module()
    layer.self_attn = _attn(d)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = nn.Linear(d, MLP, bias=False)
    layer.mlp.up_proj = nn.Linear(d, MLP, bias=False)
    layer.mlp.down_proj = nn.Linear(MLP, d, bias=False)
    layer.input_layernorm = nn.Module()
    layer.input_layernorm.dense = nn.Linear(d, d)  # adaRMS modulation in the expert
    return layer


def _gemma_forward(layer, x, key_mask=None):
    a = layer.self_attn
    logits = a.q_proj(x) @ a.k_proj(x).transpose(-1, -2) / x.shape[-1] ** 0.5
    if key_mask is not None:  # like π0.5: padding keys are masked out of attention
        logits = logits.masked_fill(~key_mask[:, None, :], float("-inf"))
    att = torch.softmax(logits, -1)
    x = x + a.o_proj(att @ a.v_proj(x))
    m = layer.mlp
    return x + m.down_proj(torch.nn.functional.gelu(m.gate_proj(x)) * m.up_proj(x))


class FakePWE(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = nn.Module()
        pm = self.paligemma.model = nn.Module()
        pm.vision_tower = nn.Module()
        pm.vision_tower.vision_model = nn.Module()
        pm.vision_tower.vision_model.encoder = nn.Module()
        vl = nn.Module()
        vl.self_attn = _attn(D_VIS, "out_proj")
        vl.mlp = nn.Module()
        vl.mlp.fc1, vl.mlp.fc2 = nn.Linear(D_VIS, MLP), nn.Linear(MLP, D_VIS)
        pm.vision_tower.vision_model.encoder.layers = nn.ModuleList([vl])
        pm.multi_modal_projector = nn.Module()
        pm.multi_modal_projector.linear = nn.Linear(D_VIS, D_VLM)
        pm.language_model = nn.Module()
        pm.language_model.layers = nn.ModuleList([_gemma_layer(D_VLM) for _ in range(N_LAYERS)])
        self.paligemma.lm_head = nn.Linear(D_VLM, 5)  # never called
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Module()
        self.gemma_expert.model.layers = nn.ModuleList(
            [_gemma_layer(D_EXP) for _ in range(N_LAYERS)]
        )
        self.gemma_expert.lm_head = nn.Linear(D_EXP, 5)  # never called

    def embed_image(self, img):  # [B, P, D_VIS]
        vl = self.paligemma.model.vision_tower.vision_model.encoder.layers[0]
        a = vl.self_attn
        att = torch.softmax(a.q_proj(img) @ a.k_proj(img).transpose(-1, -2), -1)
        x = img + a.out_proj(att @ a.v_proj(img))
        x = x + vl.mlp.fc2(torch.nn.functional.gelu(vl.mlp.fc1(x)))
        return self.paligemma.model.multi_modal_projector.linear(x)


class FakePI05(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.paligemma_with_expert = FakePWE()
        self.action_in_proj = nn.Linear(4, D_EXP)
        self.action_out_proj = nn.Linear(D_EXP, 4)
        self.vlm_to_expert = None  # (kept FP; not a Linear)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = [self.paligemma_with_expert.embed_image(im) for im in images]
        pads = [
            m[:, None].expand(im.shape[0], im.shape[1])
            for im, m in zip(images, img_masks, strict=True)
        ]
        return torch.cat(embs + [lang_tokens], 1), torch.cat(pads + [lang_masks], 1), None

    def sample_actions(self, images, img_masks, lang, lang_mask, noise):
        prefix, valid, _ = self.embed_prefix(images, img_masks, lang, lang_mask)
        for layer in self.paligemma_with_expert.paligemma.model.language_model.layers:
            prefix = _gemma_forward(layer, prefix, valid)
        w = valid[..., None].float()
        ctx = ((prefix * w).sum(1, keepdim=True) / w.sum(1, keepdim=True))[..., :D_EXP]
        x = noise
        for _ in range(STEPS):
            h = self.action_in_proj(x) + ctx
            for layer in self.paligemma_with_expert.gemma_expert.model.layers:
                h = _gemma_forward(layer, h)
            x = x - 0.3 * self.action_out_proj(h)
        return x


class FakePolicy:
    """Mimics openpi Policy.infer(obs, noise=...) -> {"actions": (H, 4)}."""

    def __init__(self, seed=0):
        self._model = FakePI05(seed)

    @torch.no_grad()
    def infer(self, obs, noise=None):
        img = torch.as_tensor(obs["observation/image"], dtype=torch.float32)
        wrist = torch.as_tensor(obs["observation/wrist_image"], dtype=torch.float32)
        images = [img[None], wrist[None], torch.full((1, PATCHES, D_VIS), BIG)]
        masks = [torch.tensor([True]), torch.tensor([True]), torch.tensor([False])]
        lang = torch.cat(
            [
                torch.as_tensor(obs["lang"], dtype=torch.float32)[None],
                torch.full((1, PAD, D_VLM), BIG),
            ],
            1,
        )
        lang_mask = torch.tensor([[True] * PROMPT + [False] * PAD])
        noise = torch.as_tensor(noise, dtype=torch.float32)[None]
        a = self._model.sample_actions(images, masks, lang, lang_mask, noise)
        return {"actions": a[0].numpy()}


def make_obs(n, seed=0, scale=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    return [
        {
            "observation/image": rng.normal(shift, scale, (PATCHES, D_VIS)).astype(np.float32),
            "observation/wrist_image": rng.normal(shift, scale, (PATCHES, D_VIS)).astype(
                np.float32
            ),
            "lang": rng.normal(0, 1, (PROMPT, D_VLM)).astype(np.float32),
        }
        for _ in range(n)
    ]


def noise_for(i):
    return np.random.default_rng(1000 + i).standard_normal((5, 4)).astype(np.float32)
