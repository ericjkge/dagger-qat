"""Official openpi π0.5 LIBERO policy (PyTorch path) with deterministic noise and a differentiable path.

`Pi05.infer(element, noise)` reproduces openpi's `Policy.infer` exactly (same input/output transforms,
same `sample_actions`), but with torch.compile disabled so activation hooks are honoured, and with the
flow-matching noise always supplied by the caller so paired BF16/quantized calls share randomness.

`Pi05.actions_with_grad(element, noise)` re-runs the same computation with autograd enabled through
the action expert only (the VLM prefix/KV cache is computed under no_grad, it is not quantized), and
returns environment-unit actions as a differentiable tensor. The output transform for LIBERO is
elementwise affine (quantile unnormalization, then slicing to 7 dims); we recover it numerically and
verify affinity once at load time.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

ACTION_HORIZON = 10
ACTION_DIM_MODEL = 32
ACTION_DIM_ENV = 7
REPLAN_STEPS = 5  # openpi's LIBERO client executes the first 5 actions of each 10-action chunk
RESIZE = 224


def quat2axisangle(quat):
    """Verbatim from openpi examples/libero/main.py (robosuite convention)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_element(obs, prompt):
    """openpi examples/libero/main.py preprocessing: 180° rotation, resize-with-pad to 224, uint8."""
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state,
            "prompt": str(prompt)}


class Pi05:
    def __init__(self, checkpoint_dir, device="cuda"):
        from openpi.models import model as _model
        from openpi.policies import policy_config
        from openpi.training import config as _config

        cfg = _config.get_config("pi05_libero")
        # Hooks need eager execution; compile changes nothing numerically we rely on.
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        self.policy = policy_config.create_trained_policy(cfg, checkpoint_dir, pytorch_device=device)
        self.model = self.policy._model
        self.device = device
        self._Observation = _model.Observation
        assert self.model.config.action_horizon == ACTION_HORIZON
        assert self.model.config.pi05
        # Elementwise-affine output transform: env = offset + scale * x[..., :7].
        zeros = self._output(np.zeros((ACTION_HORIZON, ACTION_DIM_MODEL), np.float32))
        ones = self._output(np.ones((ACTION_HORIZON, ACTION_DIM_MODEL), np.float32))
        self.out_offset = zeros
        self.out_scale = ones - zeros
        probe = np.random.default_rng(0).uniform(-1.5, 1.5, (ACTION_HORIZON, ACTION_DIM_MODEL)).astype(np.float32)
        affine = self.out_offset + self.out_scale * probe[:, :ACTION_DIM_ENV]
        if not np.allclose(self._output(probe), affine, atol=1e-5):
            raise RuntimeError("Output transform is not elementwise affine; differentiable path invalid")

    def _output(self, x_norm):
        out = self.policy._output_transform({"state": np.zeros(8, np.float32), "actions": x_norm})
        return np.asarray(out["actions"], dtype=np.float64)

    def _prepare(self, element):
        import jax

        inputs = jax.tree.map(lambda x: x, element)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...], inputs)
        return self._Observation.from_dict(inputs)

    @staticmethod
    def noise(seed):
        return np.random.default_rng(seed).standard_normal((ACTION_HORIZON, ACTION_DIM_MODEL)).astype(np.float32)

    @torch.no_grad()
    def infer(self, element, noise):
        """Identical to openpi Policy.infer with explicit noise. Returns (10, 7) env-unit actions."""
        out = self.policy.infer(element, noise=noise)
        return np.asarray(out["actions"], dtype=np.float64)

    def actions_with_grad(self, element, noise, num_steps=10):
        """Same computation as sample_actions, autograd enabled inside the action expert.

        Returns a (10, 7) float32 torch tensor in env units. Must match `infer` numerically
        (checked by the caller on the first state).
        """
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        m = self.model
        observation = self._prepare(element)
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
            prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens,
                                                                             lang_masks)
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
            m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            _, past_key_values = m.paligemma_with_expert.forward(
                attention_mask=m._prepare_attention_masks_4d(prefix_att_2d), position_ids=prefix_pos,
                past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        x_t = torch.from_numpy(noise).to(self.device)[None]
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        with torch.enable_grad():
            while time >= -dt / 2:
                v_t = m.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(1))
                x_t = x_t + dt * v_t
                time = time + dt
        scale = torch.from_numpy(self.out_scale).float().to(self.device)
        offset = torch.from_numpy(self.out_offset).float().to(self.device)
        return offset + scale * x_t[0, :, :ACTION_DIM_ENV].float()
