"""Unquantized OpenVLA-OFT with the official LIBERO preprocessing and L1 head."""

import os
from pathlib import Path

import numpy as np
from filelock import FileLock


class OFTPolicy:
    def __init__(self, checkpoint, seed=7):
        from dqat.runtime import prepare_runtime

        root = prepare_runtime()
        import torch
        from experiments.robot.libero import run_libero_eval as upstream

        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA-OFT evaluation requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The reference baseline requires a GPU with BF16 support")
        upstream.set_seed_everywhere(seed)
        self.upstream = upstream
        self.cfg = upstream.GenerateConfig(
            pretrained_checkpoint=str(Path(checkpoint).resolve()),
            task_suite_name="libero_10",
            seed=seed,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            center_crop=True,
            num_open_loop_steps=8,
            load_in_8bit=False,
            load_in_4bit=False,
        )
        # Upstream synchronizes checkpoint modeling files relative to ./prismatic.
        # Keep that official behavior while allowing launch from our repository root.
        previous_directory = Path.cwd()
        try:
            # Both arms may start on different nodes. Upstream edits config/modeling
            # files during loading, so serialize that operation on shared storage.
            with FileLock(str(Path(self.cfg.pretrained_checkpoint) / ".dqat-load.lock"), timeout=1800):
                os.chdir(root / "third_party/openvla-oft")
                (
                    self.model,
                    self.action_head,
                    self.proprio_projector,
                    self.noisy_action_projector,
                    self.processor,
                ) = upstream.initialize_model(self.cfg)
        finally:
            os.chdir(previous_directory)
        # OFT predicts an entire action chunk in one forward; no generation KV cache is needed.
        self.model.language_model.config.use_cache = False
        self.resize_size = upstream.get_image_resize_size(self.cfg)

    def predict_prepared(self, observation, instruction):
        # get_vla_action normalizes state in place. Never mutate a calibration sample.
        observation = {key: np.array(value, copy=True) for key, value in observation.items()}
        return self.upstream.get_action(
            self.cfg,
            self.model,
            observation,
            instruction,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=False,
        )

    def predict(self, observation, instruction):
        prepared, _ = self.upstream.prepare_observation(observation, self.resize_size)
        actions = np.asarray(self.predict_prepared(prepared, instruction), dtype=np.float32)
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected a finite (8, 7) action chunk, got {actions.shape}")
        return actions
