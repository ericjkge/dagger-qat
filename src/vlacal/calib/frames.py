"""Load recorded policy queries (sim/recording.py) as calibration or evaluation frames."""

from __future__ import annotations

import glob
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
PI05_LIBERO_NOISE_SHAPE = (10, 32)  # (action_horizon, model action dim) for pi05_libero


@dataclass
class Frame:
    obs: dict  # exactly the dict the policy was queried with
    source: str  # episode file
    index: int  # query index within the episode
    step: int  # env step of the query
    meta: dict  # episode metadata (suite, task, perturbation, policy, success, ...)

    @property
    def uid(self) -> str:
        return f"{Path(self.source).name}#{self.index}"


def episode_files(patterns: str | Sequence[str]) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    if not files:
        raise FileNotFoundError(f"no episode files match {list(patterns)}")
    return files


def load_frames(
    patterns: str | Sequence[str],
    *,
    per_episode: int | None = 4,
    max_frames: int | None = None,
    seed: int = 0,
) -> list[Frame]:
    """Evenly spaced queries from each episode, then (optionally) a seeded subset.

    Deterministic for a given set of files, so every arm of an experiment that uses the same
    source sees exactly the same frames.
    """
    frames = []
    for path in episode_files(patterns):
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported format {int(z['format_version'])}")
            n = len(z["step"])
            if per_episode is None or n <= per_episode:
                idx = np.arange(n)
            else:
                idx = np.unique(np.linspace(0, n - 1, per_episode).round().astype(int))
            prompt, meta = str(z["prompt"]), json.loads(str(z["meta"]))
            images, wrists, states, steps = z["image"], z["wrist_image"], z["state"], z["step"]
            for i in idx:
                obs = {
                    "observation/image": images[i],
                    "observation/wrist_image": wrists[i],
                    "observation/state": states[i],
                    "prompt": prompt,
                }
                frames.append(Frame(obs, path, int(i), int(steps[i]), meta))
    if max_frames is not None and len(frames) > max_frames:
        keep = np.sort(np.random.default_rng(seed).choice(len(frames), max_frames, replace=False))
        frames = [frames[i] for i in keep]
    return frames


def fixed_noise(i: int, shape=PI05_LIBERO_NOISE_SHAPE, seed: int = 0) -> np.ndarray:
    """Flow-matching noise for frame i: identical across arms, so comparisons are paired."""
    return np.random.default_rng([seed, i]).standard_normal(shape).astype(np.float32)


def calibration_runner(policy, frames: Sequence[Frame], *, weights=None, noise_shape=None):
    """`run_calibration` for `quantize_model`: query the FP policy on every frame.

    `weights` (optional, one per frame) set each frame's contribution to GPTQ Hessians.
    """
    shape = noise_shape or PI05_LIBERO_NOISE_SHAPE
    if weights is not None and len(weights) != len(frames):
        raise ValueError("need one weight per frame")

    def run(collector):
        for i, frame in enumerate(frames):
            collector.sample_weight = 1.0 if weights is None else float(weights[i])
            policy.infer(frame.obs, noise=fixed_noise(i, shape))
        collector.sample_weight = 1.0

    return run
