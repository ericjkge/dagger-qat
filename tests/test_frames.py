"""Recording (sim side, Python 3.8 code) -> loading (model side) round trip."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from fake_pi05 import FakePolicy

from vlacal.calib.frames import calibration_runner, fixed_noise, load_frames
from vlacal.quant import QuantConfig, quantize_model

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


recording = _load("recording", "sim/recording.py")
rollout = _load("rollout", "sim/rollout.py")


def _obs(i, prompt="put the bowl on the plate"):
    rng = np.random.default_rng(i)
    return {
        "observation/image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
        "observation/state": rng.normal(size=8),
        "prompt": prompt,
    }


def _write(tmp_path, name, n, meta=None):
    rec = recording.EpisodeRecorder()
    for i in range(n):
        rec.add(_obs(i), step=5 * i)
    path = tmp_path / name
    rec.save(str(path), meta or {"suite": "libero_spatial", "perturbation": {"type": "camera"}})
    return path


def test_roundtrip_is_exact(tmp_path):
    path = _write(tmp_path, "ep0.npz", 3)
    frames = load_frames(str(path), per_episode=None)
    assert [f.step for f in frames] == [0, 5, 10]
    for i, f in enumerate(frames):
        o = _obs(i)
        assert np.array_equal(f.obs["observation/image"], o["observation/image"])
        assert f.obs["observation/state"].dtype == np.float32
        np.testing.assert_allclose(f.obs["observation/state"], o["observation/state"], rtol=1e-6)
        assert f.obs["prompt"] == o["prompt"]
    assert frames[0].meta["perturbation"]["type"] == "camera"


def test_sampling_is_even_and_deterministic(tmp_path):
    for e in range(5):
        _write(tmp_path, f"ep{e}.npz", 10)
    frames = load_frames(str(tmp_path / "*.npz"), per_episode=4)
    assert len(frames) == 20 and [f.index for f in frames[:4]] == [0, 3, 6, 9]
    a = load_frames(str(tmp_path / "*.npz"), per_episode=4, max_frames=7, seed=1)
    b = load_frames(str(tmp_path / "*.npz"), per_episode=4, max_frames=7, seed=1)
    assert [f.uid for f in a] == [f.uid for f in b] and len(a) == 7


def test_recorder_rejects_bad_input(tmp_path):
    rec = recording.EpisodeRecorder()
    with pytest.raises(ValueError):
        rec.save(str(tmp_path / "empty.npz"), {})
    rec.add(_obs(0), 0)
    with pytest.raises(ValueError, match="prompt"):
        rec.add(_obs(1, prompt="something else"), 5)
    bad = _obs(2)
    bad["observation/image"] = bad["observation/image"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        rec.add(bad, 10)
    with pytest.raises(FileNotFoundError):
        load_frames(str(tmp_path / "nothing*.npz"))


def test_run_episode_on_query_records_every_policy_call(tmp_path):
    class Env:
        def reset(self):
            pass

        def set_init_state(self, state):
            return self._obs()

        def step(self, action):
            return self._obs(), 0, False, {}

        def _obs(self):
            return {
                "agentview_image": np.zeros((8, 8, 3), np.uint8),
                "robot0_eye_in_hand_image": np.zeros((8, 8, 3), np.uint8),
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
            }

    class Client:
        def reset(self):
            pass

        def infer(self, obs):
            return {"actions": np.zeros((10, 7))}

    rec = recording.EpisodeRecorder()
    result = rollout.run_episode(
        Env(), Client(), "s", "go", lambda x: x, max_steps=12, on_query=rec.add
    )
    assert len(rec) == result["policy_calls"] == 3 and rec.steps == [0, 5, 10]
    rec.save(str(tmp_path / "ep.npz"), {"success": result["success"]})
    assert json.loads(str(np.load(tmp_path / "ep.npz")["meta"]))["success"] is False


def test_fixed_noise_is_paired_and_distinct():
    assert np.array_equal(fixed_noise(3), fixed_noise(3))
    assert not np.array_equal(fixed_noise(3), fixed_noise(4))
    assert fixed_noise(0).shape == (10, 32)


def test_calibration_runner_feeds_quantize_model(tmp_path):
    """End to end on the fake model: frames on disk -> runner -> calibrated quantization."""
    pol = FakePolicy()
    rng = np.random.default_rng(0)

    class Adapter:  # fake policy takes (PATCHES, D) arrays, not images; adapt the frame dict
        def infer(self, obs, noise):
            o = {
                "observation/image": rng.normal(size=(4, 8)).astype(np.float32),
                "observation/wrist_image": rng.normal(size=(4, 8)).astype(np.float32),
                "lang": rng.normal(size=(3, 12)).astype(np.float32),
            }
            return pol.infer(o, noise=noise[:5, :4])

    _write(tmp_path, "ep.npz", 4)
    frames = load_frames(str(tmp_path / "*.npz"), per_episode=None)
    run = calibration_runner(Adapter(), frames, weights=[1, 2, 1, 2])
    h = quantize_model(pol._model, QuantConfig(w_method="gptq"), run, log=lambda m: None)
    vlm = [r for n, r in h.report["layers"].items() if "language_model" in n]
    assert vlm and all(r["calib_tokens"] == 4 * 11 for r in vlm)  # 4 frames x 11 real tokens
    with pytest.raises(ValueError):
        calibration_runner(Adapter(), frames, weights=[1])
