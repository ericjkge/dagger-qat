"""Offline kill-check (vlacal.eval.offline) end to end on the fake π0.5."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from fake_pi05 import D_VIS, D_VLM, PATCHES, PROMPT, FakePolicy

from vlacal.eval.offline import Plan, format_summary, run_plan

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("recording", ROOT / "sim/recording.py")
recording = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recording)


class ImagePolicy:
    """Adapts recorded frames (uint8 images, prompt) to the fake model's inputs."""

    def __init__(self):
        self.inner = FakePolicy()
        self._model = self.inner._model

    def infer(self, obs, noise):
        def feats(img):
            return (img.reshape(-1)[: PATCHES * D_VIS].astype(np.float32) / 64.0).reshape(
                PATCHES, D_VIS
            )

        seed = sum(map(ord, obs["prompt"]))
        lang = np.random.default_rng(seed).normal(size=(PROMPT, D_VLM)).astype(np.float32)
        o = {
            "observation/image": feats(obs["observation/image"]),
            "observation/wrist_image": feats(obs["observation/wrist_image"]),
            "lang": lang,
        }
        a = self.inner.infer(o, noise=noise[:5, :4])["actions"]
        return {"actions": np.concatenate([a, a[:, :3]], axis=1)}  # (5, 7)


def _episodes(directory, n, seed, bright=1.0):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for e in range(n):
        rec = recording.EpisodeRecorder()
        for t in range(3):
            img = np.clip(rng.integers(0, 120, (8, 8, 3)) * bright, 0, 255).astype(np.uint8)
            rec.add(
                {
                    "observation/image": img,
                    "observation/wrist_image": img[::-1].copy(),
                    "observation/state": np.zeros(8),
                    "prompt": f"task {e % 2}",
                },
                5 * t,
            )
        rec.save(str(directory / f"ep{e}.npz"), {"split": directory.name})
    return str(directory / "*.npz")


def _plan(tmp_path):
    clean = _episodes(tmp_path / "clean", 4, 0)
    test = _episodes(tmp_path / "test", 3, 1, bright=2.0)
    sq = {"a_bits": 4}  # SmoothQuant W4A4 static: strongly calibration dependent
    return {
        "eval_sets": {"test": {"patterns": [test], "per_episode": None}},
        "reference": "q_clean",
        "arms": [
            {"name": "q_clean", "config": sq, "calib": {"patterns": [clean]}},
            {"name": "o2_test_states", "config": sq, "calib": {"patterns": [test]}},
            {
                "name": "q_dynamic",
                "config": {**sq, "a_mode": "dynamic"},
                "calib": {"patterns": [clean]},
            },
            {
                "name": "fp_noop",
                "config": {"w_bits": None, "a_bits": None, "a_mode": None, "smooth_alpha": None},
            },
        ],
    }


def test_run_plan_end_to_end(tmp_path):
    plan = Plan.from_dict(_plan(tmp_path))
    res = run_plan(ImagePolicy(), plan, log=lambda m: None)
    s = res["summary"]["test"]
    assert set(s) == {"q_clean", "o2_test_states", "q_dynamic", "fp_noop"}
    assert s["q_clean"]["recovered"] == 0 and s["q_clean"]["delta_vs_ref"][0] == 0
    assert s["fp_noop"]["l1"] < 1e-5 and s["fp_noop"]["recovered"] == pytest.approx(1, abs=1e-3)
    assert s["q_clean"]["l1"] > 0
    assert res["determinism_max_abs"] == 0
    assert res["arms"]["o2_test_states"]["calib_frames"] == 9
    assert res["actions"]["fp"]["test"].shape == (9, 5, 7)
    json.dumps({k: v for k, v in res.items() if k != "actions"})
    assert "o2_test_states" in format_summary(res["summary"], "q_clean")


def test_plan_validation(tmp_path):
    d = _plan(tmp_path)
    bad = {**d, "reference": "missing"}
    with pytest.raises(ValueError):
        Plan.from_dict(bad)
    d["arms"][1] = {"name": "x", "config": {"a_bits": 8}}  # needs calibration, has none
    with pytest.raises(ValueError, match="needs calibration"):
        Plan.from_dict(d)


def test_shipped_plans_parse(monkeypatch):
    monkeypatch.chdir(ROOT)
    for path in sorted((ROOT / "configs/killcheck").glob("*.json")):
        plan = Plan.load(path)  # config paths are relative to the repo root
        assert plan.reference == "q_clean", path


def test_baseline_and_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    d = _plan(tmp_path)
    d["arms"].append({"name": "rtn", "config": "configs/quant/rtn_w4.json", "baseline": "fp_noop"})
    plan = Plan.from_dict(d)
    assert plan.arms[-1].config.w_method == "rtn"
    res = run_plan(ImagePolicy(), plan, log=lambda m: None)
    assert res["summary"]["test"]["rtn"]["baseline"] == "fp_noop"
    d["arms"][-1]["baseline"] = "nope"
    with pytest.raises(ValueError, match="baseline"):
        Plan.from_dict(d)
