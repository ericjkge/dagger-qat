import sys
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np

from dqat.calibration import load_demonstrations


def test_demo_preprocessing_and_balanced_sampling(tmp_path, monkeypatch):
    class Suite:
        n_tasks = 10

        def get_task(self, index):
            return SimpleNamespace(name=f"task{index}", language=f"task {index}")

    module = ModuleType("libero.libero")
    module.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": Suite})
    monkeypatch.setitem(sys.modules, "libero", ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    for task in range(10):
        with h5py.File(tmp_path / f"task{task}_demo.hdf5", "w") as file:
            obs = file.create_group("data/demo_0/obs")
            obs["agentview_rgb"] = image[None]
            obs["eye_in_hand_rgb"] = (image + 20)[None]
            obs["ee_states"] = np.arange(6)[None]
            obs["gripper_states"] = np.array([[6, 7]])
    samples, manifest = load_demonstrations(tmp_path, 13, seed=7)
    assert len(samples) == 13
    assert [sum(record["task_id"] == task for record in manifest) for task in range(10)] == [
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    np.testing.assert_array_equal(samples[0][0]["full_image"], image[::-1, ::-1])
    np.testing.assert_array_equal(samples[0][0]["wrist_image"], (image + 20)[::-1, ::-1])
    np.testing.assert_array_equal(samples[0][0]["state"], np.arange(8))
    assert manifest == load_demonstrations(tmp_path, 13, seed=7)[1]
