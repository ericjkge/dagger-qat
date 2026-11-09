"""Deterministic, task-balanced samples from official LIBERO training HDF5s."""

from pathlib import Path

import numpy as np


def load_demonstrations(dataset_dir, count=32, seed=7):
    import h5py
    from libero.libero import benchmark

    if count < 10:
        raise ValueError("Use at least 10 calibration frames to cover all LIBERO-Long tasks")
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    rng = np.random.default_rng(seed)
    samples, manifest = [], []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        path = Path(dataset_dir) / f"{task.name}_demo.hdf5"
        with h5py.File(path, "r") as data:
            demos = sorted(data["data"], key=lambda name: int(name.split("_")[-1]))
            for _ in range(count // suite.n_tasks + (task_id < count % suite.n_tasks)):
                demo_name = demos[int(rng.integers(len(demos)))]
                obs = data[f"data/{demo_name}/obs"]
                frame = int(rng.integers(len(obs["agentview_rgb"])))
                # HDF5 images have simulator orientation. RLDS and upstream evaluation
                # rotate BOTH views 180 degrees before resize/crop.
                front = np.array(obs["agentview_rgb"][frame])[::-1, ::-1].copy()
                wrist = np.array(obs["eye_in_hand_rgb"][frame])[::-1, ::-1].copy()
                state = np.concatenate((obs["ee_states"][frame], obs["gripper_states"][frame]))
                if state.shape != (8,) or not np.isfinite(state).all():
                    raise ValueError(f"Invalid 8D proprioception in {path}:{demo_name}:{frame}")
                samples.append((dict(full_image=front, wrist_image=wrist, state=state), task.language))
                manifest.append(dict(task_id=task_id, file=path.name, demo=demo_name, frame=frame))
    return samples, manifest
