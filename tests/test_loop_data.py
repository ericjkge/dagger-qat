import json
from types import SimpleNamespace

import numpy as np
import pytest

from dqat.collect_libero import collect_episode
from dqat.loop_data import (
    PROTOCOL,
    Reservoir,
    assert_disjoint_state,
    build_mixture,
    load_calibration_bundle,
    load_observations,
    save_observations,
)


def observation(size=3, value=0):
    return dict(
        full_image=np.full((size, size, 3), value, dtype=np.uint8),
        wrist_image=np.full((size, size, 3), value + 1, dtype=np.uint8),
        state=np.arange(8, dtype=float),
    )


def test_reservoir_is_deterministic_copies_data_and_preserves_global_rng():
    a, b = Reservoir(8, 7), Reservoir(8, 7)
    np.random.seed(42)
    expected = np.random.random(4)
    np.random.seed(42)
    for i in range(100):
        data = dict(index=i, array=np.array([i]))
        a.add(data)
        b.add(data)
        data["array"][0] = -1
    assert [x["index"] for x in a.items] == [x["index"] for x in b.items]
    assert len(set(x["index"] for x in a.items)) == 8
    assert max(x["index"] for x in a.items) >= 8
    assert all(x["array"][0] == x["index"] for x in a.items)
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_collection_split_ignores_time_and_velocity():
    heldout = np.array([[0.0, 1.0, 2.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint_state([20.0, 1.0, 2.0, 9.0, 8.0], heldout, 2)
    assert_disjoint_state([0.0, 1.1, 2.0, 0.0, 0.0], heldout, 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        assert_disjoint_state([0.0, np.nan, 2.0, 0.0, 0.0], heldout, 2)


def test_bundle_preserves_different_image_sizes_and_detects_corruption(tmp_path):
    samples = [(observation(3), "one"), (observation(7), "two")]
    directory = tmp_path / "bundle"
    save_observations(directory, samples, [dict(task_id=0), dict(task_id=1)], {})
    restored, _, _ = load_observations(directory)
    for original, new in zip(samples, restored):
        for key in original[0]:
            np.testing.assert_array_equal(original[0][key], new[0][key])
        assert original[1] == new[1]
    with (directory / "observations.npz").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="digest"):
        load_observations(directory)


def make_pool(path, iteration, arm="student"):
    samples, manifest = [], []
    for task in range(10):
        for i in range(8):
            samples.append((observation(4, i), f"task {task}"))
            manifest.append(dict(task_id=task, round=iteration, episode=i // 4, action_step=(i % 4) * 8))
    save_observations(
        path,
        samples,
        manifest,
        dict(
            arm=arm,
            round=iteration,
            checkpoint_source={"revision": "test"},
            protocol=PROTOCOL,
            status="completed",
        ),
    )


def test_two_round_mixture_retains_anchors_aggregates_and_matches_budget(tmp_path):
    demos, records = [], []
    for task in range(10):
        for frame in range(4 if task < 2 else 3):
            demos.append((observation(), f"task {task}"))
            records.append(dict(task_id=task, frame=frame))
    pools = [tmp_path / "pool1", tmp_path / "pool2"]
    for iteration, path in enumerate(pools, 1):
        make_pool(path, iteration)
    manifests = []
    for iteration in (1, 2):
        output = tmp_path / f"mix{iteration}"
        build_mixture(demos, records, pools[:iteration], output, "student", {"revision": "test"}, {})
        samples, manifest, _ = load_calibration_bundle(output, {"revision": "test"}, {}, 32)
        assert len(samples) == 32
        assert [sum(r["task_id"] == t for r in manifest) for t in range(10)] == [4, 4] + [3] * 8
        manifests.append(manifest)
    assert [r for r in manifests[0] if r["source"] == "demo"] == [
        r for r in manifests[1] if r["source"] == "demo"
    ]
    rollout = [r for r in manifests[1] if r["source"] == "rollout"]
    assert sum(r["round"] == 1 for r in rollout) == sum(r["round"] == 2 for r in rollout) == 8
    assert len({(r["round"], r["task_id"], r["episode"], r["action_step"]) for r in rollout}) == 16
    with pytest.raises(ValueError, match="lineage"):
        build_mixture(demos, records, pools[::-1], tmp_path / "bad", "student", {"revision": "test"}, {})
    with pytest.raises(ValueError, match="experiment/checkpoint"):
        load_calibration_bundle(tmp_path / "mix2", {"revision": "wrong"}, {}, 32)
    meta_path = tmp_path / "mix2/manifest.json"
    meta = json.loads(meta_path.read_text())
    meta["samples"][0]["source"] = "rollout"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="16 demo"):
        load_calibration_bundle(tmp_path / "mix2", {"revision": "test"}, {}, 32)


class Environment:
    def __init__(self, warning_at=None, done_at=20):
        self.actions, self.warning_at, self.done_at = [], warning_at, done_at
        self.sim = SimpleNamespace(
            model=SimpleNamespace(nq=1), data=SimpleNamespace(warning=[SimpleNamespace(number=0)])
        )

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.actions = []
        return observation()

    def get_sim_state(self):
        return np.array([len(self.actions), 5.0, 0.0])

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == self.warning_at:
            self.sim.data.warning[0].number = 1
        return observation(), 0, len(self.actions) == self.done_at, {}


class Policy:
    resize_size = 3
    upstream = SimpleNamespace(
        set_seed_everywhere=lambda seed: None,
        prepare_observation=lambda obs, size: (obs, None),
        process_action=lambda a, family: a,
    )

    def predict_prepared(self, obs, instruction):
        return np.tile(np.arange(8)[:, None], (1, 7)).astype(float)


def test_collection_fresh_seeds_chunk_boundaries_and_failure_retention():
    env, reservoir = Environment(), Reservoir(8, 7)
    record = collect_episode(env, Policy(), "task", 0, 1, 0, [[0.0, 99.0, 0.0]], reservoir)
    assert env.seed_value == 110000
    assert record["success"] and record["action_steps"] == 10 and record["policy_calls"] == 2
    assert [x[1]["action_step"] for x in reservoir.items] == [0, 8]
    assert [a[0] for a in env.actions[10:]] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    failed = Reservoir(8, 7)
    record = collect_episode(Environment(done_at=None), Policy(), "task", 0, 1, 1, [[0.0, 99.0, 0.0]], failed)
    assert not record["success"] and record["action_steps"] == 520 and record["policy_calls"] == 65
    assert failed.seen == 65 and len(failed.items) == 8


def test_collection_rejects_overlap_before_actions_and_stops_on_warning():
    env = Environment()
    with pytest.raises(ValueError, match="overlaps"):
        collect_episode(env, Policy(), "task", 0, 1, 0, [[99.0, 5.0, 7.0]], Reservoir(8, 7))
    assert env.actions == []
    with pytest.raises(RuntimeError, match="MuJoCo warning"):
        collect_episode(
            Environment(warning_at=12), Policy(), "task", 0, 1, 0, [[0.0, 99.0, 0.0]], Reservoir(8, 7)
        )


def test_collection_rejects_invalid_action_before_execution():
    class Bad(Policy):
        def predict_prepared(self, obs, instruction):
            return np.full((8, 7), np.nan)

    env = Environment()
    with pytest.raises(ValueError, match="finite"):
        collect_episode(env, Bad(), "task", 0, 1, 0, [[0.0, 99.0, 0.0]], Reservoir(8, 7))
    assert len(env.actions) == 10
