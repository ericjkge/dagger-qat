"""Bounded, task-balanced replay for the fixed two-round GPTQ experiment."""

import copy
import json
from pathlib import Path

import numpy as np

from dqat.runio import file_sha256, write_json

DEMO_QUOTAS = (2, 2, 2, 2, 2, 2, 1, 1, 1, 1)
ROLLOUT_QUOTAS = (2, 2, 1, 1, 1, 1, 2, 2, 2, 2)
PROTOCOL = {
    "name": "gptq-data-aggregation-v1",
    "rounds": 2,
    "calibration_observations": 32,
    "demo_anchors": 16,
    "rollout_observations": 16,
    "episodes_per_task_per_round": 2,
    "reservoir_per_task_per_round": 8,
    "collection_state_source": "fresh seeded resets, excluding evaluation qpos",
    "collection_seed_base": 100000,
    "selection_seed": 7,
    "teacher_action_labels_for_update": 0,
    "update": "rebuild unchanged GPTQ from original BF16 checkpoint each round",
    "primary_evaluation": "round 2, all 500 original paired episodes",
}


def validate_observation(observation):
    for name in ("full_image", "wrist_image"):
        x = np.asarray(observation[name])
        if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 1:
            raise ValueError("Calibration images must be uint8 RGB arrays")
    state = np.asarray(observation["state"])
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Calibration state must be finite and eight-dimensional")


class Reservoir:
    """Uniform stream reservoir with its own RNG, independent of simulator RNGs."""

    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError("Reservoir capacity must be positive")
        self.capacity, self.seen, self.items = capacity, 0, []
        self.rng = np.random.default_rng(seed)

    def add(self, item):
        self.seen += 1
        index = self.seen - 1 if self.seen <= self.capacity else int(self.rng.integers(self.seen))
        if index < self.capacity:
            item = copy.deepcopy(item)
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                self.items[index] = item


def assert_disjoint_state(initial_state, excluded_states, nq):
    """Ignore MuJoCo time/velocity: reject matching physical initial configurations."""
    initial, excluded = np.asarray(initial_state), np.asarray(excluded_states)
    if initial.ndim != 1 or excluded.ndim != 2 or nq < 1:
        raise ValueError("Malformed initial-state exclusion inputs")
    if initial.size < nq + 1 or excluded.shape[1] < nq + 1 or len(excluded) < 1:
        raise ValueError("Missing qpos values in initial-state exclusion inputs")
    if not np.isfinite(initial).all() or not np.isfinite(excluded).all():
        raise ValueError("Nonfinite simulator initial state")
    matches = np.isclose(excluded[:, 1 : nq + 1], initial[1 : nq + 1], rtol=0, atol=1e-8).all(1)
    if matches.any():
        raise ValueError("Collection initial state overlaps an excluded evaluation/collection state")


def save_observations(directory, samples, manifest, provenance):
    directory = Path(directory)
    if not samples or len(samples) != len(manifest):
        raise ValueError("Observation and manifest counts differ or are empty")
    directory.mkdir(parents=True, exist_ok=False)
    arrays, records = {}, []
    for i, ((observation, instruction), record) in enumerate(zip(samples, manifest)):
        validate_observation(observation)
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("Missing task instruction")
        for key in ("full_image", "wrist_image", "state"):
            arrays[f"{key}_{i}"] = np.asarray(observation[key])
        records.append(dict(record, instruction=instruction))
    path = directory / "observations.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata = dict(
        format_version=1,
        status="completed",
        count=len(samples),
        samples=records,
        provenance=provenance,
        observations_sha256=file_sha256(path),
    )
    write_json(directory / "manifest.json", metadata)
    return metadata


def load_observations(directory):
    directory = Path(directory)
    manifest_path, path = directory / "manifest.json", directory / "observations.npz"
    metadata = json.loads(manifest_path.read_text())
    if metadata["format_version"] != 1 or metadata["status"] != "completed":
        raise ValueError("Incomplete or unsupported observation bundle")
    if metadata["count"] != len(metadata["samples"]) or metadata["count"] < 1:
        raise ValueError("Invalid observation bundle count")
    if file_sha256(path) != metadata["observations_sha256"]:
        raise ValueError("Observation bundle digest mismatch")
    samples = []
    with np.load(path, allow_pickle=False) as arrays:
        expected = {
            f"{key}_{i}" for i in range(metadata["count"]) for key in ("full_image", "wrist_image", "state")
        }
        if set(arrays.files) != expected:
            raise ValueError("Observation bundle arrays do not match its manifest")
        for i, record in enumerate(metadata["samples"]):
            observation = {key: arrays[f"{key}_{i}"] for key in ("full_image", "wrist_image", "state")}
            validate_observation(observation)
            samples.append((observation, record["instruction"]))
    identity = dict(
        manifest_sha256=file_sha256(manifest_path),
        observations_sha256=metadata["observations_sha256"],
        provenance=metadata["provenance"],
    )
    return samples, metadata["samples"], identity


def build_mixture(
    demo_samples, demo_manifest, pool_directories, output, arm, checkpoint_source, dataset_source
):
    if len(demo_samples) != 32 or len(demo_manifest) != 32 or len(pool_directories) not in (1, 2):
        raise ValueError("This protocol requires 32 original frames and one or two collection rounds")
    if arm not in ("student", "expert_control"):
        raise ValueError("Unknown loop arm")
    pools = [load_observations(path) for path in pool_directories]
    for round_id, (_, records, identity) in enumerate(pools, 1):
        p = identity["provenance"]
        if (
            p["arm"] != arm
            or p["round"] != round_id
            or p["checkpoint_source"] != checkpoint_source
            or p["protocol"] != PROTOCOL
            or p["status"] != "completed"
        ):
            raise ValueError("Collection pool lineage or protocol mismatch")
        if [sum(r["task_id"] == t for r in records) for t in range(10)] != [8] * 10:
            raise ValueError("Collection pool must contain eight observations per task")
    rng = np.random.default_rng(7)
    samples, manifest, rollout_index = [], [], 0
    for task in range(10):
        anchors = [i for i, r in enumerate(demo_manifest) if r["task_id"] == task][: DEMO_QUOTAS[task]]
        if len(anchors) != DEMO_QUOTAS[task]:
            raise ValueError("Not enough original demo anchors")
        for i in anchors:
            samples.append(demo_samples[i])
            manifest.append(dict(demo_manifest[i], source="demo", original_calibration_index=i))
        selected = set()
        for _ in range(ROLLOUT_QUOTAS[task]):
            pool_index = rollout_index % len(pools)
            rollout_index += 1
            observations, records, _ = pools[pool_index]
            candidates = [
                i for i, r in enumerate(records) if r["task_id"] == task and (pool_index, i) not in selected
            ]
            i = int(rng.choice(candidates))
            selected.add((pool_index, i))
            if observations[i][1] != demo_samples[anchors[0]][1]:
                raise ValueError("Rollout instruction does not match the original task")
            samples.append(observations[i])
            manifest.append(dict(records[i], source="rollout", pool_index=pool_index))
    provenance = dict(
        protocol=PROTOCOL,
        arm=arm,
        round=len(pools),
        checkpoint_source=checkpoint_source,
        dataset_source=dataset_source,
        pools=[identity for _, _, identity in pools],
        original_demo_manifest=demo_manifest,
        status="completed",
    )
    return save_observations(output, samples, manifest, provenance)


def load_calibration_bundle(directory, checkpoint_source, dataset_source, count):
    samples, manifest, identity = load_observations(directory)
    p = identity["provenance"]
    if (
        len(samples) != count
        or count != 32
        or p["protocol"] != PROTOCOL
        or p["checkpoint_source"] != checkpoint_source
        or p["dataset_source"] != dataset_source
        or p["arm"] not in ("student", "expert_control")
        or p["round"] not in (1, 2)
        or p["status"] != "completed"
        or len(p["pools"]) != p["round"]
    ):
        raise ValueError("Calibration bundle does not match the experiment/checkpoint/data")
    if sum(r["source"] == "demo" for r in manifest) != 16:
        raise ValueError("Calibration bundle must retain exactly 16 demo anchors")
    if sum(r["source"] == "rollout" for r in manifest) != 16:
        raise ValueError("Calibration bundle must contain exactly 16 rollout observations")
    if [sum(r["task_id"] == t for r in manifest) for t in range(10)] != [4, 4] + [3] * 8:
        raise ValueError("Calibration bundle task balance differs from baseline")
    rollout = [r for r in manifest if r["source"] == "rollout"]
    identities = {(r["round"], r["task_id"], r["episode"], r["action_step"]) for r in rollout}
    if len(identities) != 16:
        raise ValueError("Repeated rollout observation in calibration bundle")
    expected = [16] if p["round"] == 1 else [8, 8]
    if [sum(r["round"] == i + 1 for r in rollout) for i in range(p["round"])] != expected:
        raise ValueError("Calibration bundle does not preserve the pre-set round mixture")
    return samples, manifest, identity
