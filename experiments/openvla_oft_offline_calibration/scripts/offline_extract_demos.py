"""Read the sampled simulator states from the remote LIBERO-Long demo HDF5s (range requests).

Selections (deterministic): disjoint demos per task (seed 2026), one state per demo at a uniformly
random time step. 4 eval states per task; calibration quotas
  [4, 4, 3, ...] = 32 states, matching the baseline's task balance.
Each saved state keeps the MuJoCo model XML, the flattened sim state, proprio (ee_states,
gripper_states), the next 8 demo actions and the recorded 128px images (for render validation).
"""

import argparse
import hashlib
import json
from pathlib import Path

import fsspec
import h5py
import numpy as np

REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
TASKS = [  # LIBERO benchmark order for libero_10 (task_order_index 0)
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]
CALIB_QUOTA = [4, 4, 3, 3, 3, 3, 3, 3, 3, 3]
EVAL_PER_TASK = 4


def demo_names(data):
    return sorted(data["data"], key=lambda name: int(name.split("_")[-1]))


def record(data, demo, frame, task_id, role):
    d = data[f"data/{demo}"]
    obs = d["obs"]
    actions = np.asarray(d["actions"])
    nxt = actions[frame : frame + 8]
    if len(nxt) < 8:
        nxt = np.concatenate([nxt, np.repeat(nxt[-1:], 8 - len(nxt), 0)])
    xml = d.attrs["model_file"]
    xml = xml.decode() if isinstance(xml, bytes) else xml
    return dict(
        role=role,
        task_id=task_id,
        task=TASKS[task_id],
        language=json.loads(data["data"].attrs["problem_info"])["language_instruction"],
        demo=demo,
        frame=int(frame),
        length=int(len(obs["agentview_rgb"])),
        xml_sha=hashlib.sha256(xml.encode()).hexdigest()[:16],
        sim_state=np.asarray(d["states"][frame], dtype=np.float64),
        ee_states=np.asarray(obs["ee_states"][frame], dtype=np.float64),
        gripper_states=np.asarray(obs["gripper_states"][frame], dtype=np.float64),
        demo_actions=nxt.astype(np.float32),
        hdf5_agentview=np.asarray(obs["agentview_rgb"][frame]),
        hdf5_wrist=np.asarray(obs["eye_in_hand_rgb"][frame]),
    ), xml


def our_draws():
    """Demo order and frame fractions for every task, drawn up front (all files have 50 demos)."""
    ours = np.random.default_rng(2026)
    draws = []
    for task_id in range(len(TASKS)):
        order = ours.permutation(50)
        fractions = [ours.random() for _ in range(EVAL_PER_TASK + CALIB_QUOTA[task_id])]
        draws.append((order, fractions))
    return draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    task_id, task = args.task, TASKS[args.task]
    done = args.out / f"task{task_id}.npz"
    if done.exists():
        return
    order, fractions = our_draws()[task_id]
    url = (
        f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{REVISION}"
        f"/libero_10/{task}_demo.hdf5"
    )
    records, xmls = [], {}
    # Read only the sampled rows over HTTP range requests (the files are ~1.3 GB each).
    remote = fsspec.open(url, block_size=2**20, cache_type="blockcache").open()
    with remote, h5py.File(remote, "r") as data:
        demos = demo_names(data)
        if len(demos) != 50:
            raise ValueError(f"expected 50 demos, found {len(demos)}")
        for j, i in enumerate(order[: len(fractions)]):
            demo = demos[int(i)]
            length = len(data[f"data/{demo}/obs/agentview_rgb"])
            frame = int(fractions[j] * length)
            role = "eval" if j < EVAL_PER_TASK else "calib"
            rec, xml = record(data, demo, frame, task_id, role)
            records.append(rec)
            xmls[rec["xml_sha"]] = xml
    arrays = {f"{k}_{i}": v for i, r in enumerate(records) for k, v in r.items() if isinstance(v, np.ndarray)}
    meta = dict(
        task_id=task_id,
        task=task,
        records=[{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in records],
        xmls=xmls,
    )
    tmp = done.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    tmp.rename(done)
    print(f"task {task_id}: {len(records)} states saved", flush=True)


if __name__ == "__main__":
    main()
