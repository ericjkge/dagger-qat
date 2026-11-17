"""Render every frame of the offline experiment and compute OFT's decoder-layer-0 input for it.

Outputs (in --out):
  manifest.json     frames (role, state, condition, LIBERO-Plus variant, sequence length, action slice)
  inputs.f16        concatenated layer-0 inputs, float16 [sum(T), 4096]; offsets in manifest
  states.npz        per state: demo actions (next 8), proprio
  thumbs/           prepared model images for a sample of frames (sanity figure)
"""

import argparse
import gc
import resource
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dqat.offline import lp_variants as lpv
from dqat.offline.frames import build_manifest, load_states
from dqat.offline.model import REPO, REVISION, OFTEnds
from dqat.offline.remote import RemoteSafetensors
from dqat.offline.scenes import Renderer, apply_camera, apply_scene_patch, localize, sensor_noise

TASKS = [
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


def scene_patches(pools: dict, asset_dir: Path) -> dict:
    """{scene_xml: patch} for every light/background variant in the pools, with textures fetched."""
    from remotezip import RemoteZip

    wanted = sorted(
        {v["scene_xml"] for t in pools.values() for c in ("light", "background") for h in t[c].values() for v in h}
    )
    xml_cache = asset_dir / "scene_xml"
    xml_cache.mkdir(parents=True, exist_ok=True)
    needed = wanted + sorted(set(lpv.BASE_SCENE_XML.values()))
    missing = [p for p in needed if not (xml_cache / p.replace("/", "__")).exists()]
    if missing:
        with RemoteZip(lpv.ASSETS_URL) as archive:
            for path in missing:
                (xml_cache / path.replace("/", "__")).write_bytes(archive.read(lpv.ASSET_PREFIX + path))
    read = lambda p: (xml_cache / p.replace("/", "__")).read_text()  # noqa: E731
    patches = {}
    for path in wanted:
        scene = next(k for k in lpv.BASE_SCENE_XML if path.split("/")[-1].startswith(_scene_prefix(k)))
        patches[path] = lpv.scene_patch(read(lpv.BASE_SCENE_XML[scene]), read(path), path)
    return patches


def _scene_prefix(scene: str) -> str:
    return {"kitchen": "kitchen", "living_room": "liv", "study": "study"}[scene]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=Path("data/offline/demos"))
    parser.add_argument("--lp", type=Path, default=Path("data/offline/lp"))
    parser.add_argument("--ckpt", type=Path, default=Path("data/offline/ckpt"))
    parser.add_argument("--out", type=Path, default=Path("data/offline/run"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "thumbs").mkdir(exist_ok=True)

    pools = lpv.build_pools(args.lp, TASKS)
    states = {s["key"]: s for s in load_states(args.demos)}
    frames = build_manifest(list(states.values()), pools)
    patches = scene_patches(pools, args.lp / "assets")
    textures = sorted({a["file"] for p in patches.values() for a in p["texture"].values() if "file" in a})
    used_textures = sorted(
        {
            a["file"]
            for f in frames
            if f["variant"] and "scene_xml" in f["variant"]
            for a in patches[f["variant"]["scene_xml"]]["texture"].values()
            if "file" in a
        }
    )
    print(f"{len(frames)} frames; textures referenced by pools {len(textures)}, used {len(used_textures)}", flush=True)
    asset_files = lpv.fetch_assets(used_textures, args.lp / "assets" / "files")

    remote = RemoteSafetensors(REPO, REVISION, args.ckpt / "model.safetensors.index.json")
    ends = OFTEnds(args.ckpt, remote)
    renderers: dict[str, Renderer] = {}
    offset = 0
    store = open(args.out / "inputs.f16", "wb")
    render_check = {}
    for f in frames:
        s = states[f["state"]]
        xml = localize(s["xml"])
        instruction = s["language"]
        cond, var = f["condition"], f["variant"]
        if cond == "camera":
            xml = apply_camera(xml, var["view"])
        elif cond in ("light", "background"):
            xml, applied = apply_scene_patch(xml, patches[var["scene_xml"]], asset_files)
            f["patch_applied"] = applied
            if applied == 0:
                raise RuntimeError(f"variant {var['name']} changes nothing in this scene")
        elif cond == "language":
            instruction = var["text"]
        key = hashlib.sha256(xml.encode()).hexdigest()
        if key not in renderers:
            while len(renderers) >= 2:  # each MuJoCo renderer holds a GL context and textures
                renderers.pop(next(iter(renderers))).close()
            gc.collect()
            renderers[key] = Renderer(xml, 256)
        images = renderers[key].render(s["sim_state"])
        full = images["agentview"]
        if cond == "noise":
            full = sensor_noise(full, var["noise"], seed=f["id"])
        if cond == "clean" and f["role"] == "eval":
            small = Renderer(xml, 128)
            ref = small.render(s["sim_state"])["agentview"]
            small.close()
            render_check[f["state"]] = float(np.abs(ref.astype(float) - s["hdf5_agentview"]).mean())
        prepared = ends.prepare(full, images["robot0_eye_in_hand"], s["proprio"], instruction)
        h, action_start = ends.layer0_input(prepared)
        if not torch.isfinite(h).all():
            raise RuntimeError(f"frame {f['id']}: nonfinite layer-0 input")
        arr = h.to(torch.float16).cpu().numpy()
        store.write(arr.tobytes())
        f.update(offset=offset, length=int(arr.shape[0]), action_start=int(action_start), instruction=instruction)
        offset += arr.shape[0]
        if f["role"] == "eval" and f["state"].endswith(tuple("0123456789")) and f["id"] % 7 == 0:
            Image.fromarray(np.concatenate([prepared["primary_image"], prepared["wrist_image"]], 1)).save(
                args.out / "thumbs" / f"{f['id']:03d}_{cond}.png"
            )
        print(f"frame {f['id']:3d} {f['role']:5s} {cond:10s} T={arr.shape[0]} maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f}GB", flush=True)
    store.close()
    np.savez(
        args.out / "states.npz",
        keys=np.array(list(states)),
        demo_actions=np.stack([s["demo_actions"] for s in states.values()]),
        proprio=np.stack([s["proprio"] for s in states.values()]),
    )
    manifest = dict(frames=frames, total_tokens=offset, render_check_mae=render_check, tasks=TASKS)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    worst = max(render_check.values())
    print(f"clean re-render vs recorded 128px images: mean abs err max {worst:.2f} (0-255 scale)")
    if worst > 10:
        raise RuntimeError("re-rendered clean frames do not match the recorded images")


if __name__ == "__main__":
    main()
