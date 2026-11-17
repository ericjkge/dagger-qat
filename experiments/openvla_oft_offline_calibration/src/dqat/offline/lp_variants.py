"""LIBERO-Plus perturbation variants for the 10 LIBERO-Long tasks, as data we can render offline.

Source: github.com/sylvestf/LIBERO-plus (task_classification.json, BDDL files, problem classes) and
the scene XMLs/textures in HF dataset Sylvest/LIBERO-plus (assets.zip, read by range requests).

Only perturbations that leave the physical state unchanged are used, so every perturbed frame is
paired with a clean frame of the *same* simulator state:
camera viewpoint, lighting, background texture, sensor noise (image corruption), language.
Robot-initial-state and object-layout perturbations change the physics and are excluded.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CATEGORIES = {
    "Camera Viewpoints": "camera",
    "Light Conditions": "light",
    "Background Textures": "background",
    "Sensor Noise": "noise",
    "Language Instructions": "language",
}
ASSET_PREFIX = (
    "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/"
)
ASSETS_URL = "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
SCENE_FILES = {"KITCHEN": "kitchen", "LIVING": "living_room", "STUDY": "study"}
BASE_SCENE_XML = {
    "kitchen": "scenes/libero_kitchen_tabletop_base_style.xml",
    "living_room": "scenes/libero_living_room_tabletop_base_style.xml",
    "study": "scenes/libero_study_base_style.xml",
}
# LIBERO-Plus noise ids 1-10 are ImageMagick motion blur (needs the Wand C library); we use the
# other four corruption families (gaussian blur, zoom blur, fog, glass blur).
NOISE_MIN_ID = 11


def scene_of(task: str) -> str:
    return SCENE_FILES[task.split("_")[0]]


def problem_scene_xml(problem_sources: dict[str, str], problem: str) -> str:
    for source in problem_sources.values():
        match = re.search(
            rf"class {re.escape(problem)}\(.*?\n(.*?)(?=\n@register_problem|\nclass |\Z)", source, re.S
        )
        if match:
            xml = re.search(r'"scene_xml":\s*"([^"]+)"', match.group(1))
            if xml:
                return xml.group(1)
    raise KeyError(f"no scene_xml for problem {problem}")


def build_pools(lp_dir: Path, tasks: list[str], seed: int = 0) -> dict:
    """{task: {category: {"calib": [...], "eval": [...]}}} with disjoint, seeded halves."""
    classification = json.loads((lp_dir / "task_classification.json").read_text())["libero_10"]
    problems = {k: (lp_dir / f"problems_{k}.py").read_text() for k in ("kitchen", "living_room", "study")}
    pools: dict = {t: defaultdict(list) for t in tasks}
    for item in classification:
        category = CATEGORIES.get(item["category"])
        if category is None:
            continue
        name = item["name"]
        base = next((t for t in tasks if name.startswith(t + "_")), None)
        if base is None:
            continue
        variant = {"name": name, "level": item["difficulty_level"]}
        if category == "camera":
            view = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_0$", name)
            if view is None:
                continue
            variant["view"] = [int(v) for v in view.groups()]
        elif category == "noise":
            noise = int(re.search(r"_noise_(\d+)$", name).group(1))
            if noise < NOISE_MIN_ID:
                continue
            variant["noise"] = noise
        else:
            bddl = (lp_dir / "bddl" / f"{name}.bddl").read_text()
            if category == "language":
                variant["text"] = re.search(r"\(:language (.*?)\)\s*$", bddl, re.M).group(1).strip()
            else:
                problem = re.search(r"\(define \(problem (\S+)\)", bddl).group(1)
                variant["scene_xml"] = problem_scene_xml(problems, problem)
        pools[base][category].append(variant)
    rng = np.random.default_rng(seed)
    out = {}
    for task in tasks:
        out[task] = {}
        for category in CATEGORIES.values():
            items = sorted(pools[task][category], key=lambda v: v["name"])
            if len(items) < 2:
                raise ValueError(f"{task}: too few {category} variants ({len(items)})")
            order = rng.permutation(len(items))
            half = len(items) // 2
            out[task][category] = {
                "calib": [items[i] for i in order[:half]],
                "eval": [items[i] for i in order[half:]],
            }
    return out


def xml_elements(xml: str, tag: str) -> dict[str, dict[str, str]]:
    found = {}
    for element in re.findall(rf"<{tag}\b[^>]*>", xml):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', element))
        if "name" in attrs:
            found[attrs["name"]] = attrs
    return found


def scene_patch(base_xml: str, variant_xml: str, variant_path: str) -> dict:
    """Attribute changes from the base scene to a variant scene, for lights and textures.

    Texture files are returned as asset paths (relative to the assets root) so they can be fetched.
    """
    patch = {"light": {}, "texture": {}}
    base_dir = str(Path(variant_path).parent)
    for tag in ("light", "texture"):
        base, var = xml_elements(base_xml, tag), xml_elements(variant_xml, tag)
        for name, attrs in var.items():
            if name not in base:
                continue
            changed = {k: v for k, v in attrs.items() if base[name].get(k) != v}
            if tag == "texture" and "file" in changed:
                changed["file"] = _normalize(Path(base_dir) / attrs["file"])
            if changed:
                patch[tag][name] = changed
    return patch


def _normalize(path: Path) -> str:
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            parts.pop()
        elif part not in (".", ""):
            parts.append(part)
    return "/".join(parts)


def fetch_assets(paths: list[str], out_dir: Path) -> dict[str, Path]:
    """Download asset files (relative to the LIBERO-Plus assets root) from assets.zip by range reads."""
    from remotezip import RemoteZip

    out_dir.mkdir(parents=True, exist_ok=True)
    local = {p: out_dir / p for p in paths}
    missing = [p for p, q in local.items() if not q.exists()]
    if missing:
        with RemoteZip(ASSETS_URL) as archive:
            for path in missing:
                local[path].parent.mkdir(parents=True, exist_ok=True)
                tmp = local[path].with_suffix(local[path].suffix + ".part")
                tmp.write_bytes(archive.read(ASSET_PREFIX + path))
                tmp.rename(local[path])
    return local
