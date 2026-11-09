"""Predeclare a small, stratified LIBERO-Plus pilot without observing outcomes."""

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from dqat.runio import file_sha256, write_json

PLUS_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
ARMS = ("fp", "awq", "offline_gptq", "student", "expert_control")
CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)
PROTOCOL = dict(
    name="libero-plus-long-pilot-v1",
    suite="libero_10",
    cases=70,
    trials_per_case=1,
    seed=7,
    seed_rule="7 + official zero-based task ID; reseed before construction and reset",
    initial_state_id=0,
    environment_image_resolution=256,
    settle_steps=10,
    max_action_steps=520,
    open_loop_steps=8,
    center_crop=True,
    num_images=2,
    use_proprio=True,
    instruction_source="BDDL language_instruction from constructed environment",
    calibration="reuse nominal artifacts unchanged; no Plus observations used for fitting",
    plus_revision=PLUS_REVISION,
    asset_revision=ASSET_REVISION,
)


def task_map(path):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing literal libero_task_map")


def make_manifest(plus_root, nominal_root):
    plus_root, nominal_root = Path(plus_root), Path(nominal_root)
    relative = "libero/libero/benchmark"
    classification_path = plus_root / relative / "task_classification.json"
    classified = json.loads(classification_path.read_text())["libero_10"]
    names = task_map(plus_root / relative / "libero_suite_task_map.py")["libero_10"]
    bases = task_map(nominal_root / relative / "libero_suite_task_map.py")["libero_10"]
    if len(classified) != 2519 or len(names) != 2519 or len(bases) != 10:
        raise ValueError("Unexpected suite size")
    for entry in classified:
        if names[entry["id"] - 1] != entry["name"]:
            raise ValueError("Classification ID does not match task order")
    cases = []
    for base_id, base in enumerate(bases):
        for category_id, category in enumerate(CATEGORIES):
            candidates = [
                e for e in classified if e["category"] == category and e["name"].startswith(base + "_")
            ]
            # Cover each of the five official noise operators twice, including glass blur.
            if category == "Sensor Noise":
                noise_type = base_id % 5
                candidates = [
                    e for e in candidates if (int(e["name"].rsplit("_noise_", 1)[1]) - 1) // 10 == noise_type
                ]
            target = 1 + (base_id + category_id) % 5

            def ranking(entry):
                digest = hashlib.sha256(f"7:{entry['name']}".encode()).hexdigest()
                return abs(entry["difficulty_level"] - target), digest

            selected = min(candidates, key=ranking)
            cases.append(
                dict(
                    case_index=len(cases),
                    task_id=selected["id"] - 1,
                    task=selected["name"],
                    base_task_id=base_id,
                    base_task=base,
                    category=category,
                    difficulty_level=selected["difficulty_level"],
                    target_difficulty=target,
                    initial_state_id=0,
                    seed=7 + selected["id"] - 1,
                )
            )
    manifest = dict(
        protocol=PROTOCOL,
        classification_sha256=file_sha256(classification_path),
        selection="One variant per base task/category; rotating target difficulty 1..5, nearest available, SHA256 seed-7 tie break. Noise operators balanced two each.",
        cases=cases,
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest["protocol"] != PROTOCOL or len(manifest["cases"]) != 70:
        raise ValueError("Unexpected Plus pilot protocol")
    cases = manifest["cases"]
    if [c["case_index"] for c in cases] != list(range(70)) or len({c["task_id"] for c in cases}) != 70:
        raise ValueError("Duplicate or unordered case identities")
    if Counter((c["base_task_id"], c["category"]) for c in cases) != Counter(
        {(b, cat): 1 for b in range(10) for cat in CATEGORIES}
    ):
        raise ValueError("Pilot must cover every task/category stratum once")
    if any(c["seed"] != 7 + c["task_id"] or c["initial_state_id"] != 0 for c in cases):
        raise ValueError("Pilot seed or state differs")


def select_cases(manifest, shard, num_shards, smoke=False):
    validate_manifest(manifest)
    if not 0 <= shard < num_shards:
        raise ValueError("Invalid shard")
    if smoke:
        if shard != 0 or num_shards != 1:
            raise ValueError("Smoke is a separate single shard")
        selected = []
        for category in CATEGORIES:
            eligible = [c for c in manifest["cases"] if c["category"] == category]
            # Include the slow glass corruption in the runtime check.
            if category == "Sensor Noise":
                eligible = [c for c in eligible if int(c["task"].rsplit("_noise_", 1)[1]) > 40]
            selected.append(eligible[0])
        return selected
    return [c for c in manifest["cases"] if c["case_index"] % num_shards == shard]


RAPID_PROTOCOL = dict(
    PROTOCOL,
    name="libero-plus-long-rapid14-v1",
    cases=14,
    hardware="Available full Ampere GPUs: A100, A40 or RTX A6000; record each arm's hardware",
    selection="For category index c, take base tasks (2*c)%10 and (2*c+1)%10 from the frozen 70-case manifest",
)

OSMESA_LIBRARY_SHA256 = "8e8ad6bc3f6d443555d495505e8f2b2f8a07fcfef27cf0f7538ad2c82b328f1c"
OSMESA_RAPID_PROTOCOL = dict(
    RAPID_PROTOCOL,
    name="libero-plus-long-rapid14-osmesa-v1",
    hardware="Matched NVIDIA A100 MIG 3g.20gb across all arms",
    rendering=dict(
        backend="osmesa",
        driver="llvmpipe",
        mesa="23.1.4-4.el8_10",
        library_sha256=OSMESA_LIBRARY_SHA256,
        threads=4,
    ),
)


def select_rapid_cases(manifest):
    validate_manifest(manifest)
    return [
        c
        for c in manifest["cases"]
        if c["base_task_id"]
        in ((2 * CATEGORIES.index(c["category"])) % 10, (2 * CATEGORIES.index(c["category"]) + 1) % 10)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plus-root", type=Path, required=True)
    parser.add_argument("--nominal-root", type=Path, default=Path("third_party/LIBERO"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Manifest already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, make_manifest(args.plus_root, args.nominal_root))
