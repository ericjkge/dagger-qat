import json
from pathlib import Path

import pytest

from dqat.eval_plus import checkpoint_view
from dqat.plus_data import (
    ARMS,
    CATEGORIES,
    PROTOCOL,
    RAPID_PROTOCOL,
    select_cases,
    select_rapid_cases,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / "configs/libero-plus-long-pilot-v1.json").read_text())


def test_pilot_shards_partition_every_stratum_and_preserve_seeds():
    m = manifest()
    shards = [select_cases(m, s, 4) for s in range(4)]
    assert sorted(c["case_index"] for shard in shards for c in shard) == list(range(70))
    assert sorted(map(len, shards)) == [17, 17, 18, 18]
    noise = [int(c["task"].rsplit("_noise_", 1)[1]) for c in m["cases"] if c["category"] == "Sensor Noise"]
    assert sorted((n - 1) // 10 for n in noise) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    smoke = select_cases(m, 0, 1, True)
    assert [c["category"] for c in smoke] == list(CATEGORIES)
    assert int(smoke[-1]["task"].rsplit("_noise_", 1)[1]) > 40


def test_rejects_duplicate_cases_or_changed_seeds():
    m = manifest()
    m["cases"][1]["task_id"] = m["cases"][0]["task_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(m)
    m = manifest()
    m["cases"][1]["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        validate_manifest(m)


def test_asset_depot_preserves_bytes_and_materializes_only_needed_files(tmp_path):
    import hashlib
    import zipfile

    from dqat.plus_assets import AssetDepot

    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("published/nested/assets/textures/a.png", b"texture bytes")
        z.writestr("published/nested/assets/new_objects/a/mesh.obj", b"mesh bytes")
    with zipfile.ZipFile(archive) as z:
        depot = AssetDepot(z, tmp_path / "assets")
        depot.common_assets()
        assert (tmp_path / "assets/textures/a.png").read_bytes() == b"texture bytes"
        mesh = tmp_path / "assets/new_objects/a/mesh.obj"
        assert not mesh.exists()
        depot.ensure(mesh)
        assert mesh.read_bytes() == b"mesh bytes"
        assert depot.extracted["new_objects/a/mesh.obj"] == hashlib.sha256(b"mesh bytes").hexdigest()
        with pytest.raises(FileNotFoundError):
            depot.ensure(tmp_path / "assets/missing.xml")


def test_asset_depot_rejects_archive_path_traversal(tmp_path):
    import zipfile

    from dqat.plus_assets import AssetDepot

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("published/assets/../../escape", b"bad")
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match="Unsafe"):
        AssetDepot(z, tmp_path / "assets")


def test_checkpoint_views_isolate_mutable_loader_files(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    (source / "config.json").write_text('{"model": "original"}')
    (source / "modeling_prismatic.py").write_text("original = True\n")
    (source / "model.safetensors").write_bytes(b"immutable weights")
    (source / ".dqat-load.lock").touch()
    a, b = (checkpoint_view(source, tmp_path / name) for name in ("a", "b"))
    (a / "config.json").write_text("changed")
    (a / "modeling_prismatic.py").write_text("changed")
    assert (b / "config.json").read_text() == (source / "config.json").read_text()
    assert (b / "modeling_prismatic.py").read_text() == (source / "modeling_prismatic.py").read_text()
    assert (a / "model.safetensors").is_symlink()
    assert (a / "model.safetensors").resolve() == (source / "model.safetensors").resolve()
    assert not (a / ".dqat-load.lock").exists()


def write_synthetic_pilot(root, rapid=False):
    from dqat.runio import file_sha256, write_json

    m = manifest()
    manifest_path = ROOT / "configs/libero-plus-long-pilot-v1.json"
    for index, arm in enumerate(ARMS):
        for shard in range(1 if rapid else 4):
            cases = select_rapid_cases(m) if rapid else select_cases(m, shard, 4)
            output = root / arm / f"shard-{shard}"
            output.mkdir(parents=True)
            meta = dict(
                status="completed",
                arm=arm,
                smoke=False,
                shard=shard,
                num_shards=1 if rapid else 4,
                expected_cases=[c["task_id"] for c in cases],
                protocol=RAPID_PROTOCOL if rapid else PROTOCOL,
                manifest_sha256=file_sha256(manifest_path),
                checkpoint_source={"fixture": True},
                normalization_sha256="b" * 64,
                git_commit="fixture",
                git_status="",
                submodules="fixture",
                gpu="fixture",
                torch_version="fixture",
            )
            if arm != "fp":
                digest = str(index) * 64
                signature = dict(
                    cache_sha256=digest,
                    checkpoint_source=meta["checkpoint_source"],
                    normalization_sha256=meta["normalization_sha256"],
                    config={"bits": 4},
                )
                if arm in ("student", "expert_control"):
                    signature["calibration_bundle"] = {"provenance": {"arm": arm, "round": 2}}
                meta["quantization"] = dict(
                    method="awq" if arm == "awq" else "gptq", cache_sha256=digest, signature=signature
                )
            rows = [
                dict(
                    c,
                    arm=arm,
                    status="completed",
                    success=(c["case_index"] + index) % 5 != 0,
                    action_steps=520,
                    policy_calls=65,
                    elapsed_seconds=1,
                    simulator_warning=c["case_index"] == 0,
                    simulator_warning_counts=[int(c["case_index"] == 0)],
                    instruction_sha256="c" * 64,
                    initial_state_sha256="d" * 64,
                )
                for c in cases
            ]
            write_json(output / "run.json", meta)
            write_json(
                output / "summary.json",
                dict(
                    status="completed",
                    arm=arm,
                    expected_episodes=len(rows),
                    completed_episodes=len(rows),
                    errors=0,
                    successes=sum(r["success"] for r in rows),
                    warning_episodes=sum(r["simulator_warning"] for r in rows),
                ),
            )
            (output / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path


def test_aggregation_preserves_pairs_and_excludes_warning_union(tmp_path):
    from dqat.compare_plus import compare_pilot

    report = compare_pilot(tmp_path, write_synthetic_pilot(tmp_path))
    assert all(x["successes"] == 56 and x["episodes"] == 70 for x in report["methods"].values())
    pair = report["comparisons"]["awq_minus_fp"]
    assert pair["difference_percentage_points"] == 0
    assert pair["candidate_only_successes"] == pair["reference_only_successes"] == 14
    assert report["simulator_warning_sensitivity"]["excluded_union_cases"] == 1
    assert report["simulator_warning_sensitivity"]["methods"]["fp"]["successes"] == 56
    assert report["simulator_warning_sensitivity"]["methods"]["awq"]["successes"] == 55
    assert all(x["episodes"] == 10 for c in report["per_category"].values() for x in c.values())


def test_rapid_subset_covers_all_categories_and_tasks_and_is_reported_separately(tmp_path):
    from collections import Counter

    from dqat.compare_plus import compare_pilot

    cases = select_rapid_cases(manifest())
    assert len(cases) == 14
    assert Counter(c["category"] for c in cases) == Counter({cat: 2 for cat in CATEGORIES})
    assert {c["base_task_id"] for c in cases} == set(range(10))
    path = write_synthetic_pilot(tmp_path, rapid=True)
    report = compare_pilot(tmp_path, path, rapid=True)
    assert report["protocol"] == RAPID_PROTOCOL
    assert all(x["episodes"] == 14 for x in report["methods"].values())
    with pytest.raises(ValueError):
        compare_pilot(tmp_path, path)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "seed", "instruction", "artifact"])
def test_aggregation_rejects_incomplete_or_unpaired_results(tmp_path, corruption):
    from dqat.compare_plus import compare_pilot

    path = write_synthetic_pilot(tmp_path)
    target = tmp_path / "student/shard-0/episodes.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "seed":
        rows[0]["seed"] += 1
    elif corruption == "instruction":
        rows[0]["instruction_sha256"] = "e" * 64
    else:
        meta_path = target.parent / "run.json"
        meta = json.loads(meta_path.read_text())
        meta["quantization"]["cache_sha256"] = "f" * 64
        meta_path.write_text(json.dumps(meta))
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        compare_pilot(tmp_path, path)
