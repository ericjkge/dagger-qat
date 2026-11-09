import json
from pathlib import Path

import pytest

from dqat.eval_plus import checkpoint_view
from dqat.plus_data import CATEGORIES, select_cases, validate_manifest

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
