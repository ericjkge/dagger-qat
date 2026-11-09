import json

import pytest

from dqat.summarize import compare, read_run


def make_run(path, method, outcomes):
    path.mkdir()
    metadata = dict(
        status="completed",
        method=method,
        checkpoint_source={"revision": "test"},
        suite="libero_10",
        seed=7,
        settle_steps=10,
        max_action_steps=520,
        open_loop_steps=8,
        center_crop=True,
        num_images=2,
        environment_seed=0,
        environment_image_resolution=256,
        use_proprio=True,
        submodules="same pins",
    )
    (path / "run.json").write_text(json.dumps(metadata))
    (path / "summary.json").write_text(json.dumps(dict(expected_episodes=2, completed_episodes=2)))
    records = [
        dict(task_id=0, initial_state_id=i, seed=7 + i, status="completed", success=success)
        for i, success in enumerate(outcomes)
    ]
    (path / "episodes.jsonl").write_text("\n".join(json.dumps(record) for record in records))


def test_paired_discordance(tmp_path):
    make_run(tmp_path / "fp", "fp", [True, False])
    make_run(tmp_path / "awq", "awq", [True, True])
    result = compare(tmp_path / "fp", tmp_path / "awq")
    assert result["awq_minus_fp_percentage_points"] == 50
    assert result["awq_only_successes"] == 1
    assert result["fp_only_successes"] == 0


def test_refuses_partial_and_duplicate_runs(tmp_path):
    run = tmp_path / "run"
    make_run(run, "fp", [True])
    with pytest.raises(ValueError, match="episode count"):
        read_run(run)
    record = (run / "episodes.jsonl").read_text()
    (run / "episodes.jsonl").write_text(record + "\n" + record)
    with pytest.raises(ValueError, match="Duplicate"):
        read_run(run)


@pytest.mark.parametrize("field", ["environment_seed", "environment_image_resolution"])
def test_refuses_different_simulator_protocol(tmp_path, field):
    make_run(tmp_path / "fp", "fp", [True, False])
    make_run(tmp_path / "awq", "awq", [True, True])
    path = tmp_path / "awq" / "run.json"
    metadata = json.loads(path.read_text())
    metadata[field] += 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        compare(tmp_path / "fp", tmp_path / "awq")
