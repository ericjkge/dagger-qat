"""Shared fixtures: the repository's paths, and a stand-in for the verl environment that records the command a step
script would run instead of running it."""

import os
import stat
import subprocess
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PEAR = ROOT / "third_party" / "pear"
VERL_SRC = PEAR / "3rdparty" / "verl" / "src"
TEACHER = "mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only"
TEACHER_REVISION = "63e96d966ad0af83f4eb63b79b94cadb06289bea"

# the stand-ins: "python -m <module> ..." and "torchrun ... -m <module> ..." write the module and its arguments,
# NUL-separated, to
# $OPD_CAPTURE; any other python call (PEAR's environment script reads sysconfig) goes to the real interpreter
STUB_PYTHON = """#!/usr/bin/env bash
if [ "$1" = -m ]; then shift; printf '%s\\0' "$@" > "$OPD_CAPTURE"; exit 0; fi
exec {python} "$@"
"""
STUB_TORCHRUN = """#!/usr/bin/env bash
while [ "$1" != -m ]; do shift; done
shift
printf '%s\\0' "$@" > "$OPD_CAPTURE"
"""


@pytest.fixture(scope="session")
def verl_env(tmp_path_factory) -> Path:
    env = tmp_path_factory.mktemp("verl_env")
    (env / "bin").mkdir()
    real_python = subprocess.run(["bash", "-c", "command -v python3"], capture_output=True, text=True).stdout.strip()
    for name, text in [("python", STUB_PYTHON.format(python=real_python)), ("torchrun", STUB_TORCHRUN)]:
        path = env / "bin" / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return env


@pytest.fixture
def run_step(verl_env, tmp_path):
    """run_step(script, overrides, **variables) -> (the entry point, the Hydra overrides the step would pass it)."""

    def run(script: str, overrides=(), **variables) -> tuple[str, list[str]]:
        capture = tmp_path / "capture"
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "WORK_DIR": str(tmp_path / "work"),
            "VERL_ENV": str(verl_env),
            "LOCAL_DIR": str(tmp_path / "local"),
            "RUN": str(tmp_path / "run"),
            "OPD_CAPTURE": str(capture),
            **{key: str(value) for key, value in variables.items()},
        }
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script), *overrides], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        module, flag, config_dir, *args = capture.read_bytes().decode().split("\0")[:-1]
        assert (flag, config_dir) == ("--config-dir", str(ROOT / "configs"))  # where the step overlays are
        return module, args

    return run


@pytest.fixture(scope="session")
def hub_file(tmp_path_factory):
    """hub_file(repo, revision, filename) -> the local copy of a pinned file of the Hugging Face Hub."""
    cache = tmp_path_factory.mktemp("hub")

    def fetch(repo: str, revision: str, filename: str) -> Path:
        path = cache / repo.replace("/", "--") / revision / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", path)
        return path

    return fetch
