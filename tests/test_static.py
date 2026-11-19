"""Lint: ruff on the Python, shellcheck and bash's parser on the shell scripts."""

import subprocess

import pytest
from conftest import ROOT

SHELL = sorted([*(ROOT / "scripts").glob("*.sh"), *(ROOT / "configs").rglob("*.sh")])


def run(*command: str) -> None:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ruff():
    run("ruff", "check", "src", "tests")
    run("ruff", "format", "--check", "src", "tests")


@pytest.mark.parametrize("script", SHELL, ids=lambda path: path.name)
def test_bash_parses(script):
    run("bash", "-n", str(script))


def test_shellcheck():
    # -x follows the sourced files; the model files source PEAR's through $PEAR, whose path is given in each file
    run("shellcheck", "-x", "-P", "SCRIPTDIR", *map(str, SHELL))
