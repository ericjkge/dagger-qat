# ruff: noqa: E501  (embedded shell/markdown)
"""Build notebooks/e1_colab.ipynb from scripts/e1_activation_shift.py (single source of truth).

Run after editing the experiment script:  uv run python scripts/build_e1_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OPENPI_SHA = "215abfb217dbac7d5f1273282331b9b1866c0479"
CKPT_REPO, CKPT_REV = "sunshk/pi05_libero_pytorch", "7f62302dc2543281ff4626cf12bb484111b7ed7d"

INTRO = """# E1: activation shift in π0.5 under image perturbations (free Colab T4)

**Before running:** Runtime → Change runtime type → **T4 GPU**. Then **Run All**.

1. **Setup** (~10 min): openpi env from its lockfile + transformers patch + π0.5 LIBERO weights.
2. **Write the experiment script** (generated from `scripts/e1_activation_shift.py`, don't edit here).
3. **Run**: a 2-task smoke test first (~3 min), then the full run (40 tasks; time per frame is printed).

When it finishes, save this notebook (Cmd+S) so the printed results are kept in the file."""

SETUP = f"""%%bash
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | sed -n 1,2p; df -h /content | tail -1
command -v uv >/dev/null || pip install -q uv
cd /content
if [ ! -d openpi ]; then
  git init -q openpi && cd openpi
  git fetch -q --depth 1 https://github.com/Physical-Intelligence/openpi.git {OPENPI_SHA}
  git checkout -q FETCH_HEAD && cd ..
fi
cd openpi
# Colab's own Python (3.13) has no wheels for some openpi deps (e.g. labmaze): force uv-managed 3.11.
export GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy UV_PYTHON=3.11 UV_PYTHON_PREFERENCE=only-managed
uv python install 3.11
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 11)"; then
  echo "removing .venv built with the wrong Python"; rm -rf .venv
fi
uv sync -q --frozen --python 3.11  # exact versions from openpi's uv.lock; also installs openpi itself
.venv/bin/python -c "import sys, torch, transformers; print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
TF=$(.venv/bin/python -c "import os, transformers; print(os.path.dirname(transformers.__file__))")
cp -r src/openpi/models_pytorch/transformers_replace/* "$TF/"
.venv/bin/python -c "from transformers.models.siglip import check; \\
assert check.check_whether_transformers_replace_is_installed_correctly(); print('transformers patch OK')"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('{CKPT_REPO}', \\
revision='{CKPT_REV}', local_dir='/content/ckpt', allow_patterns=['model.safetensors', 'assets/*'])"
ls -la /content/ckpt && echo SETUP DONE"""

RUN = """%%bash
set -euo pipefail
cd /content/openpi
PY=".venv/bin/python /content/e1_activation_shift.py --ckpt /content/ckpt --cache /content/hf_cache"
$PY --smoke --out /content/results/e1_smoke
echo; echo "================ SMOKE OK, FULL RUN ================"; echo
$PY --out /content/results/e1
.venv/bin/python - <<'PYEOF'
import json
s = json.load(open("/content/results/e1/summary.json"))
s.pop("layers")
print("SUMMARY_JSON_BEGIN"); print(json.dumps(s)); print("SUMMARY_JSON_END")
PYEOF
cd /content/results && zip -qr /content/e1_results.zip e1 && echo "zipped -> /content/e1_results.zip"
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


script = (ROOT / "scripts" / "e1_activation_shift.py").read_text()
nb = {
    "cells": [
        cell("markdown", INTRO),
        cell("code", SETUP),
        cell("code", "%%writefile /content/e1_activation_shift.py\n" + script),
        cell("code", RUN),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "e1_colab.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}")
