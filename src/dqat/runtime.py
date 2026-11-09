"""Initialize the pinned runtime before importing upstream model/simulator code."""

import os
import sys
from pathlib import Path


def prepare_runtime():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "third_party/openvla-oft", root / "third_party/LIBERO"):
        if not path.is_dir():
            raise RuntimeError("Initialize submodules with git submodule update --init --recursive")
        sys.path.insert(0, str(path))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Upstream uses TensorFlow for image preprocessing only. Do not let it reserve GPU RAM.
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    return root
