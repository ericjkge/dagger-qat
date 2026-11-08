"""Record the exact inputs a policy is queried on, one compressed .npz per episode.

Runs in the Python 3.8 simulator env. The model env reads these files with
`vlacal.calib.frames` to build calibration sets (clean, perturbed, on-policy).
Use with `rollout.run_episode(..., on_query=recorder.add)`.

File contents (format_version 1):
  image, wrist_image  uint8 [T, H, W, 3]   exactly what was sent to the policy
  state               float32 [T, 8]
  step                int32 [T]            env step index of each policy query
  prompt              str                  instruction sent to the policy
  meta                str (JSON)           suite, task, initial state, seed, perturbation,
                                           policy metadata, success, ...
"""

import json
import os
import tempfile

import numpy as np

FORMAT_VERSION = 1


class EpisodeRecorder:
    def __init__(self):
        self.images, self.wrists, self.states, self.steps = [], [], [], []
        self.prompt = None

    def add(self, observation, step):
        prompt = str(observation["prompt"])
        if self.prompt is None:
            self.prompt = prompt
        elif prompt != self.prompt:
            raise ValueError("prompt changed within an episode")
        image = np.asarray(observation["observation/image"])
        wrist = np.asarray(observation["observation/wrist_image"])
        for name, img in (("image", image), ("wrist_image", wrist)):
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"{name} must be uint8 HWC, got {img.dtype} {img.shape}")
        self.images.append(image.copy())
        self.wrists.append(wrist.copy())
        self.states.append(np.asarray(observation["observation/state"], dtype=np.float32))
        self.steps.append(int(step))

    def __len__(self):
        return len(self.steps)

    def save(self, path, meta):
        """Write atomically, so a crashed run never leaves a truncated file behind."""
        if not self.steps:
            raise ValueError("no policy queries were recorded")
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                image=np.stack(self.images),
                wrist_image=np.stack(self.wrists),
                state=np.stack(self.states),
                step=np.asarray(self.steps, dtype=np.int32),
                prompt=np.array(self.prompt),
                meta=np.array(json.dumps(meta, sort_keys=True)),
                format_version=np.array(FORMAT_VERSION),
            )
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
