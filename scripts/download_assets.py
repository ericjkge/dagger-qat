"""Download pinned official weights and training demonstrations (no evaluation rollouts)."""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "moojink/openvla-7b-oft-finetuned-libero-10"
MODEL_REVISION = "95220f9a3421a7ff12d4218e73d09ade830fa9a3"
DATASET_ID = "yifengzhu-hf/LIBERO-datasets"
DATASET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


if __name__ == "__main__":
    root = Path(os.environ.get("DQAT_DATA_DIR", "data"))
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=root / "checkpoints/openvla-oft-libero-10",
        ignore_patterns=["lora_adapter/*"],
        max_workers=4,
    )
    (root / "checkpoints/openvla-oft-libero-10/source.json").write_text(
        json.dumps({"model_id": MODEL_ID, "model_revision": MODEL_REVISION}, indent=2) + "\n"
    )
    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=root / "datasets",
        allow_patterns=["libero_10/*.hdf5"],
        max_workers=4,
    )
    (root / "datasets/source.json").write_text(
        json.dumps({"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION}, indent=2) + "\n"
    )
    (root / "asset-provenance.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
            },
            indent=2,
        )
        + "\n"
    )
