"""Download and verify the official pinned LIBERO-Plus asset archive."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import yaml
from filelock import FileLock
from huggingface_hub import hf_hub_download

from dqat.plus_assets import AssetDepot
from dqat.plus_data import validate_manifest

REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/libero-plus-long-pilot-v1.json",
    )
    args = parser.parse_args()
    with FileLock(str(args.root / ".assets.lock"), timeout=3600):
        prepare_assets(args)


def prepare_assets(args):
    archive = Path(
        hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", revision=REVISION)
    )
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("Plus asset archive checksum mismatch")
    root = (args.root / "libero/libero").resolve()
    marker = args.root / "assets-source.json"
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    identity = dict(
        dataset="Sylvest/LIBERO-plus",
        revision=REVISION,
        sha256=SHA256,
        pilot_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    )
    config = args.data_dir / "libero-plus-config"
    config.mkdir(exist_ok=True)
    (config / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(root),
                "bddl_files": str(root / "bddl_files"),
                "init_states": str(root / "init_files"),
                "datasets": str(args.data_dir / "datasets"),
                "assets": str(root / "assets"),
            }
        )
    )
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Previously prepared assets have a different identity")
        return
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.root.resolve()))
    with zipfile.ZipFile(archive) as bundle:
        required_bytes = sum(member.file_size for member in bundle.infolist())
        if shutil.disk_usage(root).free < required_bytes * 1.1 + (2 << 30):
            raise RuntimeError("Insufficient space for Plus assets plus extraction headroom")
        depot = AssetDepot(bundle, root / "assets")
        print("Verified archive; preparing only the fixed pilot's required files", flush=True)
        depot.common_assets()
        depot.prepare_cases(manifest["cases"], root / "bddl_files")
        (args.root / "asset-files.json").write_text(json.dumps(depot.extracted, indent=2) + "\n")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("All pilot assets prepared from the verified official archive", identity, flush=True)


if __name__ == "__main__":
    main()
