"""Download and verify the official pinned LIBERO-Plus asset archive."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import yaml
from filelock import FileLock
from huggingface_hub import hf_hub_download

REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
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
    identity = dict(dataset="Sylvest/LIBERO-plus", revision=REVISION, sha256=SHA256)
    if not marker.exists():
        with zipfile.ZipFile(archive) as bundle:
            required_bytes = sum(member.file_size for member in bundle.infolist())
            if shutil.disk_usage(root).free < required_bytes * 1.1 + (2 << 30):
                raise RuntimeError("Insufficient space for Plus assets plus extraction headroom")
            print(f"Extracting {len(bundle.infolist())} entries, {required_bytes / 1e9:.2f} GB", flush=True)
            for member in bundle.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root) or not member.filename.startswith("assets/"):
                    raise ValueError(f"Unexpected archive path: {member.filename}")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Symlinks are not expected in the asset archive")
            bundle.extractall(root)
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    elif json.loads(marker.read_text()) != identity:
        raise ValueError("Previously extracted assets have a different identity")
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
    print("Verified and extracted official Plus assets", identity, flush=True)


if __name__ == "__main__":
    main()
