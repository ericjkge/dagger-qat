"""Read individual tensors from a Hugging Face safetensors checkpoint with HTTP range requests.

The OFT checkpoint (15 GB) does not fit on the dev machine, so decoder layers are streamed one at a
time (about 400 MB each) and discarded after use.
"""

from __future__ import annotations

import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64}
CHUNK = 16 * 2**20


class RemoteSafetensors:
    def __init__(self, repo: str, revision: str, index_path: Path, threads: int = 8):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        self.headers: dict[str, tuple[dict, int]] = {}
        self.session = requests.Session()
        self.threads = threads
        self.bytes_read = 0

    def _range(self, shard: str, start: int, end: int) -> bytes:
        """Inclusive byte range, with retries."""
        for attempt in range(8):
            try:
                r = self.session.get(
                    self.base + shard, headers={"Range": f"bytes={start}-{end}"}, timeout=120, allow_redirects=True
                )
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    return r.content
                raise OSError(f"HTTP {r.status_code}, {len(r.content)} bytes")
            except (OSError, requests.RequestException):
                if attempt == 7:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self.headers:
            (n,) = struct.unpack("<Q", self._range(shard, 0, 7))
            header = json.loads(self._range(shard, 8, 8 + n - 1))
            self.headers[shard] = (header, 8 + n)
        return self.headers[shard]

    def get(self, names: list[str]) -> dict[str, torch.Tensor]:
        jobs = []  # (name, shard, start, end, dtype, shape)
        for name in names:
            shard = self.weight_map[name]
            header, data_start = self._header(shard)
            meta = header[name]
            begin, stop = meta["data_offsets"]
            jobs.append((name, shard, data_start + begin, data_start + stop - 1, DTYPES[meta["dtype"]], meta["shape"]))
        pieces = [
            (name, shard, s, min(s + CHUNK - 1, end))
            for name, shard, start, end, _, _ in jobs
            for s in range(start, end + 1, CHUNK)
        ]
        with ThreadPoolExecutor(self.threads) as pool:
            data = list(pool.map(lambda p: self._range(p[1], p[2], p[3]), pieces))
        buffers: dict[str, list[bytes]] = {}
        for (name, *_), chunk in zip(pieces, data):
            buffers.setdefault(name, []).append(chunk)
        out = {}
        for name, _, start, end, dtype, shape in jobs:
            raw = bytearray(b"".join(buffers[name]))
            if len(raw) != end - start + 1:
                raise OSError(f"{name}: short read")
            self.bytes_read += len(raw)
            out[name] = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
        return out
