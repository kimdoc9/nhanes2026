#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import requests

PIPELINE_DIR = Path(__file__).resolve().parent / "research" / "bidsleep_phase4_6"
sys.path.insert(0, str(PIPELINE_DIR))
import run_pipeline as locked  # noqa: E402

p = locked.p
ARCHIVE_URL = "https://physionet.org/content/bidsleep-dataset/get-zip/1.0.0/"
ARCHIVE_SIZE = 6_354_227_911


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int, block_size: int = 8 << 20, max_blocks: int = 8):
        super().__init__()
        self.url = url
        self.size = int(size)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.position = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": p.USER_AGENT, "Accept-Encoding": "identity"})
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.requests = 0
        self.bytes_downloaded = 0
        self.cache_hits = 0
        self.retries = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"Unsupported whence {whence}")
        if target < 0:
            raise ValueError("Negative seek position")
        self.position = min(int(target), self.size)
        return self.position

    def _fetch_block(self, block_index: int) -> bytes:
        if block_index in self.cache:
            self.cache_hits += 1
            data = self.cache.pop(block_index)
            self.cache[block_index] = data
            return data
        start = block_index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        expected = end - start + 1
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 900),
                )
                self.requests += 1
                if response.status_code != 206:
                    raise RuntimeError(f"Expected HTTP 206, received {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.endswith(f"/{self.size}"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                data = response.content
                if len(data) != expected:
                    raise RuntimeError(f"Range length mismatch {start}-{end}: expected={expected}, observed={len(data)}")
                self.bytes_downloaded += len(data)
                self.cache[block_index] = data
                while len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)
                return data
            except Exception as exc:
                last_error = exc
                self.retries += 1
                if attempt == 8:
                    break
                time.sleep(min(10 * attempt, 60))
        raise RuntimeError(f"Unable to fetch archive block {block_index} ({start}-{end}): {last_error}")

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(int(size), self.size - self.position)
        if size <= 0:
            return b""
        output = bytearray()
        remaining = size
        while remaining:
            block_index = self.position // self.block_size
            block = self._fetch_block(block_index)
            offset = self.position - block_index * self.block_size
            take = min(remaining, len(block) - offset)
            if take <= 0:
                raise RuntimeError("Invalid range-reader block boundary")
            output.extend(block[offset : offset + take])
            self.position += take
            remaining -= take
        return bytes(output)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def stats(self) -> dict[str, Any]:
        return {
            "archive_url": self.url,
            "archive_size": self.size,
            "block_size": self.block_size,
            "http_range_requests": self.requests,
            "http_range_bytes_downloaded": self.bytes_downloaded,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
        }

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            super().close()


def member_map(archive: zipfile.ZipFile, manifest) -> dict[str, str]:
    members = [name for name in archive.namelist() if not name.endswith("/")]
    mapping: dict[str, str] = {}
    for relative_path in manifest["path"].tolist():
        matches = [name for name in members if name == relative_path or name.endswith("/" + relative_path)]
        if len(matches) != 1:
            raise RuntimeError(f"Archive member mapping failure for {relative_path}: {matches[:10]}")
        mapping[relative_path] = matches[0]
    if len(mapping) != p.EXPECTED_DATA_FILES:
        raise RuntimeError(f"Mapped {len(mapping)} files instead of {p.EXPECTED_DATA_FILES}")
    return mapping


def install_remote_archive(output: Path):
    manifest = p.load_manifest()
    raw = HTTPRangeReader(ARCHIVE_URL, ARCHIVE_SIZE)
    buffered = io.BufferedReader(raw, buffer_size=8 << 20)
    archive = zipfile.ZipFile(buffered, "r")
    mapping = member_map(archive, manifest)

    def extract_verified(url: str, destination: Path, expected_sha256: str, retries: int = 1) -> dict[str, Any]:
        del retries
        relative_path = url.split(p.BASE_URL.rstrip("/") + "/", 1)[-1]
        member = mapping.get(relative_path)
        if member is None:
            raise RuntimeError(f"No remote archive member for {relative_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        started = time.time()
        with archive.open(member, "r") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)
        observed = p.sha256(partial)
        if observed != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Remote ZIP member SHA-256 mismatch for {relative_path}: expected={expected_sha256}, observed={observed}"
            )
        os.replace(partial, destination)
        return {
            "source": "official_zip_http_range",
            "archive_url": ARCHIVE_URL,
            "archive_member": member,
            "bytes": destination.stat().st_size,
            "sha256": observed,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    p.download_verified = extract_verified
    return manifest, raw, buffered, archive, mapping


def write_stats(path: Path, raw: HTTPRangeReader, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p.json_safe({**raw.stats(), **extra}), indent=2, sort_keys=True, allow_nan=False))


def preflight(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest, raw, buffered, archive, mapping = install_remote_archive(out)
    try:
        record = p.process_night(manifest, "Bidslab10", 1, out)
        write_stats(out / "remote_zip_preflight.json", raw, {
            "status": "success",
            "mapped_data_files": len(mapping),
            "zip_members": len(archive.namelist()),
            "representative_qc": record,
        })
    finally:
        archive.close()
        buffered.close()


def shard(index: int, count: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest, raw, buffered, archive, mapping = install_remote_archive(out)
    try:
        p.shard(index, count, out)
        write_stats(out / "remote_zip_stats.json", raw, {
            "status": "success",
            "shard_index": index,
            "shard_count": count,
            "mapped_data_files": len(mapping),
        })
    except BaseException:
        write_stats(out / "remote_zip_stats.json", raw, {
            "status": "failure",
            "shard_index": index,
            "shard_count": count,
            "mapped_data_files": len(mapping),
        })
        raise
    finally:
        archive.close()
        buffered.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("preflight")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("shard")
    b.add_argument("--index", type=int, required=True)
    b.add_argument("--count", type=int, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.out)
    else:
        shard(args.index, args.count, args.out)


if __name__ == "__main__":
    main()
