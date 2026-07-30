#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent / "research" / "bidsleep_phase4_6"
sys.path.insert(0, str(PIPELINE_DIR))
import run_pipeline as locked  # noqa: E402

p = locked.p

CONTENT_PAGE = "https://physionet.org/content/bidsleep-dataset/1.0.0/"
CANDIDATE_ARCHIVES = [
    "https://physionet.org/static/published-projects/bidsleep-dataset/bidsleep-dataset-1.0.0.zip",
    "https://physionet.org/files/bidsleep-dataset/1.0.0/bidsleep-dataset-1.0.0.zip",
    "https://physionet.org/files/bidsleep-dataset/1.0.0/bidsleep-dataset.zip",
    "https://physionet.org/files/bidsleep-dataset/1.0.0.zip",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p.json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def disk_state(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def discover_archive_urls() -> list[str]:
    urls: list[str] = []
    try:
        request = urllib.request.Request(CONTENT_PAGE, headers={"User-Agent": p.USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            html = response.read().decode("utf-8", errors="replace")
        for href in re.findall(r'href=["\']([^"\']+\.zip(?:\?[^"\']*)?)["\']', html, flags=re.IGNORECASE):
            urls.append(urllib.parse.urljoin(CONTENT_PAGE, href))
    except Exception:
        pass
    urls.extend(CANDIDATE_ARCHIVES)
    ordered: list[str] = []
    for url in urls:
        if url not in ordered:
            ordered.append(url)
    return ordered


def download_archive(destination: Path, diagnostics: Path) -> tuple[str, list[dict[str, Any]]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for url in discover_archive_urls():
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        started = time.time()
        command = [
            "curl", "--fail", "--location", "--retry", "12", "--retry-all-errors",
            "--retry-delay", "5", "--connect-timeout", "30", "--max-time", "18000",
            "--continue-at", "-", "--user-agent", p.USER_AGENT,
            "--output", str(partial), url,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        record = {
            "url": url,
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "bytes": partial.stat().st_size if partial.exists() else 0,
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode == 0 and zipfile.is_zipfile(partial):
            os.replace(partial, destination)
            record["valid_zip"] = True
            attempts.append(record)
            write_json(diagnostics, {"selected_url": url, "attempts": attempts})
            return url, attempts
        record["valid_zip"] = False
        attempts.append(record)
        partial.unlink(missing_ok=True)
    write_json(diagnostics, {"selected_url": None, "attempts": attempts})
    raise RuntimeError("No valid BID-Sleep ZIP archive URL could be downloaded")


def build_member_map(archive: zipfile.ZipFile, manifest) -> dict[str, str]:
    members = [name for name in archive.namelist() if not name.endswith("/")]
    mapping: dict[str, str] = {}
    for relative_path in manifest["path"].tolist():
        matches = [name for name in members if name == relative_path or name.endswith("/" + relative_path)]
        if len(matches) != 1:
            raise RuntimeError(f"Archive member mapping failure for {relative_path}: matches={matches[:10]}")
        mapping[relative_path] = matches[0]
    if len(mapping) != p.EXPECTED_DATA_FILES:
        raise RuntimeError(f"Archive mapping count failure: {len(mapping)}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "archive_diagnostics"
    diagnostics.mkdir(exist_ok=True)
    archive_path = output / "bidsleep-dataset-1.0.0.zip"
    write_json(diagnostics / "disk_before.json", disk_state(output))

    selected_url, attempts = download_archive(archive_path, diagnostics / "download.json")
    manifest = p.load_manifest()

    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_members = archive.testzip()
        if bad_members is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_members}")
        member_map = build_member_map(archive, manifest)
        write_json(diagnostics / "archive_inventory.json", {
            "selected_url": selected_url,
            "archive_bytes": archive_path.stat().st_size,
            "zip_members": len(archive.namelist()),
            "mapped_data_files": len(member_map),
            "download_attempts": attempts,
        })

        def extract_verified(url: str, destination: Path, expected_sha256: str, retries: int = 1) -> dict[str, Any]:
            del retries
            relative_path = url.split(p.BASE_URL.rstrip("/") + "/", 1)[-1]
            if relative_path not in member_map:
                raise RuntimeError(f"No archive member for {relative_path}")
            member = member_map[relative_path]
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            started = time.time()
            with archive.open(member, "r") as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 << 20)
            observed = p.sha256(partial)
            if observed != expected_sha256:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Archive member SHA-256 mismatch for {relative_path}: expected={expected_sha256}, observed={observed}"
                )
            os.replace(partial, destination)
            return {
                "source": "official_zip_archive",
                "archive_url": selected_url,
                "archive_member": member,
                "bytes": destination.stat().st_size,
                "sha256": observed,
                "elapsed_seconds": round(time.time() - started, 3),
            }

        p.download_verified = extract_verified
        work_root = output / "shards"
        shard_output = work_root / "shard-0"
        p.shard(0, 1, shard_output)
        final_output = output / "final"
        p.aggregate(work_root, final_output)

    write_json(final_output / "archive_execution.json", {
        "archive_url": selected_url,
        "archive_bytes": archive_path.stat().st_size,
        "archive_member_count": len(member_map),
        "disk_before": json.loads((diagnostics / "disk_before.json").read_text()),
        "disk_after_analysis": disk_state(output),
    })
    archive_path.unlink(missing_ok=True)
    shutil.rmtree(output / "shards", ignore_errors=True)
    write_json(final_output / "archive_cleanup.json", {
        "archive_deleted": not archive_path.exists(),
        "temporary_shards_deleted": not (output / "shards").exists(),
        "disk_after_cleanup": disk_state(output),
    })


if __name__ == "__main__":
    main()
