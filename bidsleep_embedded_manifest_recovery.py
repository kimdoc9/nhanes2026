#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

import bidsleep_remote_zip_runner as remote

p = remote.p


def load_completed(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        subject_id, night_index = raw_line.split(",", 1)
        completed.add((subject_id.strip(), int(night_index)))
    return completed


def install_embedded_manifest_archive(output: Path):
    raw = remote.HTTPRangeReader(remote.ARCHIVE_URL, remote.ARCHIVE_SIZE)
    buffered = io.BufferedReader(raw, buffer_size=8 << 20)
    archive = zipfile.ZipFile(buffered, "r")
    candidates = [
        name for name in archive.namelist()
        if not name.endswith("/") and name.rsplit("/", 1)[-1] == "SHA256SUMS.txt"
    ]
    if len(candidates) != 1:
        archive.close()
        buffered.close()
        raise RuntimeError(f"Expected one embedded SHA256SUMS.txt, found {candidates}")
    manifest_member = candidates[0]
    manifest_text = archive.read(manifest_member).decode("utf-8")
    p.fetch_manifest_text = lambda: manifest_text
    manifest = p.load_manifest()
    mapping = remote.member_map(archive, manifest)

    def extract_verified(url: str, destination: Path, expected_sha256: str, retries: int = 1):
        del retries
        relative_path = url.split(p.BASE_URL.rstrip("/") + "/", 1)[-1]
        member = mapping.get(relative_path)
        if member is None:
            raise RuntimeError(f"No official ZIP member for {relative_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        started = time.time()
        with archive.open(member, "r") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)
        observed = p.sha256(partial)
        if observed != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Official ZIP member SHA-256 mismatch for {relative_path}: "
                f"expected={expected_sha256}, observed={observed}"
            )
        os.replace(partial, destination)
        return {
            "source": "official_zip_http_range_embedded_manifest",
            "archive_url": remote.ARCHIVE_URL,
            "manifest_member": manifest_member,
            "archive_member": member,
            "bytes": destination.stat().st_size,
            "sha256": observed,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    p.download_verified = extract_verified
    return manifest, raw, buffered, archive, mapping, manifest_member


def recover(index: int, count: int, completed_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    completed = load_completed(completed_path)
    manifest, raw, buffered, archive, mapping, manifest_member = install_embedded_manifest_archive(output)
    qc_rows: list[dict] = []
    failures: list[dict] = []
    try:
        all_pairs = sorted(
            (str(row.subject_id), int(row.night_index))
            for row in manifest[["subject_id", "night_index"]].drop_duplicates().itertuples(index=False)
        )
        missing = [pair for pair in all_pairs if pair not in completed]
        if len(all_pairs) != p.EXPECTED_NIGHTS:
            raise RuntimeError(f"Expected {p.EXPECTED_NIGHTS} pairs, observed {len(all_pairs)}")
        if len(completed) != 156:
            raise RuntimeError(f"Expected 156 completed pairs, observed {len(completed)}")
        if len(missing) != 97:
            raise RuntimeError(f"Expected 97 missing pairs, observed {len(missing)}")
        assigned = missing[index::count]
        for subject_id, night_index in assigned:
            try:
                qc_rows.append(p.process_night(manifest, subject_id, night_index, output))
            except Exception as exc:
                failures.append({
                    "subject_id": subject_id,
                    "night_index": int(night_index),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                })
        pd.DataFrame(qc_rows).to_csv(output / "night_qc.csv", index=False)
        pd.DataFrame(
            failures,
            columns=["subject_id", "night_index", "error_type", "error"],
        ).to_csv(output / "failures.csv", index=False)
        p.write_json(output / "summary.json", {
            "recovery_index": index,
            "recovery_count": count,
            "all_pairs": len(all_pairs),
            "previously_completed": len(completed),
            "missing_total": len(missing),
            "assigned_pairs": len(assigned),
            "qc_rows": len(qc_rows),
            "failures": len(failures),
            "manifest_source": "official_zip_embedded_SHA256SUMS",
            "manifest_member": manifest_member,
            "assigned": [{"subject_id": s, "night_index": n} for s, n in assigned],
        })
        remote.write_stats(output / "remote_zip_stats.json", raw, {
            "status": "success" if not failures else "failure",
            "recovery_index": index,
            "recovery_count": count,
            "mapped_data_files": len(mapping),
            "manifest_member": manifest_member,
            "assigned_pairs": len(assigned),
            "qc_rows": len(qc_rows),
            "failures": len(failures),
        })
        if failures:
            raise SystemExit(f"Embedded-manifest recovery shard {index} had {len(failures)} failures")
    finally:
        archive.close()
        buffered.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recover(args.index, args.count, args.completed, args.out)


if __name__ == "__main__":
    main()
