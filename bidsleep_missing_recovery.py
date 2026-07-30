#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import bidsleep_remote_zip_runner as remote

p = remote.p


def load_completed(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        subject_id, night_index = raw.split(",", 1)
        completed.add((subject_id.strip(), int(night_index)))
    return completed


def recover(index: int, count: int, completed_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    completed = load_completed(completed_path)
    manifest, raw, buffered, archive, mapping = remote.install_remote_archive(output)
    qc_rows: list[dict] = []
    failures: list[dict] = []
    try:
        all_pairs = sorted(
            (str(row.subject_id), int(row.night_index))
            for row in manifest[["subject_id", "night_index"]].drop_duplicates().itertuples(index=False)
        )
        missing = [pair for pair in all_pairs if pair not in completed]
        if len(all_pairs) != p.EXPECTED_NIGHTS:
            raise RuntimeError(f"Expected {p.EXPECTED_NIGHTS} participant-nights, observed {len(all_pairs)}")
        if len(completed) != 106:
            raise RuntimeError(f"Expected 106 completed participant-nights, observed {len(completed)}")
        if len(missing) != 147:
            raise RuntimeError(f"Expected 147 missing participant-nights, observed {len(missing)}")
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
        p.write_json(
            output / "summary.json",
            {
                "recovery_index": index,
                "recovery_count": count,
                "all_pairs": len(all_pairs),
                "previously_completed": len(completed),
                "missing_total": len(missing),
                "assigned_pairs": len(assigned),
                "qc_rows": len(qc_rows),
                "failures": len(failures),
                "assigned": [{"subject_id": s, "night_index": n} for s, n in assigned],
            },
        )
        remote.write_stats(
            output / "remote_zip_stats.json",
            raw,
            {
                "status": "success" if not failures else "failure",
                "recovery_index": index,
                "recovery_count": count,
                "mapped_data_files": len(mapping),
                "assigned_pairs": len(assigned),
                "qc_rows": len(qc_rows),
                "failures": len(failures),
            },
        )
        if failures:
            raise SystemExit(f"Recovery shard {index} had {len(failures)} failures")
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
