#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent / "research" / "bidsleep_phase4_6"
import sys
sys.path.insert(0, str(PIPELINE_DIR))
import run_pipeline as locked  # noqa: E402

p = locked.p
PRIMARY = [
    "expert_tst_min",
    "expert_waso_min",
    "expert_wake_bout_burden_ge_1min_min",
    "expert_sleep_wake_transition_rate_per_valid_hour",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p.json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def confusion(expert: np.ndarray, dreem: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(out, (expert, dreem), 1)
    return out


def kappa_from_confusion(matrix: np.ndarray) -> float:
    n = int(matrix.sum())
    if n == 0:
        return float("nan")
    observed = float(np.trace(matrix) / n)
    expected = float((matrix.sum(axis=1) * matrix.sum(axis=0)).sum() / (n * n))
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def shard(index: int, count: int, out: Path) -> None:
    manifest = p.load_manifest()
    subjects = sorted(manifest["subject_id"].unique())[index::count]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    conf5 = np.zeros((5, 5), dtype=np.int64)
    conf2 = np.zeros((2, 2), dtype=np.int64)

    for subject_id in subjects:
        nights = sorted(manifest.loc[manifest["subject_id"] == subject_id, "night_index"].unique())
        for night_index in nights:
            local = cache / subject_id / str(int(night_index)) / "labels.mat"
            try:
                record = manifest[
                    (manifest["subject_id"] == subject_id)
                    & (manifest["night_index"] == int(night_index))
                    & (manifest["file_type"] == "labels.mat")
                ].iloc[0]
                p.download_verified(f"{p.BASE_URL}/{record['path']}", local, str(record["sha256"]))
                recording_start, expert, dreem, mat_keys = p.load_labels(local)
                jointly_valid = (expert != p.UNKNOWN) & (dreem != p.UNKNOWN)
                expert_valid = expert[jointly_valid].astype(int)
                dreem_valid = dreem[jointly_valid].astype(int)
                if len(expert_valid):
                    conf5 += confusion(expert_valid, dreem_valid, 5)
                    expert_binary = np.isin(expert_valid, list(p.SLEEP_CODES)).astype(int)
                    dreem_binary = np.isin(dreem_valid, list(p.SLEEP_CODES)).astype(int)
                    conf2 += confusion(expert_binary, dreem_binary, 2)
                duration_hours = len(expert) * p.EPOCH_SECONDS / 3600.0
                expert_valid_fraction = float((expert != p.UNKNOWN).mean())
                reasons: list[str] = []
                if duration_hours < 4.0:
                    reasons.append("label_duration_lt_4h")
                if expert_valid_fraction < 0.80:
                    reasons.append("expert_valid_fraction_lt_0.80")
                row: dict[str, Any] = {
                    "subject_id": subject_id,
                    "night_index": int(night_index),
                    "recording_start_unix": float(recording_start),
                    "label_epochs": int(len(expert)),
                    "label_duration_hours": float(duration_hours),
                    "expert_valid_fraction": expert_valid_fraction,
                    "label_structural_eligible": not reasons,
                    "label_exclusion_reasons": ";".join(reasons),
                    "mat_keys": ";".join(mat_keys),
                    **p.reference_metrics(expert, "expert"),
                    **p.reference_metrics(dreem, "dreem"),
                    "joint_valid_epochs": int(jointly_valid.sum()),
                }
                if len(expert_valid):
                    row["expert_dreem_exact_agreement_5class"] = float((expert_valid == dreem_valid).mean())
                    row["expert_dreem_kappa_5class"] = kappa_from_confusion(confusion(expert_valid, dreem_valid, 5))
                    row["expert_dreem_exact_agreement_binary"] = float((expert_binary == dreem_binary).mean())
                    row["expert_dreem_kappa_binary"] = kappa_from_confusion(confusion(expert_binary, dreem_binary, 2))
                rows.append(row)
            except Exception as exc:
                failures.append({
                    "subject_id": subject_id,
                    "night_index": int(night_index),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                })
            finally:
                shutil.rmtree(local.parent, ignore_errors=True)

    pd.DataFrame(rows).to_csv(out / "label_metrics.csv", index=False)
    pd.DataFrame(failures, columns=["subject_id", "night_index", "error_type", "error"]).to_csv(out / "failures.csv", index=False)
    np.save(out / "confusion5.npy", conf5)
    np.save(out / "confusion2.npy", conf2)
    write_json(out / "summary.json", {
        "shard_index": index,
        "shard_count": count,
        "subjects": subjects,
        "metric_rows": len(rows),
        "failures": len(failures),
    })
    if failures:
        raise SystemExit(f"label shard {index} had {len(failures)} failures")


def distribution_rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return p.distribution_rows(frame, columns)


def aggregate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    conf5 = np.zeros((5, 5), dtype=np.int64)
    conf2 = np.zeros((2, 2), dtype=np.int64)
    for directory in sorted(root.glob("shard-*")):
        for path in directory.rglob("label_metrics.csv"):
            metric_frames.append(pd.read_csv(path))
        for path in directory.rglob("failures.csv"):
            frame = pd.read_csv(path)
            if len(frame):
                failure_frames.append(frame)
        for path in directory.rglob("confusion5.npy"):
            conf5 += np.load(path)
        for path in directory.rglob("confusion2.npy"):
            conf2 += np.load(path)
    if not metric_frames:
        raise RuntimeError("No label metric shards found")
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["subject_id", "night_index"]).reset_index(drop=True)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    if len(failures):
        failures.to_csv(out / "06_label_failures.csv", index=False)
        raise RuntimeError(f"Cannot aggregate with {len(failures)} label failures")
    if len(metrics) != p.EXPECTED_NIGHTS or metrics["subject_id"].nunique() != p.EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Label completeness failure: rows={len(metrics)}, subjects={metrics['subject_id'].nunique()}"
        )
    if metrics[["subject_id", "night_index"]].duplicated().any():
        raise RuntimeError("Duplicate label subject-night rows")

    for suffix in (
        "tst_min", "waso_min", "wake_bout_burden_ge_1min_min",
        "sleep_wake_transition_rate_per_valid_hour", "recording_window_sleep_proportion",
    ):
        metrics[f"dreem_minus_expert_{suffix}"] = metrics[f"dreem_{suffix}"] - metrics[f"expert_{suffix}"]
    metrics.to_csv(out / "06_label_reference_night_metrics.csv", index=False)

    eligible = metrics[metrics["label_structural_eligible"].astype(bool)].copy()
    descriptive = PRIMARY + [
        "expert_recording_window_sleep_proportion",
        "expert_fragmentation_rate_per_sleep_hour",
        "expert_dreem_exact_agreement_5class",
        "expert_dreem_kappa_5class",
        "expert_dreem_exact_agreement_binary",
        "expert_dreem_kappa_binary",
    ]
    pd.DataFrame(distribution_rows(eligible, descriptive)).to_csv(
        out / "06_label_reference_metric_distribution.csv", index=False
    )
    difference_columns = [column for column in metrics.columns if column.startswith("dreem_minus_expert_")]
    pd.DataFrame(distribution_rows(eligible, difference_columns)).to_csv(
        out / "06_label_reference_difference_distribution.csv", index=False
    )
    pd.DataFrame([p.unbalanced_oneway_repeatability(eligible, column) for column in PRIMARY]).to_csv(
        out / "06_label_reference_repeatability.csv", index=False
    )

    gate: dict[str, Any] = {}
    for column in PRIMARY:
        values = pd.to_numeric(eligible[column], errors="coerce").dropna()
        iqr = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else float("nan")
        floor = float((values == values.min()).mean()) if len(values) else float("nan")
        gate[column] = {
            "n": int(len(values)),
            "unique": int(values.nunique()),
            "iqr": iqr,
            "floor_fraction": floor,
            "pass": bool(len(values) >= 200 and values.nunique() >= 10 and iqr > 0 and floor < 0.50),
        }
    write_json(out / "06_label_endpoint_gate.json", {
        "scope": "label-structural eligibility only; final primary gate requires Phase 5 sensor QC",
        "n_nights": int(len(eligible)),
        "n_participants": int(eligible["subject_id"].nunique()),
        "components": gate,
        "label_distribution_gate": "pass" if all(x["pass"] for x in gate.values()) else "reframe_required",
    })
    write_json(out / "06_label_agreement_summary.json", {
        "jointly_valid_epochs": int(conf5.sum()),
        "exact_agreement_5class": float(np.trace(conf5) / conf5.sum()),
        "kappa_5class": kappa_from_confusion(conf5),
        "exact_agreement_binary_sleep_wake": float(np.trace(conf2) / conf2.sum()),
        "kappa_binary_sleep_wake": kappa_from_confusion(conf2),
        "confusion_5class": conf5.tolist(),
        "confusion_binary": conf2.tolist(),
    })
    write_json(out / "06_label_completion_manifest.json", {
        "participants": int(metrics["subject_id"].nunique()),
        "participant_nights": int(len(metrics)),
        "label_structurally_eligible_nights": int(len(eligible)),
        "extraction_failures": 0,
        "generated_files": sorted(path.name for path in out.iterdir() if path.is_file()),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("shard")
    a.add_argument("--index", type=int, required=True)
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("aggregate")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        shard(args.index, args.count, args.out)
    else:
        aggregate(args.root, args.out)


if __name__ == "__main__":
    main()
