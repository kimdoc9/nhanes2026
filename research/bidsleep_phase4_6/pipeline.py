#!/usr/bin/env python3
"""BID-Sleep Phase 5-6 streaming extraction and reference-metric pipeline.

The pipeline downloads one participant-night at a time from PhysioNet, verifies
its official SHA-256 checksum, derives compact 30-second features, deletes raw
files, and aggregates all 253 nights. No model training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.metrics import cohen_kappa_score

BASE_URL = "https://physionet.org/files/bidsleep-dataset/1.0.0"
EPOCH_SECONDS = 30.0
UNKNOWN = 5
SLEEP_CODES = {1, 2, 3, 4}
EXPECTED_SUBJECTS = 47
EXPECTED_NIGHTS = 253
EXPECTED_DATA_FILES = EXPECTED_NIGHTS * 3
USER_AGENT = "BID-Sleep-operating-envelope/1.2"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str, retries: int = 3) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            command = [
                "curl", "--fail", "--location", "--retry", "12",
                "--retry-all-errors", "--retry-delay", "5",
                "--connect-timeout", "30", "--max-time", "2700",
                "--continue-at", "-", "--user-agent", USER_AGENT,
                "--output", str(partial), url,
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            attempts.append({
                "attempt": attempt,
                "returncode": completed.returncode,
                "elapsed_seconds": round(time.time() - started, 3),
                "stderr_tail": completed.stderr[-2000:],
                "bytes_after_attempt": partial.stat().st_size if partial.exists() else 0,
            })
            if completed.returncode != 0:
                raise RuntimeError(f"curl failed with exit code {completed.returncode}")
            observed = sha256(partial)
            if observed != expected_sha256:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 mismatch: expected={expected_sha256}, observed={observed}")
            os.replace(partial, destination)
            return {"url": url, "bytes": destination.stat().st_size, "sha256": observed, "attempts": attempts}
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {url}; {exc}; attempts={attempts}") from exc
            time.sleep(15 * attempt)
    raise AssertionError("unreachable")


def fetch_manifest_text() -> str:
    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            request = urllib.request.Request(f"{BASE_URL}/SHA256SUMS.txt", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            last_error = exc
            if attempt == 8:
                break
            time.sleep(5 * attempt)
    raise RuntimeError(f"Unable to retrieve SHA256SUMS.txt: {last_error}")


def load_manifest() -> pd.DataFrame:
    rows: list[tuple[str, str, str, int, str]] = []
    for line in fetch_manifest_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        path = path.lstrip("*")
        parts = path.split("/")
        if len(parts) == 3 and parts[0].startswith("Bidslab") and parts[1].isdigit() and parts[2] in {"hr.csv", "motion.csv", "labels.mat"}:
            rows.append((digest, path, parts[0], int(parts[1]), parts[2]))
    manifest = pd.DataFrame(rows, columns=["sha256", "path", "subject_id", "night_index", "file_type"])
    subject_count = manifest["subject_id"].nunique()
    night_count = manifest[["subject_id", "night_index"]].drop_duplicates().shape[0]
    if len(manifest) != EXPECTED_DATA_FILES or subject_count != EXPECTED_SUBJECTS or night_count != EXPECTED_NIGHTS:
        raise RuntimeError(f"Manifest completeness failure: files={len(manifest)}, subjects={subject_count}, nights={night_count}")
    triplets = manifest.groupby(["subject_id", "night_index"])["file_type"].nunique()
    if not (triplets == 3).all():
        raise RuntimeError("At least one subject-night does not have a complete three-file triplet")
    return manifest.sort_values(["subject_id", "night_index", "file_type"]).reset_index(drop=True)


def scalar(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_recording_start(value: Any) -> float:
    value = scalar(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric > 1e9:
            return numeric
        return (numeric - 719529.0) * 86400.0
    if isinstance(value, str):
        text = value.strip().strip("[]'")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
            for template in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, template)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return parse_recording_start(float(text))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("America/New_York"))
        return parsed.timestamp()
    raise RuntimeError(f"Unsupported recStart type: {type(value)}")


def load_labels(path: Path) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    try:
        matlab = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py
        with h5py.File(path) as handle:
            matlab = {key: np.array(handle[key]) for key in handle.keys()}
    available = {key.lower(): key for key in matlab if not key.startswith("__")}
    def locate(candidates: Iterable[str]) -> str:
        for candidate in candidates:
            if candidate.lower() in available:
                return available[candidate.lower()]
        raise RuntimeError(f"Missing MAT key {list(candidates)}; found={list(available.values())}")
    rec_key = locate(["recStart", "rec_start", "recording_start"])
    expert_key = locate(["expert_label", "expertLabel", "manual_label"])
    dreem_key = locate(["dreem_label", "dreemLabel", "auto_label"])
    recording_start = parse_recording_start(matlab[rec_key])
    expert = np.asarray(matlab[expert_key]).astype(np.int16).reshape(-1)
    dreem = np.asarray(matlab[dreem_key]).astype(np.int16).reshape(-1)
    if not len(expert) or len(expert) != len(dreem):
        raise RuntimeError(f"Invalid label lengths: expert={len(expert)}, dreem={len(dreem)}")
    valid_codes = set(range(6))
    if set(np.unique(expert)) - valid_codes or set(np.unique(dreem)) - valid_codes:
        raise RuntimeError("Labels contain values outside 0..5")
    return recording_start, expert, dreem, sorted(available.values())


def read_two_column_csv(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=["timestamp", value_name], usecols=[0, 1])
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    return frame


def heart_rate_features(path: Path, start: float, epoch_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_two_column_csv(path, "hr")
    finite = np.isfinite(raw["timestamp"].to_numpy(float)) & np.isfinite(raw["hr"].to_numpy(float))
    timestamps = raw.loc[finite, "timestamp"].to_numpy(float)
    values = raw.loc[finite, "hr"].to_numpy(float)
    epoch_index = np.floor((timestamps - start) / EPOCH_SECONDS).astype(int)
    within = (epoch_index >= 0) & (epoch_index < epoch_count)
    grouped = pd.DataFrame({"epoch": epoch_index[within], "hr": values[within]}).groupby("epoch")["hr"]
    output = pd.DataFrame({"epoch": np.arange(epoch_count)})
    output["hr_count"] = grouped.size().reindex(output["epoch"], fill_value=0).to_numpy()
    output["hr_mean"] = grouped.mean().reindex(output["epoch"]).to_numpy()
    output["hr_std"] = grouped.std(ddof=0).reindex(output["epoch"]).to_numpy()
    output["hr_min"] = grouped.min().reindex(output["epoch"]).to_numpy()
    output["hr_max"] = grouped.max().reindex(output["epoch"]).to_numpy()
    output["hr_present"] = (output["hr_count"] > 0).astype(np.int8)
    deltas = np.diff(timestamps)
    diagnostics = {
        "hr_rows_total": int(len(raw)), "hr_rows_finite": int(finite.sum()), "hr_rows_in_label_window": int(within.sum()),
        "hr_epoch_coverage": float(output["hr_present"].mean()),
        "hr_nonmonotonic_timestamp_count": int((deltas < 0).sum()), "hr_duplicate_timestamp_count": int((deltas == 0).sum()),
        "hr_gap_gt_30s_count": int((deltas > 30).sum()), "hr_dt_median_s": float(np.median(deltas)) if len(deltas) else np.nan,
        "hr_dt_p95_s": float(np.quantile(deltas, 0.95)) if len(deltas) else np.nan, "hr_dt_max_s": float(np.max(deltas)) if len(deltas) else np.nan,
        "hr_min_bpm": float(np.min(values)) if len(values) else np.nan, "hr_max_bpm": float(np.max(values)) if len(values) else np.nan,
    }
    return output, diagnostics


def accelerometer_features(path: Path, start: float, epoch_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = np.zeros(epoch_count, dtype=np.int64)
    sums = np.zeros((epoch_count, 4), dtype=np.float64)
    squared_sums = np.zeros((epoch_count, 4), dtype=np.float64)
    previous_timestamp: float | None = None
    rows_total = rows_finite = rows_in_window = nonmonotonic = duplicates = gaps_gt_1s = 0
    max_gap = np.nan
    for chunk in pd.read_csv(path, header=None, names=["timestamp", "x", "y", "z"], usecols=[0, 1, 2, 3], chunksize=500_000):
        rows_total += len(chunk)
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(float)
        xyz = np.column_stack([pd.to_numeric(chunk[axis], errors="coerce").to_numpy(float) for axis in ("x", "y", "z")])
        finite = np.isfinite(timestamps) & np.isfinite(xyz).all(axis=1)
        timestamps = timestamps[finite]
        xyz = xyz[finite]
        rows_finite += len(timestamps)
        if not len(timestamps):
            continue
        if previous_timestamp is not None:
            boundary_delta = timestamps[0] - previous_timestamp
            nonmonotonic += int(boundary_delta < 0)
            duplicates += int(boundary_delta == 0)
            gaps_gt_1s += int(boundary_delta > 1)
            max_gap = boundary_delta if not math.isfinite(max_gap) else max(max_gap, boundary_delta)
        deltas = np.diff(timestamps)
        nonmonotonic += int((deltas < 0).sum())
        duplicates += int((deltas == 0).sum())
        gaps_gt_1s += int((deltas > 1).sum())
        if len(deltas):
            local_max = float(np.max(deltas))
            max_gap = local_max if not math.isfinite(max_gap) else max(max_gap, local_max)
        previous_timestamp = float(timestamps[-1])
        epoch_index = np.floor((timestamps - start) / EPOCH_SECONDS).astype(int)
        within = (epoch_index >= 0) & (epoch_index < epoch_count)
        epoch_index = epoch_index[within]
        xyz = xyz[within]
        rows_in_window += int(within.sum())
        if not len(epoch_index):
            continue
        magnitude = np.sqrt(np.sum(xyz * xyz, axis=1))
        values = np.column_stack([xyz, magnitude])
        np.add.at(counts, epoch_index, 1)
        for column in range(4):
            np.add.at(sums[:, column], epoch_index, values[:, column])
            np.add.at(squared_sums[:, column], epoch_index, values[:, column] ** 2)
    denominator = np.where(counts > 0, counts, 1)[:, None]
    means = sums / denominator
    standard_deviations = np.sqrt(np.maximum(squared_sums / denominator - means ** 2, 0))
    means[counts == 0] = np.nan
    standard_deviations[counts == 0] = np.nan
    output = pd.DataFrame({
        "epoch": np.arange(epoch_count), "acc_count": counts,
        "acc_x_mean": means[:, 0], "acc_y_mean": means[:, 1], "acc_z_mean": means[:, 2],
        "acc_mag_mean": means[:, 3], "acc_mag_std": standard_deviations[:, 3],
        "acc_present": (counts > 0).astype(np.int8),
    })
    diagnostics = {
        "acc_rows_total": int(rows_total), "acc_rows_finite": int(rows_finite), "acc_rows_in_label_window": int(rows_in_window),
        "acc_epoch_coverage": float(output["acc_present"].mean()),
        "acc_nonmonotonic_timestamp_count": int(nonmonotonic), "acc_duplicate_timestamp_count": int(duplicates),
        "acc_gap_gt_1s_count": int(gaps_gt_1s), "acc_dt_max_s": float(max_gap) if math.isfinite(max_gap) else np.nan,
    }
    return output, diagnostics


def run_lengths(values: np.ndarray, target: int) -> list[int]:
    lengths: list[int] = []
    if not len(values):
        return lengths
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if values[start] == target:
                lengths.append(index - start)
            start = index
    return lengths


def reference_metrics(labels: np.ndarray, prefix: str) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    valid_indices = np.flatnonzero(labels != UNKNOWN)
    if not len(valid_indices):
        return {f"{prefix}_metric_status": "no_valid_labels"}
    window = labels[valid_indices[0]: valid_indices[-1] + 1]
    valid = window != UNKNOWN
    asleep = np.isin(window, list(SLEEP_CODES))
    awake = window == 0
    sleep_indices = np.flatnonzero(asleep)
    result: dict[str, Any] = {
        f"{prefix}_metric_status": "ok" if len(sleep_indices) else "no_sleep",
        f"{prefix}_analysis_span_epochs": int(len(window)), f"{prefix}_valid_epochs_in_span": int(valid.sum()),
        f"{prefix}_valid_fraction_in_span": float(valid.mean()), f"{prefix}_tst_min": float(asleep.sum() / 2.0),
        f"{prefix}_recording_window_sleep_proportion": float(asleep.sum() / max(valid.sum(), 1)),
    }
    if not len(sleep_indices):
        return result
    first_sleep, last_sleep = int(sleep_indices[0]), int(sleep_indices[-1])
    post_onset = window[first_sleep:last_sleep + 1]
    wake_runs = run_lengths(post_onset, target=0)
    long_wake_runs = [length for length in wake_runs if length >= 2]
    result[f"{prefix}_waso_min"] = float((post_onset == 0).sum() / 2.0)
    result[f"{prefix}_wake_bout_count_ge_1min"] = int(len(long_wake_runs))
    result[f"{prefix}_wake_bout_burden_ge_1min_min"] = float(sum(long_wake_runs) / 2.0)
    result[f"{prefix}_fragmentation_rate_per_sleep_hour"] = float(len(long_wake_runs) / max(asleep.sum() / 120.0, 1e-12))
    binary = np.full(len(window), -1, dtype=np.int8)
    binary[asleep] = 1
    binary[awake] = 0
    adjacent_valid = (binary[:-1] >= 0) & (binary[1:] >= 0)
    transitions = int(((binary[:-1] != binary[1:]) & adjacent_valid).sum())
    result[f"{prefix}_sleep_wake_transition_count"] = transitions
    result[f"{prefix}_sleep_wake_transition_rate_per_valid_hour"] = float(transitions / max(valid.sum() / 120.0, 1e-12))
    return result


def process_night(manifest: pd.DataFrame, subject_id: str, night_index: int, output: Path) -> dict[str, Any]:
    selection = manifest[(manifest["subject_id"] == subject_id) & (manifest["night_index"] == night_index)]
    cache = output / "cache" / subject_id / str(night_index)
    epoch_directory = output / "epoch_features"
    diagnostics_directory = output / "download_diagnostics"
    epoch_directory.mkdir(parents=True, exist_ok=True)
    diagnostics_directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    downloads: dict[str, Any] = {}
    try:
        for file_type in ("labels.mat", "hr.csv", "motion.csv"):
            row = selection[selection["file_type"] == file_type]
            if len(row) != 1:
                raise RuntimeError(f"Expected exactly one {file_type}; found={len(row)}")
            record = row.iloc[0]
            destination = cache / file_type
            downloads[file_type] = download_verified(f"{BASE_URL}/{record['path']}", destination, str(record["sha256"]))
            files[file_type] = destination
        recording_start, expert, dreem, mat_keys = load_labels(files["labels.mat"])
        epoch_count = len(expert)
        hr, hr_qc = heart_rate_features(files["hr.csv"], recording_start, epoch_count)
        acc, acc_qc = accelerometer_features(files["motion.csv"], recording_start, epoch_count)
        epoch = hr.merge(acc, on="epoch", validate="one_to_one")
        epoch.insert(0, "night_index", int(night_index))
        epoch.insert(0, "subject_id", subject_id)
        epoch["expert_label"] = expert
        epoch["dreem_label"] = dreem
        epoch_file = epoch_directory / f"{subject_id}_night{night_index}.csv.gz"
        epoch.to_csv(epoch_file, index=False, compression="gzip")
        duration_hours = epoch_count * EPOCH_SECONDS / 3600.0
        expert_valid_fraction = float((expert != UNKNOWN).mean())
        reasons: list[str] = []
        if duration_hours < 4:
            reasons.append("label_duration_lt_4h")
        if expert_valid_fraction < 0.80:
            reasons.append("expert_valid_fraction_lt_0.80")
        if hr_qc["hr_nonmonotonic_timestamp_count"] or acc_qc["acc_nonmonotonic_timestamp_count"]:
            reasons.append("nonmonotonic_timestamp")
        qc_decision = "exclude" if reasons else "include"
        joint_model_eligible = qc_decision == "include" and hr_qc["hr_epoch_coverage"] >= 0.10 and acc_qc["acc_epoch_coverage"] >= 0.10
        record = {
            "subject_id": subject_id, "night_index": int(night_index), "recording_start_unix": float(recording_start),
            "label_epochs": int(epoch_count), "label_duration_hours": float(duration_hours),
            "expert_valid_fraction": expert_valid_fraction, "mat_keys": ";".join(mat_keys),
            **hr_qc, **acc_qc, "qc_decision": qc_decision, "qc_reasons": ";".join(reasons),
            "joint_model_eligible": bool(joint_model_eligible), "epoch_feature_file": epoch_file.name,
        }
        write_json(diagnostics_directory / f"{subject_id}_night{night_index}.json", json_safe({"downloads": downloads, "qc": record}))
        return record
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def shard(index: int, shard_count: int, output: Path) -> None:
    manifest = load_manifest()
    subjects = sorted(manifest["subject_id"].unique())[index::shard_count]
    output.mkdir(parents=True, exist_ok=True)
    qc_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for subject_id in subjects:
        nights = sorted(manifest.loc[manifest["subject_id"] == subject_id, "night_index"].unique())
        for night_index in nights:
            try:
                qc_rows.append(process_night(manifest, subject_id, int(night_index), output))
            except Exception as exc:
                failures.append({"subject_id": subject_id, "night_index": int(night_index), "error_type": type(exc).__name__, "error": repr(exc)})
    pd.DataFrame(qc_rows).to_csv(output / "night_qc.csv", index=False)
    pd.DataFrame(failures, columns=["subject_id", "night_index", "error_type", "error"]).to_csv(output / "failures.csv", index=False)
    write_json(output / "summary.json", {"shard_index": index, "shard_count": shard_count, "subjects": subjects, "qc_rows": len(qc_rows), "failures": len(failures)})
    if failures:
        raise SystemExit(f"Shard {index} had {len(failures)} subject-night failures")


def pooled_label_agreement(epoch_files: list[Path]) -> dict[str, Any]:
    expert_values: list[np.ndarray] = []
    dreem_values: list[np.ndarray] = []
    for path in epoch_files:
        frame = pd.read_csv(path, usecols=["expert_label", "dreem_label"])
        expert = frame["expert_label"].to_numpy(int)
        dreem = frame["dreem_label"].to_numpy(int)
        jointly_valid = (expert != UNKNOWN) & (dreem != UNKNOWN)
        expert_values.append(expert[jointly_valid])
        dreem_values.append(dreem[jointly_valid])
    expert_all = np.concatenate(expert_values)
    dreem_all = np.concatenate(dreem_values)
    expert_binary = np.isin(expert_all, list(SLEEP_CODES)).astype(int)
    dreem_binary = np.isin(dreem_all, list(SLEEP_CODES)).astype(int)
    return {
        "jointly_valid_epochs": int(len(expert_all)), "exact_agreement_5class": float((expert_all == dreem_all).mean()),
        "kappa_5class": float(cohen_kappa_score(expert_all, dreem_all, labels=[0, 1, 2, 3, 4])),
        "exact_agreement_binary_sleep_wake": float((expert_binary == dreem_binary).mean()),
        "kappa_binary_sleep_wake": float(cohen_kappa_score(expert_binary, dreem_binary, labels=[0, 1])),
    }


def distribution_rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        rows.append({
            "metric": column, "n": int(len(values)), "unique": int(values.nunique()),
            "mean": float(values.mean()) if len(values) else np.nan,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "min": float(values.min()) if len(values) else np.nan,
            "p25": float(values.quantile(0.25)) if len(values) else np.nan,
            "median": float(values.median()) if len(values) else np.nan,
            "p75": float(values.quantile(0.75)) if len(values) else np.nan,
            "max": float(values.max()) if len(values) else np.nan,
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan,
            "floor_fraction": float((values == values.min()).mean()) if len(values) else np.nan,
            "ceiling_fraction": float((values == values.max()).mean()) if len(values) else np.nan,
        })
    return rows


def unbalanced_oneway_repeatability(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    subset = frame[["subject_id", metric]].dropna().copy()
    groups = [group[metric].to_numpy(float) for _, group in subset.groupby("subject_id")]
    groups = [values for values in groups if len(values) >= 2]
    participant_count = len(groups)
    observation_count = sum(len(values) for values in groups)
    if participant_count < 2 or observation_count <= participant_count:
        return {"metric": metric, "status": "insufficient_repeated_data"}
    all_values = np.concatenate(groups)
    grand_mean = float(all_values.mean())
    counts = np.asarray([len(values) for values in groups], dtype=float)
    means = np.asarray([values.mean() for values in groups], dtype=float)
    ss_between = float(np.sum(counts * (means - grand_mean) ** 2))
    ss_within = float(sum(np.sum((values - values.mean()) ** 2) for values in groups))
    ms_between = ss_between / (participant_count - 1)
    ms_within = ss_within / (observation_count - participant_count)
    n0 = (observation_count - np.sum(counts ** 2) / observation_count) / (participant_count - 1)
    variance_between = max((ms_between - ms_within) / n0, 0.0)
    variance_within = max(ms_within, 0.0)
    total_variance = variance_between + variance_within
    icc = variance_between / total_variance if total_variance > 0 else np.nan
    sem = math.sqrt(variance_within)
    mdc95 = 1.96 * math.sqrt(2.0) * sem
    return {
        "metric": metric, "status": "ok", "participants_with_repeats": participant_count,
        "observations": observation_count, "variance_between": variance_between, "variance_within": variance_within,
        "icc_oneway_random": icc, "within_person_sd_sem": sem, "mdc95": mdc95,
        "method": "unbalanced one-way random-effects method-of-moments",
    }


def choose_nearest(value: float, grid: list[float]) -> float:
    return min(grid, key=lambda candidate: (abs(candidate - value), candidate))


def aggregate(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    epoch_output = output / "epoch_features"
    epoch_output.mkdir(exist_ok=True)
    qc_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for shard_directory in sorted(root.glob("shard-*")):
        for path in shard_directory.rglob("*.csv.gz"):
            destination = epoch_output / path.name
            if destination.exists():
                raise RuntimeError(f"Duplicate epoch feature file: {destination.name}")
            shutil.copy2(path, destination)
        for path in shard_directory.rglob("night_qc.csv"):
            qc_frames.append(pd.read_csv(path))
        for path in shard_directory.rglob("failures.csv"):
            failure = pd.read_csv(path)
            if len(failure):
                failure_frames.append(failure)
    if not qc_frames:
        raise RuntimeError("No shard QC files found")
    qc = pd.concat(qc_frames, ignore_index=True).sort_values(["subject_id", "night_index"]).reset_index(drop=True)
    epoch_files = sorted(epoch_output.glob("*.csv.gz"))
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    if len(failures):
        failures.to_csv(output / "05_failures.csv", index=False)
        raise RuntimeError(f"Cannot aggregate with {len(failures)} extraction failures")
    if len(qc) != EXPECTED_NIGHTS or len(epoch_files) != EXPECTED_NIGHTS:
        raise RuntimeError(f"Completeness failure: qc_rows={len(qc)}, epoch_files={len(epoch_files)}, expected={EXPECTED_NIGHTS}")
    if qc[["subject_id", "night_index"]].duplicated().any():
        raise RuntimeError("Duplicate subject-night QC rows")
    if qc["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Participant completeness failure: {qc['subject_id'].nunique()}")
    qc.to_csv(output / "05_night_qc.csv", index=False)
    subject_summary = qc.groupby("subject_id").agg(
        nights=("night_index", "count"),
        included_nights=("qc_decision", lambda x: int((x == "include").sum())),
        joint_eligible_nights=("joint_model_eligible", lambda x: int(pd.Series(x).astype(bool).sum())),
        median_hr_epoch_coverage=("hr_epoch_coverage", "median"),
        median_acc_epoch_coverage=("acc_epoch_coverage", "median"),
    ).reset_index()
    subject_summary.to_csv(output / "05_subject_qc_summary.csv", index=False)
    signal_columns = ["hr_epoch_coverage", "acc_epoch_coverage", "hr_dt_median_s", "hr_dt_p95_s", "hr_gap_gt_30s_count", "acc_gap_gt_1s_count"]
    pd.DataFrame(distribution_rows(qc, signal_columns)).to_csv(output / "05_signal_availability_distribution.csv", index=False)
    included_for_natural = qc[qc["qc_decision"] == "include"].copy()
    hr_deficit_p90 = float((1.0 - included_for_natural["hr_epoch_coverage"]).quantile(0.90))
    acc_deficit_p90 = float((1.0 - included_for_natural["acc_epoch_coverage"]).quantile(0.90))
    natural_state = {
        "selection_rule": "nearest pre-specified grid point to the cohort 90th percentile natural epoch deficit; model outcomes unseen",
        "hr_epoch_deficit_p90": hr_deficit_p90, "acc_epoch_deficit_p90": acc_deficit_p90,
        "hr_random_thinning_selected": choose_nearest(hr_deficit_p90, [0.0, 0.10, 0.25, 0.50, 0.75]),
        "acc_duty_cycle_selected": 1.0 - choose_nearest(acc_deficit_p90, [0.0, 0.25, 0.50, 0.75]),
    }
    write_json(output / "05_natural_degradation_state.json", json_safe(natural_state))
    metric_rows: list[dict[str, Any]] = []
    for path in epoch_files:
        epoch = pd.read_csv(path)
        expert = epoch["expert_label"].to_numpy(int)
        dreem = epoch["dreem_label"].to_numpy(int)
        row: dict[str, Any] = {
            "subject_id": str(epoch["subject_id"].iloc[0]), "night_index": int(epoch["night_index"].iloc[0]),
            **reference_metrics(expert, "expert"), **reference_metrics(dreem, "dreem"),
        }
        jointly_valid = (expert != UNKNOWN) & (dreem != UNKNOWN)
        row["joint_valid_epochs"] = int(jointly_valid.sum())
        if jointly_valid.any():
            expert_valid = expert[jointly_valid]
            dreem_valid = dreem[jointly_valid]
            row["expert_dreem_exact_agreement_5class"] = float((expert_valid == dreem_valid).mean())
            row["expert_dreem_kappa_5class"] = float(cohen_kappa_score(expert_valid, dreem_valid, labels=[0, 1, 2, 3, 4]))
            expert_binary = np.isin(expert_valid, list(SLEEP_CODES)).astype(int)
            dreem_binary = np.isin(dreem_valid, list(SLEEP_CODES)).astype(int)
            row["expert_dreem_exact_agreement_binary"] = float((expert_binary == dreem_binary).mean())
            row["expert_dreem_kappa_binary"] = float(cohen_kappa_score(expert_binary, dreem_binary, labels=[0, 1]))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).merge(qc, on=["subject_id", "night_index"], how="left", validate="one_to_one")
    for suffix in ("tst_min", "waso_min", "wake_bout_burden_ge_1min_min", "sleep_wake_transition_rate_per_valid_hour", "recording_window_sleep_proportion"):
        metrics[f"dreem_minus_expert_{suffix}"] = metrics[f"dreem_{suffix}"] - metrics[f"expert_{suffix}"]
    metrics.to_csv(output / "06_reference_night_metrics.csv", index=False)
    primary = metrics[(metrics["qc_decision"] == "include") & metrics["joint_model_eligible"].astype(bool)].copy()
    primary_columns = ["expert_tst_min", "expert_waso_min", "expert_wake_bout_burden_ge_1min_min", "expert_sleep_wake_transition_rate_per_valid_hour"]
    descriptive_columns = primary_columns + [
        "expert_recording_window_sleep_proportion", "expert_fragmentation_rate_per_sleep_hour",
        "expert_dreem_exact_agreement_5class", "expert_dreem_kappa_5class",
        "expert_dreem_exact_agreement_binary", "expert_dreem_kappa_binary",
    ]
    pd.DataFrame(distribution_rows(primary, descriptive_columns)).to_csv(output / "06_reference_metric_distribution.csv", index=False)
    gate_components: dict[str, Any] = {}
    for column in primary_columns:
        values = pd.to_numeric(primary[column], errors="coerce").dropna()
        iqr = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
        floor_fraction = float((values == values.min()).mean()) if len(values) else np.nan
        gate_components[column] = {
            "n": int(len(values)), "unique": int(values.nunique()), "iqr": iqr, "floor_fraction": floor_fraction,
            "pass": bool(len(values) >= 200 and values.nunique() >= 10 and iqr > 0 and floor_fraction < 0.50),
        }
    endpoint_gate = {
        "n_nights": int(len(primary)), "n_participants": int(primary["subject_id"].nunique()), "components": gate_components,
        "primary_composite_distribution_gate": "pass" if all(component["pass"] for component in gate_components.values()) else "reframe_required",
    }
    write_json(output / "06_endpoint_gate.json", json_safe(endpoint_gate))
    write_json(output / "06_label_agreement_summary.json", json_safe(pooled_label_agreement(epoch_files)))
    difference_columns = [column for column in metrics.columns if column.startswith("dreem_minus_expert_")]
    pd.DataFrame(distribution_rows(primary, difference_columns)).to_csv(output / "06_reference_difference_distribution.csv", index=False)
    pd.DataFrame([unbalanced_oneway_repeatability(primary, column) for column in primary_columns]).to_csv(output / "06_reference_repeatability.csv", index=False)
    decision = {
        "phase5_status": "complete", "phase6_status": "complete",
        "participants_manifest": EXPECTED_SUBJECTS, "nights_manifest": EXPECTED_NIGHTS,
        "participants_primary": int(primary["subject_id"].nunique()), "nights_primary": int(len(primary)),
        "qc_included_nights": int((qc["qc_decision"] == "include").sum()),
        "joint_model_eligible_nights": int(qc["joint_model_eligible"].astype(bool).sum()),
        "endpoint_gate": endpoint_gate["primary_composite_distribution_gate"],
        "next_phase_allowed": bool(endpoint_gate["primary_composite_distribution_gate"] == "pass" and len(primary) >= 200),
        "claim_boundary": "Engineering validation of multi-night sleep-continuity monitoring; no insomnia diagnosis or clinical validation claim.",
    }
    write_json(output / "06_phase6_decision.json", json_safe(decision))
    write_json(output / "06_completion_manifest.json", {
        "epoch_feature_files": len(epoch_files), "qc_rows": len(qc), "metric_rows": len(metrics),
        "participants": int(metrics["subject_id"].nunique()),
        "generated_files": sorted(path.name for path in output.iterdir() if path.is_file()),
    })


def preflight(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    subject_id = "Bidslab10" if "Bidslab10" in set(manifest["subject_id"]) else sorted(manifest["subject_id"])[0]
    night_index = int(manifest.loc[manifest["subject_id"] == subject_id, "night_index"].min())
    qc = process_night(manifest, subject_id, night_index, output)
    write_json(output / "preflight_summary.json", json_safe({
        "status": "pass", "manifest_files": len(manifest), "subjects": int(manifest["subject_id"].nunique()),
        "nights": int(manifest[["subject_id", "night_index"]].drop_duplicates().shape[0]),
        "representative_subject": subject_id, "representative_night": night_index, "qc": qc,
    }))


def selftest(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    start = 1_700_000_000.0
    epoch_count = 960
    expert = np.full(epoch_count, 2, dtype=np.int16)
    expert[:40] = 0
    expert[300:304] = 0
    expert[700:704] = 0
    expert[-20:] = 0
    dreem = expert.copy()
    dreem[100:110] = 1
    labels_path = output / "labels.mat"
    savemat(labels_path, {"recStart": start, "expert_label": expert, "dreem_label": dreem})
    hr_path = output / "hr.csv"
    hr_times = start + np.arange(0, epoch_count * EPOCH_SECONDS, 5.0)
    pd.DataFrame({0: hr_times, 1: 60 + 5 * np.sin(np.arange(len(hr_times)) / 100)}).to_csv(hr_path, header=False, index=False)
    motion_path = output / "motion.csv"
    motion_times = start + np.arange(0, epoch_count * EPOCH_SECONDS, 0.5)
    pd.DataFrame({0: motion_times, 1: np.sin(np.arange(len(motion_times)) / 20), 2: np.cos(np.arange(len(motion_times)) / 20), 3: np.ones(len(motion_times)) * 0.1}).to_csv(motion_path, header=False, index=False)
    parsed_start, parsed_expert, parsed_dreem, keys = load_labels(labels_path)
    assert parsed_start == start
    assert np.array_equal(parsed_expert, expert)
    assert np.array_equal(parsed_dreem, dreem)
    hr, hr_qc = heart_rate_features(hr_path, start, epoch_count)
    acc, acc_qc = accelerometer_features(motion_path, start, epoch_count)
    assert len(hr) == epoch_count and len(acc) == epoch_count
    assert hr_qc["hr_epoch_coverage"] == 1.0
    assert acc_qc["acc_epoch_coverage"] == 1.0
    expert_metrics = reference_metrics(expert, "expert")
    assert expert_metrics["expert_tst_min"] > 0
    write_json(output / "selftest_summary.json", json_safe({"status": "pass", "mat_keys": keys, "hr_qc": hr_qc, "acc_qc": acc_qc, "expert_metrics": expert_metrics}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    selftest_parser = subparsers.add_parser("selftest")
    selftest_parser.add_argument("--out", type=Path, required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--out", type=Path, required=True)
    shard_parser = subparsers.add_parser("shard")
    shard_parser.add_argument("--index", type=int, required=True)
    shard_parser.add_argument("--count", type=int, required=True)
    shard_parser.add_argument("--out", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--root", type=Path, required=True)
    aggregate_parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "selftest":
        selftest(arguments.out)
    elif arguments.command == "preflight":
        preflight(arguments.out)
    elif arguments.command == "shard":
        shard(arguments.index, arguments.count, arguments.out)
    elif arguments.command == "aggregate":
        aggregate(arguments.root, arguments.out)
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
