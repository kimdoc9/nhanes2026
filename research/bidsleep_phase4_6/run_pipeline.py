#!/usr/bin/env python3
"""Run the locked Phase 5–6 pipeline with a narrow dataset-specific label-span correction.

The BID-Sleep data contain a small number of nights where the automated Dreem
vector is not the same length as the expert-reviewed vector. Expert labels are
the primary reference. Therefore, automated labels are restricted to the expert
review window: a longer Dreem vector is tail-truncated; a shorter vector is
padded with Unknown (5). No expert epoch is created, removed, or imputed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

import pipeline as p

_ORIGINAL_LOAD_LABELS = p.load_labels


def load_labels_expert_window(path: Path) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    try:
        return _ORIGINAL_LOAD_LABELS(path)
    except RuntimeError as exc:
        if "Invalid label lengths" not in str(exc):
            raise

    try:
        matlab = p.loadmat(path, squeeze_me=True, struct_as_record=False)
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
    recording_start = p.parse_recording_start(matlab[rec_key])
    expert = np.asarray(matlab[expert_key]).astype(np.int16).reshape(-1)
    dreem = np.asarray(matlab[dreem_key]).astype(np.int16).reshape(-1)

    if not len(expert) or not len(dreem):
        raise RuntimeError(f"Empty label vector: expert={len(expert)}, dreem={len(dreem)}")
    valid_codes = set(range(6))
    if set(np.unique(expert)) - valid_codes or set(np.unique(dreem)) - valid_codes:
        raise RuntimeError("Labels contain values outside 0..5")

    original_dreem_length = len(dreem)
    if len(dreem) > len(expert):
        dreem = dreem[: len(expert)]
        action = "tail_truncated"
    elif len(dreem) < len(expert):
        dreem = np.pad(dreem, (0, len(expert) - len(dreem)), constant_values=p.UNKNOWN)
        action = "tail_padded_unknown"
    else:
        action = "none"

    audit_marker = f"label_span_adjustment:{action}:expert={len(expert)}:dreem_original={original_dreem_length}"
    return recording_start, expert, dreem, sorted(available.values()) + [audit_marker]


p.load_labels = load_labels_expert_window

if __name__ == "__main__":
    p.main()
