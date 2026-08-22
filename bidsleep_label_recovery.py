#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent / "research" / "bidsleep_phase4_6"
sys.path.insert(0, str(PIPELINE_DIR))
import run_pipeline as locked  # noqa: E402

p = locked.p
SUBJECT = "Bidslab30"
NIGHT = 6


def kappa(expert: np.ndarray, dreem: np.ndarray, classes: int) -> float:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (expert, dreem), 1)
    total = int(matrix.sum())
    observed = float(np.trace(matrix) / total)
    expected = float((matrix.sum(1) * matrix.sum(0)).sum() / (total * total))
    return float((observed - expected) / (1.0 - expected)) if expected < 1 else float("nan")


def main() -> None:
    out = Path("recovery-output")
    out.mkdir(exist_ok=True)
    manifest = p.load_manifest()
    row = manifest[
        (manifest.subject_id == SUBJECT)
        & (manifest.night_index == NIGHT)
        & (manifest.file_type == "labels.mat")
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one labels.mat row, found {len(row)}")
    record = row.iloc[0]
    cache = out / "cache" / SUBJECT / str(NIGHT)
    path = cache / "labels.mat"
    try:
        download = p.download_verified(f"{p.BASE_URL}/{record['path']}", path, str(record.sha256))
        start, expert, dreem, keys = p.load_labels(path)
        jointly_valid = (expert != p.UNKNOWN) & (dreem != p.UNKNOWN)
        ev = expert[jointly_valid].astype(int)
        dv = dreem[jointly_valid].astype(int)
        eb = np.isin(ev, list(p.SLEEP_CODES)).astype(int)
        db = np.isin(dv, list(p.SLEEP_CODES)).astype(int)
        duration_hours = len(expert) * p.EPOCH_SECONDS / 3600.0
        valid_fraction = float((expert != p.UNKNOWN).mean())
        reasons = []
        if duration_hours < 4:
            reasons.append("label_duration_lt_4h")
        if valid_fraction < 0.80:
            reasons.append("expert_valid_fraction_lt_0.80")
        output = {
            "subject_id": SUBJECT,
            "night_index": NIGHT,
            "recording_start_unix": float(start),
            "label_epochs": int(len(expert)),
            "label_duration_hours": float(duration_hours),
            "expert_valid_fraction": valid_fraction,
            "label_structural_eligible": not reasons,
            "label_exclusion_reasons": ";".join(reasons),
            "mat_keys": ";".join(keys),
            **p.reference_metrics(expert, "expert"),
            **p.reference_metrics(dreem, "dreem"),
            "joint_valid_epochs": int(jointly_valid.sum()),
            "expert_dreem_exact_agreement_5class": float((ev == dv).mean()),
            "expert_dreem_kappa_5class": kappa(ev, dv, 5),
            "expert_dreem_exact_agreement_binary": float((eb == db).mean()),
            "expert_dreem_kappa_binary": kappa(eb, db, 2),
        }
        pd.DataFrame([output]).to_csv(out / "recovered_label_metric.csv", index=False)
        conf5 = np.zeros((5, 5), dtype=np.int64)
        conf2 = np.zeros((2, 2), dtype=np.int64)
        np.add.at(conf5, (ev, dv), 1)
        np.add.at(conf2, (eb, db), 1)
        np.save(out / "confusion5.npy", conf5)
        np.save(out / "confusion2.npy", conf2)
        (out / "recovery.json").write_text(json.dumps({
            "subject_id": SUBJECT,
            "night_index": NIGHT,
            "download": download,
            "label_epochs": len(expert),
            "dreem_epochs_after_alignment": len(dreem),
            "mat_keys": keys,
            "status": "success",
        }, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
