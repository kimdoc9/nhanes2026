#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

PIPELINE_DIR = Path(__file__).resolve().parent / "research" / "bidsleep_phase4_6"
sys.path.insert(0, str(PIPELINE_DIR))
import run_pipeline as locked  # noqa: E402
import bidsleep_remote_zip_runner as remote  # noqa: E402

p = locked.p
MANIFEST_URLS = [
    "https://physionet.org/files/bidsleep-dataset/1.0.0/SHA256SUMS.txt",
    "https://physionet.org/content/bidsleep-dataset/1.0.0/SHA256SUMS.txt?download=1",
]


def robust_manifest_text() -> str:
    session = requests.Session()
    session.headers.update({"User-Agent": p.USER_AGENT, "Accept-Encoding": "identity"})
    errors: list[str] = []
    for round_index in range(1, 21):
        for url in MANIFEST_URLS:
            try:
                response = session.get(url, timeout=(30, 180))
                response.raise_for_status()
                text = response.text
                count = sum(
                    1 for line in text.splitlines()
                    if line.strip().endswith(("/hr.csv", "/motion.csv", "/labels.mat"))
                )
                if count != p.EXPECTED_DATA_FILES:
                    raise RuntimeError(f"manifest data-file count {count}")
                return text
            except Exception as exc:
                errors.append(f"round={round_index} url={url} error={exc!r}")
        time.sleep(min(5 * round_index, 60))
    raise RuntimeError("Unable to fetch official manifest: " + " | ".join(errors[-10:]))


p.fetch_manifest_text = robust_manifest_text

if __name__ == "__main__":
    remote.shard(2, 12, Path("recovered-shard-2"))
