#!/usr/bin/env python3
"""
Download & verify the UNSW-NB15 dataset (partitioned train/test CSV version).

Usage
-----
    python -m src.preprocessing.download_data          # from project root
    python src/preprocessing/download_data.py          # direct invocation

The script will:
  1. Try to download via the Kaggle CLI (``kaggle datasets download``).
  2. If that fails, print clear manual-download instructions.
  3. Verify that the CSVs exist, check their shapes, and optionally
     compare MD5 checksums.

Expected files in ``data/raw/``:
    UNSW_NB15_training-set.csv   (175,341 rows × 45 columns)
    UNSW_NB15_testing-set.csv    ( 82,332 rows × 45 columns)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

KAGGLE_DATASET = "mrwellsdavid/unsw-nb15"

EXPECTED_FILES = {
    "UNSW_NB15_training-set.csv": {
        "rows": 175_341,
        "cols": 45,
        # MD5 may vary across mirrors; set to None to skip checksum
        "md5": None,
    },
    "UNSW_NB15_testing-set.csv": {
        "rows": 82_332,
        "cols": 45,
        "md5": None,
    },
}

# Alternative download sources (in priority order)
DOWNLOAD_URLS = [
    {
        "source": "Kaggle (recommended)",
        "url": "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15",
        "instructions": (
            "1. Go to the URL above and sign in to Kaggle.\n"
            "2. Click the 'Download' button (down-arrow icon) to get the zip.\n"
            "3. Extract UNSW_NB15_training-set.csv and\n"
            "   UNSW_NB15_testing-set.csv into data/raw/."
        ),
    },
    {
        "source": "UNSW Canberra (CloudStor / AARNet)",
        "url": (
            "https://cloudstor.aarnet.edu.au/plus/index.php/s/"
            "2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files"
            "%2Fa%20part%20of%20training%20and%20testing%20set"
        ),
        "instructions": (
            "1. Open the URL above in a browser.\n"
            "2. Download the two CSV files into data/raw/."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _md5(filepath: Path) -> str:
    """Return the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_manual_instructions() -> None:
    """Pretty-print manual download instructions."""
    print("\n" + "=" * 70)
    print("  MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    for src in DOWNLOAD_URLS:
        print(f"\n[*]  {src['source']}")
        print(f"    URL: {src['url']}")
        print()
        for line in src["instructions"].splitlines():
            print(f"    {line}")
    print()
    print(
        f"After downloading, place the CSVs in:\n"
        f"    {RAW_DIR.as_posix()}\n"
    )
    print(
        "Then re-run this script to verify:\n"
        "    python -m src.preprocessing.download_data --verify-only\n"
    )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Download via Kaggle CLI
# ---------------------------------------------------------------------------
def try_kaggle_download() -> bool:
    """
    Attempt to download the dataset using the ``kaggle`` CLI.

    Returns True on success, False otherwise.
    Requires:
        - ``pip install kaggle``
        - A valid ``~/.kaggle/kaggle.json`` API token.
    """
    if shutil.which("kaggle") is None:
        print("[INFO] Kaggle CLI not found on PATH. Skipping API download.")
        return False

    print(f"[INFO] Attempting Kaggle API download: {KAGGLE_DATASET}")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(RAW_DIR),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[INFO] Kaggle download completed successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Kaggle download failed: {exc.stderr.strip()}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Kaggle download error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_dataset() -> bool:
    """
    Check that the expected CSV files exist with the right shape.

    Returns True if all checks pass.
    """
    print("\n" + "-" * 50)
    print("  DATASET VERIFICATION")
    print("-" * 50)

    all_ok = True

    for filename, meta in EXPECTED_FILES.items():
        filepath = RAW_DIR / filename
        print(f"\n[FILE]  {filename}")

        # --- existence ---
        if not filepath.is_file():
            print(f"    [X]  File NOT FOUND at {filepath}")
            all_ok = False
            continue
        print(f"    [OK]  Found ({filepath.stat().st_size / 1e6:.1f} MB)")

        # --- shape ---
        df = pd.read_csv(filepath, nrows=0)  # header only for column count
        n_cols = len(df.columns)
        n_rows = sum(1 for _ in open(filepath, encoding="utf-8")) - 1  # minus header

        shape_ok = True
        if n_rows != meta["rows"]:
            print(f"    [!]  Row count: {n_rows:,} (expected {meta['rows']:,})")
            shape_ok = False
        if n_cols != meta["cols"]:
            print(f"    [!]  Col count: {n_cols} (expected {meta['cols']})")
            shape_ok = False
        if shape_ok:
            print(f"    [OK]  Shape: ({n_rows:,} x {n_cols})")

        # --- checksum ---
        if meta["md5"] is not None:
            file_md5 = _md5(filepath)
            if file_md5 == meta["md5"]:
                print(f"    [OK]  MD5: {file_md5}")
            else:
                print(f"    [!]  MD5 mismatch: {file_md5} (expected {meta['md5']})")
                all_ok = False
        else:
            print("    [i]  MD5 check skipped (no reference checksum set)")

    print()
    if all_ok:
        print("[OK]  All checks passed -- dataset is ready!")
    else:
        print("[!]  Some checks failed. See messages above.")
    print("-" * 50 + "\n")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    verify_only = "--verify-only" in sys.argv

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not verify_only:
        # Check if files already exist
        all_present = all(
            (RAW_DIR / fn).is_file() for fn in EXPECTED_FILES
        )
        if all_present:
            print("[INFO] Dataset files already present in data/raw/.")
        else:
            success = try_kaggle_download()
            if not success:
                _print_manual_instructions()

    verify_dataset()


if __name__ == "__main__":
    main()
