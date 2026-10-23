"""
Run the preprocessing pipeline end-to-end.

Usage
-----
    python -m src.preprocessing.run_pipeline                 # default
    python -m src.preprocessing.run_pipeline --drop-leakage  # also drop ct_state_ttl
    python -m src.preprocessing.run_pipeline --drop-features ct_state_ttl --output-dir data/processed_no_ttl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import UNSW_NB15_Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 preprocessing pipeline")
    parser.add_argument(
        "--drop-leakage",
        action="store_true",
        help="Also drop leakage-flagged columns (ct_state_ttl).",
    )
    parser.add_argument(
        "--drop-features",
        nargs="+",
        default=[],
        metavar="COLUMN",
        help="Input columns to drop (default: none). Requires --output-dir other than data/processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Where to write the processed arrays and pipeline (default: data/processed).",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if args.drop_features and out_dir == PROCESSED_DIR.resolve():
        parser.error("--drop-features must write to a new --output-dir, not data/processed")

    # ── Load raw CSVs ────────────────────────────────────────────────────
    train_path = RAW_DIR / "UNSW_NB15_training-set.csv"
    test_path = RAW_DIR / "UNSW_NB15_testing-set.csv"

    if not train_path.is_file() or not test_path.is_file():
        logger.error(
            "Dataset not found in %s. Run download_data.py first.", RAW_DIR
        )
        sys.exit(1)

    logger.info("Loading training set: %s", train_path)
    train_df = pd.read_csv(train_path)
    logger.info("Loading testing set:  %s", test_path)
    test_df = pd.read_csv(test_path)
    logger.info("Train: %s  |  Test: %s", train_df.shape, test_df.shape)

    # ── Fit on train, transform both ─────────────────────────────────────
    pipe = UNSW_NB15_Pipeline(drop_leakage=args.drop_leakage, drop_features=tuple(args.drop_features))

    X_train, y_train, y_cat_train = pipe.fit_transform(train_df)
    X_test, y_test, y_cat_test = pipe.transform(test_df)

    logger.info("X_train: %s  |  X_test: %s", X_train.shape, X_test.shape)
    logger.info("NaN in X_train: %d  |  NaN in X_test: %d",
                int(np.isnan(X_train).sum()), int(np.isnan(X_test).sum()))
    logger.info("Inf in X_train: %d  |  Inf in X_test: %d",
                int(np.isinf(X_train).sum()), int(np.isinf(X_test).sum()))

    # ── Save artefacts ───────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "y_cat_train.npy", y_cat_train)
    np.save(out_dir / "y_cat_test.npy", y_cat_test)

    pipe_path = pipe.save(out_dir)

    # ── Save canonical class list (single source of truth) ───────────────
    le = pipe.label_encoder
    if le is not None:
        # Exclude the <UNKNOWN> sentinel — it's only used for unseen-
        # category handling inside the pipeline, never as a real label.
        real_classes = sorted([c for c in le.classes_ if c != "<UNKNOWN>"])
    else:
        real_classes = sorted(
            set(np.unique(y_cat_train).tolist())
            | set(np.unique(y_cat_test).tolist())
        )
    classes_path = out_dir / "label_classes.json"
    with open(classes_path, "w") as f:
        json.dump(real_classes, f, indent=2)

    logger.info("Saved processed arrays + pipeline to %s", out_dir)
    logger.info("Pipeline artefact: %s", pipe_path)
    logger.info("Label classes (%d): %s", len(real_classes), real_classes)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Features:   {X_train.shape[1]}")
    print(f"  Train rows: {X_train.shape[0]:,}")
    print(f"  Test rows:  {X_test.shape[0]:,}")
    print(f"  Output dir: {out_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
