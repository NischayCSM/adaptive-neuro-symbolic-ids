"""
Reusable preprocessing pipeline for the UNSW-NB15 dataset.

Usage
-----
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline

    pipe = UNSW_NB15_Pipeline()
    X_train, y_train, y_cat_train = pipe.fit_transform(train_df)
    X_test,  y_test,  y_cat_test  = pipe.transform(test_df)
    pipe.save("data/processed/")

Design decisions (from EDA findings)
-------------------------------------
- **Dropped columns**: ``id`` (row index), ``attack_cat`` (leakage — *is*
  the multi-class label), ``label`` (extracted as target, not a feature).
- **Flagged columns**: ``ct_state_ttl`` is kept but flagged via
  ``LEAKAGE_FLAGGED`` so callers can optionally exclude it.
- **Categorical encoding**: ``proto``, ``state``, ``service`` are
  ordinal-encoded with an ``<UNKNOWN>`` category for unseen values at
  test time.  One-hot is available as an option.
- **Infinite values**: replaced with NaN, then imputed.
- **Missing values**: median imputation (fitted on train).
- **Scaling**: ``StandardScaler`` fitted on train only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Columns that should never be used as input features.
DROP_COLUMNS: list[str] = ["id", "attack_cat", "label"]

# Columns identified as potential label-leakage in the EDA notebook.
# They are *not* dropped by default, but flagged so callers can decide.
LEAKAGE_FLAGGED: list[str] = ["ct_state_ttl"]

# Columns that require categorical encoding.
CATEGORICAL_COLUMNS: list[str] = ["proto", "state", "service"]

# Sentinel used for unseen categories at transform time.
UNKNOWN_CATEGORY = "<UNKNOWN>"


# ── Pipeline ─────────────────────────────────────────────────────────────────

@dataclass
class UNSW_NB15_Pipeline:
    """End-to-end preprocessing pipeline for UNSW-NB15.

    Parameters
    ----------
    drop_leakage : bool
        If *True*, also drop ``LEAKAGE_FLAGGED`` columns. Default *False*
        (columns are kept but callers are warned).
    encoding : str
        ``"ordinal"`` (default) or ``"onehot"`` for categorical features.
    """

    drop_leakage: bool = False
    encoding: str = "ordinal"

    # ── Fitted state (populated by `fit`) ────────────────────────────────
    _is_fitted: bool = field(default=False, init=False, repr=False)
    _encoder: Optional[OrdinalEncoder] = field(default=None, init=False, repr=False)
    _imputer: Optional[SimpleImputer] = field(default=None, init=False, repr=False)
    _scaler: Optional[StandardScaler] = field(default=None, init=False, repr=False)
    _label_encoder: Optional[LabelEncoder] = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)
    _cat_indices: list[int] = field(default_factory=list, init=False, repr=False)
    _num_indices: list[int] = field(default_factory=list, init=False, repr=False)

    @property
    def label_encoder(self) -> Optional[LabelEncoder]:
        return self._label_encoder

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "UNSW_NB15_Pipeline":
        """Fit encoders, imputer, and scaler on the **training** data."""
        X, _, y_cat_raw = self._separate(df)

        # 0. Fit LabelEncoder for attack_cat
        self._label_encoder = LabelEncoder()
        unique_cats = np.unique(y_cat_raw).tolist()
        if UNKNOWN_CATEGORY not in unique_cats:
            unique_cats.append(UNKNOWN_CATEGORY)
        self._label_encoder.fit(unique_cats)

        # 1. Identify column types
        self._feature_names = X.columns.tolist()
        self._cat_indices = [
            i for i, c in enumerate(self._feature_names)
            if c in CATEGORICAL_COLUMNS
        ]
        self._num_indices = [
            i for i in range(len(self._feature_names))
            if i not in self._cat_indices
        ]

        # 2. Encode categoricals
        cat_cols = [self._feature_names[i] for i in self._cat_indices]
        if cat_cols:
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            X[cat_cols] = X[cat_cols].fillna(UNKNOWN_CATEGORY).astype(str)
            self._encoder.fit(X[cat_cols])
            logger.info("Fitted OrdinalEncoder on %s", cat_cols)

        # 3. Replace infinities → NaN, then impute
        arr = self._encode_categoricals(X)
        arr = np.where(np.isinf(arr), np.nan, arr)
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self._imputer.fit(arr)
        arr = self._imputer.transform(arr)
        logger.info("Fitted SimpleImputer (median)")

        # 4. Scale numeric columns only (leave encoded cats as-is)
        self._scaler = StandardScaler()
        self._scaler.fit(arr[:, self._num_indices])
        logger.info("Fitted StandardScaler on %d numeric features", len(self._num_indices))

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform a dataframe using the **fitted** pipeline.

        Returns
        -------
        X : np.ndarray, shape (n, d)
            Processed feature matrix (float64, no NaN / inf).
        y_binary : np.ndarray, shape (n,)
            Binary label (0 = Normal, 1 = Attack).
        y_cat : np.ndarray, shape (n,)
            Multi-class attack category strings.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit() or fit_transform() first."
            )

        X, y_binary, y_cat = self._separate(df)

        # Encode categoricals
        arr = self._encode_categoricals(X)

        # Replace inf → NaN, then impute
        arr = np.where(np.isinf(arr), np.nan, arr)
        arr = self._imputer.transform(arr)

        # Scale numeric columns
        arr[:, self._num_indices] = self._scaler.transform(
            arr[:, self._num_indices]
        )

        # Encode attack categories to integers
        known_cats = set(self._label_encoder.classes_)
        y_cat_safe = np.array([c if c in known_cats else UNKNOWN_CATEGORY for c in y_cat])
        y_cat_encoded = self._label_encoder.transform(y_cat_safe)

        # Final safety check
        if np.isnan(arr).any():
            n_nan = int(np.isnan(arr).sum())
            logger.warning("%d NaN values remain after transform!", n_nan)

        return arr, y_binary, y_cat_encoded

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: fit then transform on the same dataframe."""
        self.fit(df)
        return self.transform(df)

    def save(self, directory: str | Path) -> Path:
        """Persist fitted pipeline artefacts with joblib.

        Saves a single file ``pipeline.joblib`` containing all fitted
        state needed to transform new data.

        Returns the path to the saved file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted pipeline.")

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pipeline.joblib"

        state = {
            "drop_leakage": self.drop_leakage,
            "encoding": self.encoding,
            "encoder": self._encoder,
            "imputer": self._imputer,
            "scaler": self._scaler,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "cat_indices": self._cat_indices,
            "num_indices": self._num_indices,
        }
        joblib.dump(state, out_path)
        logger.info("Pipeline saved to %s", out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "UNSW_NB15_Pipeline":
        """Load a previously saved pipeline."""
        state = joblib.load(path)
        pipe = cls(
            drop_leakage=state["drop_leakage"],
            encoding=state["encoding"],
        )
        pipe._encoder = state["encoder"]
        pipe._imputer = state["imputer"]
        pipe._scaler = state["scaler"]
        pipe._label_encoder = state.get("label_encoder", None)
        pipe._feature_names = state["feature_names"]
        pipe._cat_indices = state["cat_indices"]
        pipe._num_indices = state["num_indices"]
        pipe._is_fitted = True
        return pipe

    @property
    def feature_names(self) -> list[str]:
        """Column names of the output feature matrix (post-encoding)."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted.")
        return list(self._feature_names)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _columns_to_drop(self) -> list[str]:
        """Return the full list of columns to exclude from features."""
        cols = list(DROP_COLUMNS)
        if self.drop_leakage:
            cols.extend(LEAKAGE_FLAGGED)
        return cols

    def _separate(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Split a raw dataframe into X, y_binary, y_cat."""
        df = df.copy()

        # Standardise attack_cat
        if "attack_cat" in df.columns:
            df["attack_cat"] = (
                df["attack_cat"]
                .astype(str)
                .str.strip()
                .str.title()
            )
            df.loc[
                df["attack_cat"].isin(["Nan", "", "None", "Normal "]),
                "attack_cat",
            ] = "Normal"
            if "label" in df.columns:
                df.loc[df["label"] == 0, "attack_cat"] = "Normal"

        # Extract targets
        y_binary = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)
        y_cat = df["attack_cat"].values if "attack_cat" in df.columns else np.full(len(df), "Unknown")

        # Drop non-feature columns
        drop = [c for c in self._columns_to_drop() if c in df.columns]
        X = df.drop(columns=drop)

        # Also drop any _split helper column from the EDA notebook
        if "_split" in X.columns:
            X = X.drop(columns=["_split"])

        return X, y_binary, y_cat

    def _encode_categoricals(self, X: pd.DataFrame) -> np.ndarray:
        """Apply ordinal encoding to categorical columns, return ndarray."""
        X = X.copy()
        cat_cols = [self._feature_names[i] for i in self._cat_indices]

        if cat_cols and self._encoder is not None:
            X[cat_cols] = X[cat_cols].fillna(UNKNOWN_CATEGORY).astype(str)
            X[cat_cols] = self._encoder.transform(X[cat_cols])

        # Convert everything to float64
        arr = X.values.astype(np.float64)
        return arr
