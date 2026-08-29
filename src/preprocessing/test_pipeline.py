"""
Unit tests for the UNSW-NB15 preprocessing pipeline.

Run with:
    python -m pytest src/preprocessing/test_pipeline.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import (
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    LEAKAGE_FLAGGED,
    UNSW_NB15_Pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic data that mirrors the UNSW-NB15 schema
# ---------------------------------------------------------------------------

# Realistic UNSW-NB15 feature columns (excluding id, label, attack_cat)
_NUM_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_srv_dst", "is_sm_ips_ports",
    "tcprtt", "synack", "ackdat",
]


def _make_df(
    n: int = 200,
    *,
    seed: int = 42,
    add_nan: bool = False,
    add_inf: bool = False,
    add_unseen_cat: bool = False,
) -> pd.DataFrame:
    """Create a synthetic dataframe mimicking UNSW-NB15."""
    rng = np.random.RandomState(seed)

    data: dict[str, object] = {"id": np.arange(n)}

    # Categorical features
    data["proto"] = rng.choice(["tcp", "udp", "icmp", "arp"], size=n)
    data["state"] = rng.choice(["FIN", "CON", "INT", "RST", "REQ"], size=n)
    data["service"] = rng.choice(["http", "ftp", "dns", "ssh", "-"], size=n)

    # Numeric features
    for col in _NUM_FEATURES:
        data[col] = rng.randn(n) * 10 + 50

    # Labels
    data["attack_cat"] = rng.choice(
        ["Normal", "DoS", "Exploits", "Generic", "Reconnaissance"],
        size=n,
    )
    data["label"] = np.where(
        np.array(data["attack_cat"]) == "Normal", 0, 1
    )

    df = pd.DataFrame(data)

    if add_nan:
        # Inject NaN into ~5 % of numeric cells
        mask = rng.rand(n, len(_NUM_FEATURES)) < 0.05
        for i, col in enumerate(_NUM_FEATURES):
            df.loc[mask[:, i], col] = np.nan

    if add_inf:
        # Inject a few infinities
        df.loc[0, "sload"] = np.inf
        df.loc[1, "dload"] = -np.inf
        df.loc[2, "rate"] = np.inf

    if add_unseen_cat:
        df.loc[0, "proto"] = "sctp"       # unseen protocol
        df.loc[1, "service"] = "snmp"      # unseen service

    return df


@pytest.fixture
def train_df() -> pd.DataFrame:
    return _make_df(300, seed=1, add_nan=True, add_inf=True)


@pytest.fixture
def test_df() -> pd.DataFrame:
    return _make_df(100, seed=2, add_nan=True, add_inf=True, add_unseen_cat=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineFitTransform:
    """Core fit / transform behaviour."""

    def test_fit_transform_returns_correct_shapes(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        X_train, y_train, y_cat_train = pipe.fit_transform(train_df)
        X_test, y_test, y_cat_test = pipe.transform(test_df)

        n_train = len(train_df)
        n_test = len(test_df)

        assert X_train.shape[0] == n_train
        assert X_test.shape[0] == n_test
        # Same number of features
        assert X_train.shape[1] == X_test.shape[1]
        # Targets match row count
        assert y_train.shape == (n_train,)
        assert y_test.shape == (n_test,)
        assert y_cat_train.shape == (n_train,)
        assert y_cat_test.shape == (n_test,)

    def test_no_nan_after_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        X_train, _, _ = pipe.fit_transform(train_df)
        X_test, _, _ = pipe.transform(test_df)

        assert not np.isnan(X_train).any(), "NaN found in X_train"
        assert not np.isnan(X_test).any(), "NaN found in X_test"

    def test_no_inf_after_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        X_train, _, _ = pipe.fit_transform(train_df)
        X_test, _, _ = pipe.transform(test_df)

        assert not np.isinf(X_train).any(), "Inf found in X_train"
        assert not np.isinf(X_test).any(), "Inf found in X_test"

    def test_dropped_columns_not_in_features(
        self, train_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        pipe.fit_transform(train_df)

        for col in DROP_COLUMNS:
            assert col not in pipe.feature_names, f"{col} should be dropped"

    def test_leakage_columns_present_by_default(
        self, train_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline(drop_leakage=False)
        pipe.fit_transform(train_df)

        for col in LEAKAGE_FLAGGED:
            assert col in pipe.feature_names, (
                f"{col} should be kept when drop_leakage=False"
            )

    def test_leakage_columns_dropped_when_requested(
        self, train_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline(drop_leakage=True)
        pipe.fit_transform(train_df)

        for col in LEAKAGE_FLAGGED:
            assert col not in pipe.feature_names, (
                f"{col} should be dropped when drop_leakage=True"
            )

    def test_drop_leakage_reduces_feature_count(
        self, train_df: pd.DataFrame
    ):
        pipe_keep = UNSW_NB15_Pipeline(drop_leakage=False)
        X_keep, _, _ = pipe_keep.fit_transform(train_df)

        pipe_drop = UNSW_NB15_Pipeline(drop_leakage=True)
        X_drop, _, _ = pipe_drop.fit_transform(train_df)

        assert X_drop.shape[1] < X_keep.shape[1]


class TestPipelineLabels:
    """Target extraction."""

    def test_binary_labels_are_0_or_1(self, train_df: pd.DataFrame):
        pipe = UNSW_NB15_Pipeline()
        _, y, _ = pipe.fit_transform(train_df)
        assert set(np.unique(y)).issubset({0, 1})

    def test_attack_cat_labels_are_encoded_integers(self, train_df: pd.DataFrame):
        pipe = UNSW_NB15_Pipeline()
        _, _, y_cat = pipe.fit_transform(train_df)
        assert y_cat.dtype.kind in ("i", "u")  # Integer


class TestPipelineScaling:
    """StandardScaler behaviour."""

    def test_train_numeric_features_are_standardised(
        self, train_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        X, _, _ = pipe.fit_transform(train_df)

        # Numeric columns should have ~0 mean, ~1 std
        num_X = X[:, pipe._num_indices]
        means = num_X.mean(axis=0)
        stds = num_X.std(axis=0)

        np.testing.assert_allclose(means, 0.0, atol=0.1)
        # Some constant columns may have std=0; allow those
        non_const = stds > 0.01
        np.testing.assert_allclose(stds[non_const], 1.0, atol=0.15)


class TestPipelineCategoricals:
    """Categorical encoding."""

    def test_unseen_categories_handled(self, train_df: pd.DataFrame):
        """Unseen categories at transform time should not raise."""
        pipe = UNSW_NB15_Pipeline()
        pipe.fit_transform(train_df)

        test = _make_df(50, seed=99, add_unseen_cat=True)
        X, _, _ = pipe.transform(test)

        assert not np.isnan(X).any()
        assert X.shape[0] == 50

    def test_categorical_columns_are_encoded_as_numbers(
        self, train_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        X, _, _ = pipe.fit_transform(train_df)

        # All values should be finite floats
        assert X.dtype == np.float64


class TestPipelinePersistence:
    """Save / load round-trip."""

    def test_save_load_produces_identical_output(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        pipe.fit_transform(train_df)
        X_orig, y_orig, yc_orig = pipe.transform(test_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save(tmpdir)
            loaded = UNSW_NB15_Pipeline.load(Path(tmpdir) / "pipeline.joblib")

        X_loaded, y_loaded, yc_loaded = loaded.transform(test_df)

        np.testing.assert_array_equal(X_orig, X_loaded)
        np.testing.assert_array_equal(y_orig, y_loaded)
        np.testing.assert_array_equal(yc_orig, yc_loaded)

    def test_unfitted_pipeline_cannot_save(self):
        pipe = UNSW_NB15_Pipeline()
        with pytest.raises(RuntimeError, match="unfitted"):
            pipe.save("/tmp/should_not_exist")


class TestPipelineEdgeCases:
    """Edge cases and error handling."""

    def test_unfitted_transform_raises(self):
        pipe = UNSW_NB15_Pipeline()
        df = _make_df(10)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.transform(df)

    def test_all_nan_column_handled(self, train_df: pd.DataFrame):
        """A column that is entirely NaN should still produce valid output."""
        train_df["dur"] = np.nan
        pipe = UNSW_NB15_Pipeline()
        X, _, _ = pipe.fit_transform(train_df)

        assert not np.isnan(X).any()

    def test_consistent_shape_across_multiple_transforms(
        self, train_df: pd.DataFrame
    ):
        pipe = UNSW_NB15_Pipeline()
        X1, _, _ = pipe.fit_transform(train_df)
        X2, _, _ = pipe.transform(train_df)

        assert X1.shape == X2.shape
        np.testing.assert_array_almost_equal(X1, X2)
