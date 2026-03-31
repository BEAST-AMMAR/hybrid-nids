"""
Preprocessing pipeline for the NSL-KDD dataset.

Mirrors the logic from notebooks/pre_processing.ipynb so it can be
called from training scripts, the inference pipeline, and tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils import get_project_root


# -----------------------------------------------------------------
# Column definitions (NSL-KDD raw 43-column format)
# -----------------------------------------------------------------
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_COLS = ["difficulty"]


def load_raw(filepath: str | Path) -> pd.DataFrame:
    """Load a raw NSL-KDD .txt file into a DataFrame."""
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    return df


def preprocess_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    train_df : raw training DataFrame (with 'label' and 'difficulty' columns)
    test_df  : raw test DataFrame
    scaler   : a pre-fitted StandardScaler (use None to fit a new one on train)

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler, feature_columns
    """
    for df in (train_df, test_df):
        df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # Binary labels: normal → 0, everything else → 1
    y_train = (train_df["label"].str.strip() != "normal").astype(int).values
    y_test  = (test_df["label"].str.strip()  != "normal").astype(int).values

    train_df = train_df.drop(columns=["label"])
    test_df  = test_df.drop(columns=["label"])

    # One-hot encode categorical columns
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLS)
    test_df  = pd.get_dummies(test_df,  columns=CATEGORICAL_COLS)

    # Align test to train columns (left join — fill missing OHE cols with 0)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    feature_columns = list(train_df.columns)
    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns


# -----------------------------------------------------------------
# Convenience: load the .npy files saved by notebooks/
# -----------------------------------------------------------------
def load_preprocessed(data_dir: str | Path | None = None) -> dict:
    """
    Load the pre-saved .npy arrays from notebooks/.

    Returns a dict with keys:
      X_train, X_test, X_train_normal, y_train, y_test
    """
    if data_dir is None:
        data_dir = get_project_root() / "notebooks"
    data_dir = Path(data_dir)

    return {
        "X_train":        np.load(data_dir / "X_train_scaled.npy"),
        "X_test":         np.load(data_dir / "X_test_scaled.npy"),
        "X_train_normal": np.load(data_dir / "X_train_normal.npy"),
        "y_train":        np.load(data_dir / "y_train.npy"),
        "y_test":         np.load(data_dir / "y_test.npy"),
    }


# -----------------------------------------------------------------
# Convenience: preprocess a single raw dict / DataFrame row
# -----------------------------------------------------------------
def preprocess_single(
    row: dict,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> np.ndarray:
    """
    Preprocess a single sample (dict of raw feature values) for inference.

    The row must contain all raw KDD feature names (minus 'label' and
    'difficulty'). Returns a (1, n_features) float32 array.
    """
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
    df = df.reindex(columns=feature_columns, fill_value=0)
    X = scaler.transform(df.values.astype(np.float32))
    return X
