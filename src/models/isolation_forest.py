"""
Isolation Forest wrapper for anomaly detection.

Trained on the FULL (normal + attack) scaled training set.
The decision_function score is negated and normalised to [0,1] so
that higher score means more anomalous — consistent with the AE score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib


class IFModel:
    """Sklearn Isolation Forest wrapper."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model: IsolationForest | None = None
        self.score_scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, verbose: bool = True) -> None:
        cfg = self.config
        self.model = IsolationForest(
            n_estimators  = cfg.get("n_estimators",  200),
            contamination = cfg.get("contamination",  0.1),
            random_state  = cfg.get("random_state",   42),
            n_jobs        = -1,
        )
        if verbose:
            print("Training Isolation Forest...")
        self.model.fit(X_train)
        if verbose:
            print("Isolation Forest training complete.")

    # ------------------------------------------------------------------
    # Scoring — returns per-sample anomaly score in [0, 1]
    # ------------------------------------------------------------------
    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Return negated decision_function (higher = more anomalous)."""
        return -self.model.decision_function(X)

    def score(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Return anomaly scores normalised to [0, 1].

        Parameters
        ----------
        X          : scaled feature array
        fit_scaler : True when scoring training set (fits the MinMaxScaler)
        """
        raw = self.raw_score(X)
        if fit_scaler or self.score_scaler is None:
            self.score_scaler = MinMaxScaler()
            scores = self.score_scaler.fit_transform(raw.reshape(-1, 1)).ravel()
        else:
            scores = self.score_scaler.transform(raw.reshape(-1, 1)).ravel()
        return scores.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dir / "isolation_forest.pkl")
        if self.score_scaler is not None:
            joblib.dump(self.score_scaler, model_dir / "if_score_scaler.pkl")

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "isolation_forest.pkl")
        scaler_path = model_dir / "if_score_scaler.pkl"
        if scaler_path.exists():
            self.score_scaler = joblib.load(scaler_path)
