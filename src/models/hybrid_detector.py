"""
Hybrid detector: weighted combination of Autoencoder + Isolation Forest scores.

  hybrid_score = ae_weight * ae_score + if_weight * if_score

Weights come from config.yaml (default 0.6 / 0.4).
The Youden's J threshold is computed on training data and stored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve
import joblib

from src.models.autoencoder import AutoencoderModel
from src.models.isolation_forest import IFModel


class HybridDetector:
    """Combines AE + IF scores into a single hybrid anomaly score."""

    def __init__(
        self,
        ae_model: AutoencoderModel,
        if_model: IFModel,
        ae_weight: float = 0.6,
        if_weight: float = 0.4,
    ):
        self.ae = ae_model
        self.ifm = if_model
        self.ae_weight = ae_weight
        self.if_weight = if_weight
        self.threshold: float | None = None  # Youden's J threshold

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------
    def hybrid_score(
        self,
        X: np.ndarray,
        fit_scalers: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute hybrid score for each sample.

        Returns
        -------
        hybrid_scores : (n,) float32 in [0,1]
        ae_scores     : (n,) float32 AE anomaly score
        if_scores     : (n,) float32 IF anomaly score
        """
        ae_scores = self.ae.score(X, fit_scaler=fit_scalers)
        if_scores = self.ifm.score(X, fit_scaler=fit_scalers)
        hybrid = self.ae_weight * ae_scores + self.if_weight * if_scores
        return hybrid.astype(np.float32), ae_scores, if_scores

    # ------------------------------------------------------------------
    # Threshold calibration (Youden's J on training / validation set)
    # ------------------------------------------------------------------
    def calibrate_threshold(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> float:
        """
        Find the Youden's J optimal threshold using the provided labelled set.
        Stores result in self.threshold and returns it.
        """
        scores, _, _ = self.hybrid_score(X)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        idx = np.argmax(tpr - fpr)
        self.threshold = float(thresholds[idx])
        print(f"Youden's J threshold calibrated: {self.threshold:.4f}")
        return self.threshold

    # ------------------------------------------------------------------
    # Prediction (raw hybrid, no agreement layer)
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions using the calibrated threshold."""
        if self.threshold is None:
            raise RuntimeError("Call calibrate_threshold() before predict().")
        scores, _, _ = self.hybrid_score(X)
        return (scores >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.ae.save(model_dir)
        self.ifm.save(model_dir)
        meta = {
            "ae_weight":  self.ae_weight,
            "if_weight":  self.if_weight,
            "threshold":  self.threshold,
        }
        joblib.dump(meta, model_dir / "hybrid_meta.pkl")

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.ae.load(model_dir)
        self.ifm.load(model_dir)
        meta_path = model_dir / "hybrid_meta.pkl"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.ae_weight = meta.get("ae_weight", self.ae_weight)
            self.if_weight = meta.get("if_weight", self.if_weight)
            self.threshold  = meta.get("threshold")
