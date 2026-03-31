"""
Agreement Layer — tiered arbitration for uncertain hybrid predictions.

Tier 1: Confidence Thresholding (threshold-relative band)
    The uncertainty band is derived from the Youden's J threshold:
      low_thresh  = max(0, hybrid_threshold * (1 - uncertainty_factor))
      high_thresh = hybrid_threshold * (1 + uncertainty_factor)

    hybrid_score > high_thresh  → attack  (HIGH confidence)
    hybrid_score < low_thresh   → normal  (HIGH confidence)
    else                        → UNCERTAIN → escalate to Tier 2

    Default uncertainty_factor = 0.5, meaning the band spans
    [0.5×threshold, 1.5×threshold]. This is recalculated every call
    so it always stays in sync with the calibrated Youden's J threshold.

Tier 2: 3-Model Majority Vote
    AE prediction + IF prediction + arbitrator prediction
    Individual predictions use the hybrid_threshold directly:
      ae_pred  = 1 if ae_score  >= hybrid_threshold else 0
      if_pred  = 1 if if_score  >= hybrid_threshold else 0
      xgb_pred = arbitrator.predict(X)
    Final = majority vote (2/3)
    agreement_score = max(votes_for, votes_against) / 3
    Confidence: HIGH (3/3 = 1.0) | MEDIUM (2/3 = 0.6667) | LOW (impossible in 3-model vote)

Output per sample:
  final_pred       : int   (0 = normal, 1 = attack)
  confidence       : str   ("HIGH" | "MEDIUM" | "LOW")
  tier_used        : int   (1 or 2)
  agreement_score  : float (0.33, 0.67, or 1.0)
  hybrid_score     : float
  ae_score         : float
  if_score         : float
  ae_pred          : int
  if_pred          : int
  xgb_pred         : int   (only when tier_used == 2, else -1)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib


class AgreementLayer:
    """
    Tiered arbitration layer that sits on top of the HybridDetector.

    Parameters
    ----------
    uncertainty_factor : float
        Controls the width of the uncertain zone around the Youden's J threshold.
        Band = [threshold × (1 - factor), threshold × (1 + factor)].
        Samples within this band → Tier 2.  Default 0.5.
    config : dict
        Hyperparameter overrides for the arbitrator (XGBoost / RandomForest).
    """

    def __init__(
        self,
        uncertainty_factor: float = 0.5,
        config: dict | None = None,
    ):
        self.uncertainty_factor = uncertainty_factor
        self.config = config or {}
        self.arbitrator = None

    # ------------------------------------------------------------------
    # Derive Tier 1 band from calibrated threshold
    # ------------------------------------------------------------------
    def _band(self, hybrid_threshold: float) -> tuple[float, float]:
        """Return (low_thresh, high_thresh) relative to the Youden's J threshold."""
        low  = max(0.0, hybrid_threshold * (1.0 - self.uncertainty_factor))
        high = hybrid_threshold * (1.0 + self.uncertainty_factor)
        return low, high

    # ------------------------------------------------------------------
    # Tier 2 arbitrator training
    # ------------------------------------------------------------------
    def train_arbitrator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> None:
        """
        Train the arbitrator on the full labelled training set.
        Tries XGBoost first; falls back to RandomForestClassifier if XGBoost
        cannot be loaded (e.g. missing libomp on macOS).
        """
        cfg = self.config
        arbitrator = None

        try:
            from xgboost import XGBClassifier
            arbitrator = XGBClassifier(
                n_estimators      = cfg.get("n_estimators",  200),
                max_depth         = cfg.get("max_depth",      6),
                learning_rate     = cfg.get("learning_rate",  0.1),
                random_state      = cfg.get("random_state",   42),
                eval_metric       = "logloss",
                use_label_encoder = False,
                n_jobs            = -1,
            )
            if verbose:
                print("Training XGBoost arbitrator...")
        except Exception as xgb_err:
            if verbose:
                print(f"XGBoost unavailable ({xgb_err.__class__.__name__}), "
                      f"falling back to RandomForestClassifier.")
            from sklearn.ensemble import RandomForestClassifier
            arbitrator = RandomForestClassifier(
                n_estimators = cfg.get("n_estimators", 200),
                max_depth    = cfg.get("max_depth", None),
                random_state = cfg.get("random_state", 42),
                n_jobs       = -1,
            )
            if verbose:
                print("Training RandomForest arbitrator...")

        arbitrator.fit(X_train, y_train)
        self.arbitrator = arbitrator
        if verbose:
            print(f"Arbitrator training complete ({type(self.arbitrator).__name__}).")

    # ------------------------------------------------------------------
    # Full tiered prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        hybrid_scores: np.ndarray,
        ae_scores: np.ndarray,
        if_scores: np.ndarray,
        hybrid_threshold: float,
    ) -> List[dict]:
        """
        Run tiered prediction for all samples.

        Parameters
        ----------
        X                : (n, features) scaled feature array
        hybrid_scores    : (n,) hybrid anomaly scores from HybridDetector
        ae_scores        : (n,) AE anomaly scores (normalised [0,1])
        if_scores        : (n,) IF anomaly scores (normalised [0,1])
        hybrid_threshold : Youden's J threshold from HybridDetector.calibrate_threshold()
        """
        low_thresh, high_thresh = self._band(hybrid_threshold)

        n = len(hybrid_scores)
        # Individual model binary preds use the calibrated threshold directly
        ae_preds = (ae_scores >= hybrid_threshold).astype(int)
        if_preds = (if_scores >= hybrid_threshold).astype(int)

        # Run arbitrator once for all uncertain samples
        uncertain_mask = (hybrid_scores >= low_thresh) & (hybrid_scores <= high_thresh)
        xgb_preds = np.full(n, -1, dtype=int)
        if self.arbitrator is not None and uncertain_mask.any():
            xgb_preds[uncertain_mask] = self.arbitrator.predict(X[uncertain_mask])

        results: List[dict] = []

        for i in range(n):
            hs   = float(hybrid_scores[i])
            ae_s = float(ae_scores[i])
            if_s = float(if_scores[i])
            ae_p = int(ae_preds[i])
            if_p = int(if_preds[i])

            if hs > high_thresh:
                result = {
                    "final_pred":      1,
                    "confidence":      "HIGH",
                    "tier_used":       1,
                    "agreement_score": 1.0,
                    "hybrid_score":    hs,
                    "ae_score":        ae_s,
                    "if_score":        if_s,
                    "ae_pred":         ae_p,
                    "if_pred":         if_p,
                    "xgb_pred":        -1,
                }
            elif hs < low_thresh:
                result = {
                    "final_pred":      0,
                    "confidence":      "HIGH",
                    "tier_used":       1,
                    "agreement_score": 1.0,
                    "hybrid_score":    hs,
                    "ae_score":        ae_s,
                    "if_score":        if_s,
                    "ae_pred":         ae_p,
                    "if_pred":         if_p,
                    "xgb_pred":        -1,
                }
            else:
                # Tier 2: uncertain — 3-model majority vote
                xgb_p   = int(xgb_preds[i])
                votes    = [ae_p, if_p, xgb_p]
                n_attack = sum(votes)
                final_pred  = 1 if n_attack >= 2 else 0
                agree_score = round(max(n_attack, 3 - n_attack) / 3, 4)

                confidence = (
                    "HIGH"   if agree_score >= 1.0    else
                    "MEDIUM" if agree_score >= 0.6667 else
                    "LOW"
                )

                result = {
                    "final_pred":      final_pred,
                    "confidence":      confidence,
                    "tier_used":       2,
                    "agreement_score": agree_score,
                    "hybrid_score":    hs,
                    "ae_score":        ae_s,
                    "if_score":        if_s,
                    "ae_pred":         ae_p,
                    "if_pred":         if_p,
                    "xgb_pred":        xgb_p,
                }

            results.append(result)

        return results

    def predict_df(
        self,
        X: np.ndarray,
        hybrid_scores: np.ndarray,
        ae_scores: np.ndarray,
        if_scores: np.ndarray,
        hybrid_threshold: float,
    ) -> pd.DataFrame:
        """Same as predict() but returns a DataFrame."""
        return pd.DataFrame(
            self.predict(X, hybrid_scores, ae_scores, if_scores, hybrid_threshold)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.arbitrator is not None:
            joblib.dump(self.arbitrator, model_dir / "agreement_classifier.pkl")
        joblib.dump({"uncertainty_factor": self.uncertainty_factor},
                    model_dir / "agreement_meta.pkl")

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        clf_path = model_dir / "agreement_classifier.pkl"
        if clf_path.exists():
            self.arbitrator = joblib.load(clf_path)
        meta_path = model_dir / "agreement_meta.pkl"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.uncertainty_factor = meta.get("uncertainty_factor", self.uncertainty_factor)
