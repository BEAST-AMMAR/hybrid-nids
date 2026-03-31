"""
src/pipeline.py
===============
End-to-end NIDS detection pipeline.

Loads all trained models from models/ and provides a clean API for:
  - Batch inference on a scaled numpy array (detect)
  - Batch inference on a raw CSV / DataFrame (detect_raw)
  - Single-sample inference (detect_single)

Usage example:
    from src.pipeline import NIDSPipeline
    pipeline = NIDSPipeline()
    df_results = pipeline.detect(X_scaled)          # numpy input
    df_results = pipeline.detect_raw(df_features)   # raw DataFrame input
    result     = pipeline.detect_single(row_dict)   # single dict input
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from src.utils import load_config, get_project_root
from src.models.autoencoder import AutoencoderModel
from src.models.isolation_forest import IFModel
from src.models.hybrid_detector import HybridDetector
from src.models.agreement_layer import AgreementLayer
from src.preprocessing import preprocess_single, CATEGORICAL_COLS


class NIDSPipeline:
    """
    Loads all trained models and runs full tiered detection.

    Parameters
    ----------
    model_dir : path to the directory containing saved model files.
                Defaults to <project_root>/models
    config    : optional config dict override (loads config.yaml otherwise)
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        config: dict | None = None,
    ):
        root = get_project_root()
        self.config = config or load_config()
        self.model_dir = Path(model_dir) if model_dir else root / "models"
        self._feature_columns: list[str] | None = None

        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        cfg = self.config
        ae_cfg  = cfg.get("autoencoder", {})
        hyb_cfg = cfg.get("hybrid", {})
        agr_cfg = cfg.get("agreement", {})
        xgb_cfg = cfg.get("xgboost_arbitrator", {})

        self.ae  = AutoencoderModel(config=ae_cfg)
        self.ifm = IFModel()
        self.hybrid = HybridDetector(
            ae_model  = self.ae,
            if_model  = self.ifm,
            ae_weight = hyb_cfg.get("ae_weight", 0.6),
            if_weight = hyb_cfg.get("if_weight", 0.4),
        )
        self.hybrid.load(self.model_dir)

        self.layer = AgreementLayer(
            uncertainty_factor = agr_cfg.get("uncertainty_factor", 0.5),
            config             = xgb_cfg,
        )
        self.layer.load(self.model_dir)

        # Load StandardScaler if available
        scaler_path = self.model_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        # Load feature columns if available
        feat_path = self.model_dir / "feature_columns.pkl"
        if feat_path.exists():
            self._feature_columns = joblib.load(feat_path)

    # ------------------------------------------------------------------
    # Core detection (expects already-scaled numpy array)
    # ------------------------------------------------------------------
    def detect(self, X: np.ndarray) -> pd.DataFrame:
        """
        Run full detection on a pre-scaled feature array.

        Parameters
        ----------
        X : (n, 116) float32 scaled feature array

        Returns
        -------
        DataFrame with columns:
          final_pred, confidence, tier_used, agreement_score,
          hybrid_score, ae_score, if_score, ae_pred, if_pred, xgb_pred
        """
        hybrid_scores, ae_scores, if_scores = self.hybrid.hybrid_score(X)
        results_df = self.layer.predict_df(
            X, hybrid_scores, ae_scores, if_scores, self.hybrid.threshold
        )
        return results_df

    # ------------------------------------------------------------------
    # Detection on raw DataFrame (applies preprocessing)
    # ------------------------------------------------------------------
    def detect_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run detection on a raw feature DataFrame (no label column).

        Applies one-hot encoding and StandardScaler (must be loaded).
        """
        if self.scaler is None or self._feature_columns is None:
            raise RuntimeError(
                "scaler.pkl and feature_columns.pkl must exist in model_dir "
                "to use detect_raw(). Run scripts/train_all.py first."
            )
        df_enc = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
        df_enc = df_enc.reindex(columns=self._feature_columns, fill_value=0)
        X = self.scaler.transform(df_enc.values.astype(np.float32))
        return self.detect(X)

    # ------------------------------------------------------------------
    # Single-sample detection
    # ------------------------------------------------------------------
    def detect_single(self, row: dict[str, Any]) -> dict:
        """
        Run detection on a single raw sample dict.

        Returns a single result dict (same structure as a row from detect()).
        """
        if self.scaler is None or self._feature_columns is None:
            raise RuntimeError(
                "scaler.pkl and feature_columns.pkl must exist in model_dir "
                "to use detect_single(). Run scripts/train_all.py first."
            )
        X = preprocess_single(row, self.scaler, self._feature_columns)
        df_result = self.detect(X)
        return df_result.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Convenience: summary statistics on detect() output
    # ------------------------------------------------------------------
    @staticmethod
    def summarize(results_df: pd.DataFrame) -> dict:
        """Return summary statistics for a detect() result DataFrame."""
        n = len(results_df)
        n_attack = int(results_df["final_pred"].sum())
        n_normal = n - n_attack
        tier_counts = results_df["tier_used"].value_counts().to_dict()
        conf_counts = results_df["confidence"].value_counts().to_dict()
        return {
            "total":           n,
            "attacks":         n_attack,
            "normals":         n_normal,
            "attack_rate":     round(n_attack / n, 4) if n else 0,
            "tier_breakdown":  tier_counts,
            "confidence_dist": conf_counts,
            "avg_hybrid_score": float(results_df["hybrid_score"].mean()),
        }
