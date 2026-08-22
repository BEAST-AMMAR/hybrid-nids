import numpy as np
import tensorflow as tf
import joblib
import os


def _minmax_normalize(values, lo=0.0, hi=1.0):
    """Simple min-max clamp to [0,1] when dedicated scaler files are absent."""
    span = hi - lo
    if span == 0:
        return np.zeros_like(values)
    return np.clip((values - lo) / span, 0.0, 1.0)


class HybridDetector:
    def __init__(self, model_dir):
        # Resolve to absolute path so this works regardless of CWD
        model_dir = os.path.abspath(model_dir)

        self.autoencoder = tf.keras.models.load_model(
            os.path.join(model_dir, "autoencoder.keras")
        )
        self.iso_forest = joblib.load(os.path.join(model_dir, "iso_forest.joblib"))

        # ae_scaler / if_scaler are optional — fall back to inline normalisation
        ae_path = os.path.join(model_dir, "ae_scaler.joblib")
        if_path = os.path.join(model_dir, "if_scaler.joblib")
        self.ae_scaler = joblib.load(ae_path) if os.path.exists(ae_path) else None
        self.if_scaler = joblib.load(if_path) if os.path.exists(if_path) else None

        with open(os.path.join(model_dir, "threshold.txt"), "r") as f:
            self.threshold = float(f.read())

        self.alpha = 0.6

    def predict(self, scaled_features):
        """
        Takes scaled features and returns (is_anomaly, hybrid_score).
        """
        anomalies, scores = self.predict_batch(scaled_features)
        return anomalies[0], scores[0]

    def predict_batch(self, scaled_features_batch):
        """
        Takes a batch of scaled features and returns lists of (is_anomaly, hybrid_score).
        """
        if len(scaled_features_batch) == 0:
            return [], []
            
        # AE score (MSE)
        reconstruction = self.autoencoder.predict(scaled_features_batch, verbose=0)
        mse = np.mean(np.power(scaled_features_batch - reconstruction, 2), axis=1)

        # IF score
        if_scores = -self.iso_forest.decision_function(scaled_features_batch)

        # Normalize — use saved scalers if present, otherwise inline min-max
        if self.ae_scaler is not None:
            ae_norm = self.ae_scaler.transform(mse.reshape(-1, 1)).flatten()
        else:
            ae_norm = _minmax_normalize(mse, lo=0.0, hi=max(float(mse.max()), 1e-6))

        if self.if_scaler is not None:
            if_norm = self.if_scaler.transform(if_scores.reshape(-1, 1)).flatten()
        else:
            span = max(float(if_scores.max() - if_scores.min()), 1e-6)
            if_norm = _minmax_normalize(
                if_scores,
                lo=float(if_scores.min()),
                hi=float(if_scores.min()) + span,
            )

        # Hybrid score
        hybrid_scores = self.alpha * ae_norm + (1 - self.alpha) * if_norm
        
        is_anomalies = [bool(score > self.threshold) for score in hybrid_scores]
        scores_float = [float(score) for score in hybrid_scores]
        
        return is_anomalies, scores_float
