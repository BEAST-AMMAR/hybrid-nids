import numpy as np
import tensorflow as tf
import joblib
import os

class HybridDetector:
    def __init__(self, model_dir):
        self.autoencoder = tf.keras.models.load_model(os.path.join(model_dir, "autoencoder.keras"))
        self.iso_forest = joblib.load(os.path.join(model_dir, "iso_forest.joblib"))
        self.ae_scaler = joblib.load(os.path.join(model_dir, "ae_scaler.joblib"))
        self.if_scaler = joblib.load(os.path.join(model_dir, "if_scaler.joblib"))

        with open(os.path.join(model_dir, "threshold.txt"), "r") as f:
            self.threshold = float(f.read())

        self.alpha = 0.6

    def predict(self, scaled_features):
        """
        Takes scaled features and returns (is_anomaly, hybrid_score)
        """
        # AE score (MSE)
        reconstruction = self.autoencoder.predict(scaled_features, verbose=0)
        mse = np.mean(np.power(scaled_features - reconstruction, 2), axis=1)

        # IF score
        if_scores = -self.iso_forest.decision_function(scaled_features)

        # Normalize
        ae_norm = self.ae_scaler.transform(mse.reshape(-1,1)).flatten()
        if_norm = self.if_scaler.transform(if_scores.reshape(-1,1)).flatten()

        # Hybrid
        hybrid_score = self.alpha * ae_norm + (1 - self.alpha) * if_norm

        is_anomaly = hybrid_score[0] > self.threshold
        return bool(is_anomaly), float(hybrid_score[0])
