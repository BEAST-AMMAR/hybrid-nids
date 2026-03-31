"""
Autoencoder model for anomaly detection.

Architecture mirrors notebooks/ae_modeltraining.ipynb exactly:
  Input(116) → Dense(32) → Dense(16) → Dense(8) → Dense(16) → Dense(32) → Dense(116)
Each encoder/decoder layer has BatchNorm + Dropout(0.2).

Trained ONLY on normal traffic samples so that anomalies produce high
reconstruction error (MSE), which is used as the anomaly score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


class AutoencoderModel:
    """Keras autoencoder for anomaly detection on network traffic."""

    def __init__(self, input_dim: int = 116, config: dict | None = None):
        self.input_dim = input_dim
        self.config = config or {}
        self.model = None
        self.score_scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Architecture (matches notebook exactly)
    # ------------------------------------------------------------------
    def build(self) -> None:
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        cfg = self.config
        encoder_dims  = cfg.get("encoder_dims",  [32, 16])
        latent_dim    = cfg.get("latent_dim",     8)
        decoder_dims  = cfg.get("decoder_dims",   [16, 32])
        dropout_rate  = cfg.get("dropout_rate",   0.2)

        inp = layers.Input(shape=(self.input_dim,), name="input_features")

        # Encoder
        x = inp
        for i, dim in enumerate(encoder_dims, 1):
            x = layers.Dense(dim, activation="relu", name=f"encoder_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"encoder_bn_{i}")(x)
            x = layers.Dropout(dropout_rate, name=f"encoder_dropout_{i}")(x)

        # Bottleneck
        x = layers.Dense(latent_dim, activation="relu", name="latent_space")(x)

        # Decoder
        for i, dim in enumerate(decoder_dims, 1):
            x = layers.Dense(dim, activation="relu", name=f"decoder_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"decoder_bn_{i}")(x)
            x = layers.Dropout(dropout_rate, name=f"decoder_dropout_{i}")(x)

        out = layers.Dense(self.input_dim, activation="linear",
                           name="reconstructed_features")(x)

        self.model = Model(inp, out, name="hybrid_nids_autoencoder")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=cfg.get("learning_rate", 0.001)
            ),
            loss="mse",
            metrics=["mae"],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train_normal: np.ndarray,
        verbose: int = 1,
    ) -> "keras.callbacks.History":
        import tensorflow as tf

        cfg = self.config
        epochs         = cfg.get("epochs",           100)
        batch_size     = cfg.get("batch_size",        32)
        patience       = cfg.get("patience",          10)
        lr_patience    = cfg.get("lr_patience",       5)
        lr_factor      = cfg.get("lr_factor",         0.5)
        min_lr         = cfg.get("min_lr",            1e-6)
        val_split      = cfg.get("validation_split",  0.2)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=verbose,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=verbose,
            ),
        ]

        history = self.model.fit(
            X_train_normal, X_train_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    # ------------------------------------------------------------------
    # Scoring — returns per-sample MSE reconstruction error
    # ------------------------------------------------------------------
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return raw per-sample MSE reconstruction error."""
        X_pred = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - X_pred), axis=1)

    def score(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Return anomaly scores normalised to [0, 1].

        Parameters
        ----------
        X          : scaled feature array
        fit_scaler : if True, fit a new MinMaxScaler on these scores
                     (use True during training on the training set)
        """
        errors = self.reconstruction_error(X)
        if fit_scaler or self.score_scaler is None:
            self.score_scaler = MinMaxScaler()
            scores = self.score_scaler.fit_transform(errors.reshape(-1, 1)).ravel()
        else:
            scores = self.score_scaler.transform(errors.reshape(-1, 1)).ravel()
        return scores.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_dir / "autoencoder.keras"))
        if self.score_scaler is not None:
            joblib.dump(self.score_scaler, model_dir / "ae_score_scaler.pkl")

    def load(self, model_dir: str | Path) -> None:
        import tensorflow as tf
        model_dir = Path(model_dir)
        # Prefer modern .keras format; fall back to legacy .h5
        keras_path = model_dir / "autoencoder.keras"
        h5_path    = model_dir / "autoencoder.h5"
        if keras_path.exists():
            self.model = tf.keras.models.load_model(str(keras_path))
        elif h5_path.exists():
            self.model = tf.keras.models.load_model(str(h5_path), compile=False)
        else:
            raise FileNotFoundError(f"No autoencoder model found in {model_dir}")
        scaler_path = model_dir / "ae_score_scaler.pkl"
        if scaler_path.exists():
            self.score_scaler = joblib.load(scaler_path)
