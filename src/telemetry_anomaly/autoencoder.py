"""Autoencoder helpers for reconstruction-error anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Threshold:
    """Frozen anomaly threshold metadata."""

    value: float
    source: str
    percentile: float


def build_dense_autoencoder(input_dim: int, latent_dim: int = 8):
    """Build a compact TensorFlow/Keras dense autoencoder."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for build_dense_autoencoder. "
            "Install with `pip install .[deep-learning]`."
        ) from exc

    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(64, activation="relu")(inputs)
    encoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
    encoded = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(encoded)
    decoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(64, activation="relu")(decoded)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(decoded)

    model = tf.keras.Model(inputs, outputs, name="telemetry_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def reconstruction_error(model, x: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
    """Return per-row mean squared reconstruction error."""

    reconstructed = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean(np.square(x - reconstructed), axis=1)


def choose_threshold(errors: np.ndarray, *, percentile: float = 95, source: str = "validation") -> Threshold:
    """Freeze a percentile-based threshold before final test evaluation."""

    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    return Threshold(value=float(np.percentile(errors, percentile)), source=source, percentile=percentile)


def flag_anomalies(errors: np.ndarray, threshold: Threshold | float) -> np.ndarray:
    """Convert reconstruction errors to binary anomaly flags."""

    value = threshold.value if isinstance(threshold, Threshold) else float(threshold)
    return errors > value
