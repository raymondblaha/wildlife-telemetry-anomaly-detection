"""Autoencoder helpers for reconstruction-error anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Threshold:
    """Frozen anomaly threshold metadata."""

    value: float
    source: str
    percentile: float | None = None
    policy: str = "validation_percentile"
    metadata: dict[str, float | int | str] = field(default_factory=dict)


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


def scaled_mse_loss(loss_scale: float = 1.0):
    """Build the scaled-MSE loss used in the manuscript notebooks."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for scaled_mse_loss. "
            "Install with `pip install .[deep-learning]`."
        ) from exc

    def custom_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return loss_scale * mse

    return custom_loss


def reconstruct(model, x: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
    """Return reconstructed feature rows from a fitted autoencoder."""

    return model.predict(x, batch_size=batch_size, verbose=0)


def squared_residuals(x: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Return per-feature squared reconstruction residuals."""

    if x.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: x={x.shape}, reconstructed={reconstructed.shape}")
    return np.square(x - reconstructed)


def reconstruction_error(
    model,
    x: np.ndarray,
    *,
    batch_size: int = 128,
    loss_scale: float = 1.0,
) -> np.ndarray:
    """Return per-row scaled mean squared reconstruction error."""

    reconstructed = reconstruct(model, x, batch_size=batch_size)
    return reconstruction_error_from_residuals(
        squared_residuals(x, reconstructed),
        loss_scale=loss_scale,
    )


def reconstruction_error_from_residuals(
    residual_sq: np.ndarray,
    *,
    loss_scale: float = 1.0,
) -> np.ndarray:
    """Return per-row scaled mean squared error from squared residuals."""

    return loss_scale * np.mean(residual_sq, axis=1)


def choose_threshold(
    errors: np.ndarray,
    *,
    percentile: float = 95,
    source: str = "validation",
) -> Threshold:
    """Freeze a percentile-based threshold before final test evaluation."""

    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    return Threshold(
        value=float(np.percentile(errors, percentile)),
        source=source,
        percentile=percentile,
        policy=f"{source}_percentile",
    )


def choose_threshold_by_policy(
    *,
    train_errors: np.ndarray,
    val_errors: np.ndarray | None = None,
    val_ids: np.ndarray | None = None,
    policy: str = "validation_percentile",
    percentile: float = 95,
    lowest_frac: float = 0.75,
    lowest_quantile: float = 90,
    train_quantile: float = 0.90,
    val_row_fpr: float = 0.20,
    val_id_rate: float = 0.20,
    min_rows_per_id: int = 1,
    mad_k: float = 3.0,
) -> Threshold:
    """Choose a frozen threshold using manuscript-style policy options."""

    train_errors = np.asarray(train_errors, dtype=float)
    val_errors = None if val_errors is None else np.asarray(val_errors, dtype=float)
    metadata: dict[str, float | int | str] = {"policy": policy}

    if policy == "validation_percentile":
        if val_errors is None:
            raise ValueError("validation_percentile requires val_errors")
        value = float(np.percentile(val_errors, percentile))
        metadata.update({"percentile": float(percentile)})
        return Threshold(
            value=value,
            source="validation",
            percentile=percentile,
            policy=policy,
            metadata=metadata,
        )

    if policy == "train_lowest_p95":
        if not 0 < lowest_frac <= 1:
            raise ValueError("lowest_frac must be in (0, 1]")
        n_low = max(1, int(len(train_errors) * lowest_frac))
        base = np.sort(train_errors)[:n_low]
        value = float(np.percentile(base, lowest_quantile))
        metadata.update(
            {
                "lowest_frac": float(lowest_frac),
                "lowest_quantile": float(lowest_quantile),
                "n_low": int(n_low),
            }
        )
        return Threshold(
            value=value,
            source="train_lowest",
            percentile=lowest_quantile,
            policy=policy,
            metadata=metadata,
        )

    if policy == "train_quantile":
        value = float(np.quantile(train_errors, train_quantile))
        metadata.update({"train_quantile": float(train_quantile)})
        return Threshold(
            value=value,
            source="train",
            percentile=train_quantile * 100,
            policy=policy,
            metadata=metadata,
        )

    if policy == "val_row_fpr":
        if val_errors is None:
            raise ValueError("val_row_fpr requires val_errors")
        quantile = 1.0 - float(val_row_fpr)
        value = float(np.quantile(val_errors, quantile))
        metadata.update({"val_row_fpr": float(val_row_fpr), "val_quantile": float(quantile)})
        return Threshold(
            value=value,
            source="validation",
            percentile=quantile * 100,
            policy=policy,
            metadata=metadata,
        )

    if policy == "val_id_flag_rate":
        if val_errors is None or val_ids is None:
            raise ValueError("val_id_flag_rate requires val_errors and val_ids")
        import pandas as pd

        q_grid = np.linspace(0.50, 0.999, 200)
        ids = np.asarray(val_ids)
        best_value, best_rate, best_quantile = None, None, None
        n_rows = pd.Series(1, index=ids).groupby(level=0).sum()
        eligible = n_rows[n_rows >= min_rows_per_id].index
        for quantile in q_grid:
            candidate = float(np.quantile(val_errors, quantile))
            row_flag = val_errors > candidate
            id_any = pd.Series(row_flag, index=ids).groupby(level=0).any()
            rate = float(id_any.reindex(eligible).fillna(False).mean())
            if rate <= val_id_rate:
                best_value, best_rate, best_quantile = candidate, rate, float(quantile)
                break
        if best_value is None:
            best_quantile = float(q_grid[-1])
            best_value = float(np.quantile(val_errors, best_quantile))
            row_flag = val_errors > best_value
            id_any = pd.Series(row_flag, index=ids).groupby(level=0).any()
            best_rate = float(id_any.reindex(eligible).fillna(False).mean())
        metadata.update(
            {
                "val_id_rate_target": float(val_id_rate),
                "achieved_rate": float(best_rate),
                "val_quantile": float(best_quantile),
                "min_rows_per_id": int(min_rows_per_id),
            }
        )
        return Threshold(
            value=float(best_value),
            source="validation_id_rate",
            percentile=float(best_quantile) * 100,
            policy=policy,
            metadata=metadata,
        )

    if policy == "mad":
        median = float(np.median(train_errors))
        mad = float(np.median(np.abs(train_errors - median)) + 1e-12)
        value = median + mad_k * 1.4826 * mad
        metadata.update({"median": median, "mad": mad, "mad_k": float(mad_k)})
        return Threshold(
            value=value,
            source="train_mad",
            percentile=None,
            policy=policy,
            metadata=metadata,
        )

    raise ValueError(f"Unknown threshold policy: {policy}")


def flag_anomalies(errors: np.ndarray, threshold: Threshold | float) -> np.ndarray:
    """Convert reconstruction errors to binary anomaly flags."""

    value = threshold.value if isinstance(threshold, Threshold) else float(threshold)
    return errors > value
