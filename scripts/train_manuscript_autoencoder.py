"""Train the manuscript-style dense autoencoder on processed feature files."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.autoencoder import choose_threshold_by_policy, scaled_mse_loss
from telemetry_anomaly.manuscript_profiles import get_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["wtd_primary", "wtd_four_pings", "mule_60_40"])
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold-policy", default="train_lowest_p95")
    parser.add_argument("--lowest-frac", type=float, default=0.75)
    parser.add_argument("--lowest-q", type=float, default=90)
    parser.add_argument("--lambda-dev", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def load_matrix(path: Path, feature_columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    matrix = df[feature_columns].values.astype(np.float32)
    return matrix, df


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    processed_dir = Path(args.processed_dir)
    suffix = profile.processed_suffix
    batch_size = args.batch_size or profile.default_batch_size
    learning_rate = args.learning_rate or profile.default_learning_rate

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    try:
        import tensorflow as tf
        from tensorflow.keras import Model, layers, regularizers
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError as exc:
        raise ImportError("Install TensorFlow with `pip install .[deep-learning]`.") from exc

    tf.keras.utils.set_random_seed(args.random_state)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    feature_columns = joblib.load(processed_dir / f"feature_columns_{suffix}.pkl")
    x_train, train_df = load_matrix(
        processed_dir / f"train_processed_{suffix}.csv",
        feature_columns,
    )
    x_val, val_df = load_matrix(processed_dir / f"val_processed_{suffix}.csv", feature_columns)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    train_ds = train_ds.shuffle(
        min(10000, len(x_train)),
        seed=args.random_state,
        reshuffle_each_iteration=True,
    )
    train_ds = train_ds.batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, x_val))
    val_ds = val_ds.batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)

    input_dim = x_train.shape[1]
    regularizer = regularizers.l2(1e-4)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(inputs)
    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizer)(x)
    z = layers.Dense(4, activation="relu", kernel_regularizer=regularizer, name="bottleneck")(x)
    y = layers.Dense(16, activation="relu", kernel_regularizer=regularizer)(z)
    y = layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(y)
    outputs = layers.Dense(input_dim, activation="linear")(y)
    autoencoder = Model(inputs, outputs, name="autoencoder")

    loss_scale = 1.0 + args.lambda_dev * input_dim
    loss_fn = scaled_mse_loss(loss_scale)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss_fn)

    ckpt_path = processed_dir / "ae_ckpt.weights.h5"
    history = autoencoder.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True,
                monitor="val_loss",
                save_best_only=True,
            ),
        ],
        verbose=1,
    )

    autoencoder.save(processed_dir / "autoencoder_model.h5")
    np.save(processed_dir / "training_history.npy", history.history, allow_pickle=True)

    recon_train = autoencoder.predict(x_train, batch_size=batch_size, verbose=0)
    train_errors = loss_fn(x_train, recon_train).numpy().astype(np.float64)
    recon_val = autoencoder.predict(x_val, batch_size=batch_size, verbose=0)
    val_errors = loss_fn(x_val, recon_val).numpy().astype(np.float64)

    threshold = choose_threshold_by_policy(
        train_errors=train_errors,
        val_errors=val_errors,
        val_ids=val_df["id"].values,
        policy=args.threshold_policy,
        lowest_frac=args.lowest_frac,
        lowest_quantile=args.lowest_q,
    )
    val_row_flag = val_errors > threshold.value
    val_id_any = pd.Series(val_row_flag).groupby(val_df["id"].values, sort=False).any()
    val_n_rows = pd.Series(1, index=val_df["id"].values).groupby(level=0).sum()
    val_eligible = val_n_rows[val_n_rows >= 1].index
    val_id_rate = float(val_id_any.reindex(val_eligible).fillna(False).mean())

    threshold_info = {
        "policy": args.threshold_policy,
        "theta": float(threshold.value),
        "lowest_frac": float(args.lowest_frac),
        "percentile": float(args.lowest_q),
        "lambda_dev": float(args.lambda_dev),
        "input_dim": int(input_dim),
        "loss_scale": float(loss_scale),
        "loss_name": "scaled_mse_equivalent_to_mse_plus_lambda_sse",
        "train_rows_flagged": float(np.mean(train_errors > threshold.value)),
        "val_rows_flagged": float(np.mean(val_errors > threshold.value)),
        "val_ids_flagged": val_id_rate,
        "min_rows_per_id": 1,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
    }
    joblib.dump(threshold_info, processed_dir / "reconstruction_threshold.pkl")
    print(f"Saved model, history, checkpoint, and threshold to {processed_dir}")
    print(f"Theta={threshold.value:.6g}; policy={args.threshold_policy}")


if __name__ == "__main__":
    main()
