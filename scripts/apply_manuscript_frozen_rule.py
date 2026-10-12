"""Apply frozen manuscript AE threshold and ID aggregation rule to holdout rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.manuscript_profiles import get_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["wtd_primary", "wtd_four_pings", "mule_60_40"])
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument(
        "--aggregation-rule",
        choices=["OR", "AND", "FRAC_ONLY", "COUNT_ONLY"],
        default=None,
    )
    parser.add_argument("--min-rows-per-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    processed_dir = Path(args.processed_dir)
    suffix = profile.processed_suffix
    count = args.count if args.count is not None else profile.default_count_rule
    frac = args.frac if args.frac is not None else profile.default_fraction_rule
    aggregation_rule = args.aggregation_rule or profile.default_aggregation_rule

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ImportError("Install TensorFlow with `pip install .[deep-learning]`.") from exc

    feature_columns = joblib.load(processed_dir / f"feature_columns_{suffix}.pkl")
    threshold_info = joblib.load(processed_dir / "reconstruction_threshold.pkl")
    theta = float(threshold_info["theta"])
    lambda_dev = float(threshold_info.get("lambda_dev", 0.5))
    input_dim = int(threshold_info.get("input_dim", len(feature_columns)))
    loss_scale = float(threshold_info.get("loss_scale", 1.0 + lambda_dev * input_dim))

    def custom_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return loss_scale * mse

    model = load_model(processed_dir / "autoencoder_model.h5", compile=False)
    test_path = processed_dir / f"test_processed_{suffix}.csv"
    test = pd.read_csv(test_path)
    if profile.datetime_col in test.columns:
        test[profile.datetime_col] = pd.to_datetime(test[profile.datetime_col], errors="coerce")

    x_test = test[feature_columns].values.astype(np.float32)
    recon = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    row_error = custom_loss(x_test, recon).numpy().astype(np.float64)
    row_flag = row_error > theta
    residuals = np.abs(x_test - recon)

    rows = pd.DataFrame(
        {
            "id": test["id"].values,
            profile.datetime_col: test[profile.datetime_col]
            if profile.datetime_col in test.columns
            else pd.NaT,
            "reconstruction_error": row_error,
            "row_flag": row_flag,
        }
    )
    rows.to_csv(
        processed_dir / f"test_processed_{suffix}_rows_with_flags_FROZEN.csv",
        index=False,
    )

    residual_frame = pd.concat(
        [
            test[
                [col for col in profile.metadata_features if col in test.columns]
            ].reset_index(drop=True),
            pd.DataFrame(recon, columns=[f"{col}_recon" for col in feature_columns]),
            pd.DataFrame(residuals, columns=[f"{col}_resid" for col in feature_columns]),
        ],
        axis=1,
    )
    residual_frame.to_csv(
        processed_dir / f"test_processed_{suffix}_rows_with_recon_and_residuals.csv",
        index=False,
    )

    grouped = pd.DataFrame(
        {"id": test["id"].values, "row_flag": row_flag, "error": row_error}
    ).groupby("id", sort=False)
    n_rows = grouped.size()
    n_flags = grouped["row_flag"].sum()
    flag_frac = n_flags / n_rows
    id_max = grouped["error"].max()
    id_mean = grouped["error"].mean()

    eligible = n_rows[n_rows >= args.min_rows_per_id].index
    n_flags_eligible = n_flags.reindex(eligible)
    flag_frac_eligible = flag_frac.reindex(eligible)
    count_ids = set(n_flags_eligible[n_flags_eligible >= count].index)
    frac_ids = set(flag_frac_eligible[flag_frac_eligible >= frac].index)

    if aggregation_rule == "AND":
        pred_ids = count_ids & frac_ids
    elif aggregation_rule == "FRAC_ONLY":
        pred_ids = frac_ids
    elif aggregation_rule == "COUNT_ONLY":
        pred_ids = count_ids
    else:
        pred_ids = count_ids | frac_ids

    count_norm = n_flags / max(count, 1e-12)
    frac_norm = flag_frac / max(frac, 1e-12)
    if aggregation_rule == "AND":
        rule_score = np.minimum(count_norm, frac_norm)
    elif aggregation_rule == "FRAC_ONLY":
        rule_score = frac_norm
    elif aggregation_rule == "COUNT_ONLY":
        rule_score = count_norm
    else:
        rule_score = np.maximum(count_norm, frac_norm)

    id_summary = pd.DataFrame(
        {
            "id": n_rows.index,
            "n_rows": n_rows.values,
            "n_flags": n_flags.reindex(n_rows.index).values,
            "flag_frac": flag_frac.reindex(n_rows.index).values,
            "rule_aligned_score": rule_score.reindex(n_rows.index).values,
            "frac_norm_score": frac_norm.reindex(n_rows.index).values,
            "count_norm_score": count_norm.reindex(n_rows.index).values,
            "id_max_error": id_max.reindex(n_rows.index).values,
            "id_mean_error": id_mean.reindex(n_rows.index).values,
            "eligible": [1 if animal_id in set(eligible) else 0 for animal_id in n_rows.index],
            "is_positive_id": [1 if animal_id in pred_ids else 0 for animal_id in n_rows.index],
        }
    )
    id_summary.to_csv(
        processed_dir / f"test_processed_{suffix}_id_summary_FROZEN.csv",
        index=False,
    )

    frozen = {
        "theta": theta,
        "count": int(count),
        "frac": float(frac),
        "lambda_dev": lambda_dev,
        "input_dim": input_dim,
        "loss_scale": loss_scale,
        "policy": threshold_info.get("policy", "unknown"),
        "aggregation_rule": aggregation_rule,
        "val_meta": {
            "min_rows_per_id": int(args.min_rows_per_id),
            "use_mixed_val": False,
        },
    }
    joblib.dump(frozen, processed_dir / "id_rule_params.pkl")
    print("Predicted positive IDs:", sorted(pred_ids))
    print(f"Saved frozen row flags, residuals, ID summary, and params to {processed_dir}")


if __name__ == "__main__":
    main()
