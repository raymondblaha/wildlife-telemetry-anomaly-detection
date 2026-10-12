"""Config-driven training entrypoint for the public telemetry anomaly pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.autoencoder import (
    build_dense_autoencoder,
    choose_threshold_by_policy,
    reconstruct,
    reconstruction_error_from_residuals,
    scaled_mse_loss,
    squared_residuals,
)
from telemetry_anomaly.evaluation import animal_level_flags, row_level_metrics
from telemetry_anomaly.feature_contributions import feature_contribution_table
from telemetry_anomaly.preprocessing import (
    add_movement_features,
    animal_group_split,
    fit_feature_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/example_config.yml")
    return parser.parse_args()


def bundle_feature_names(bundle) -> list[str]:
    """Return the frozen model feature order from fitted preprocessing artifacts."""

    names = list(bundle.numeric_features)
    if bundle.encoder is not None and bundle.categorical_features:
        names.extend(bundle.encoder.get_feature_names_out(bundle.categorical_features).tolist())
    return names


def processed_frame(
    source: pd.DataFrame,
    matrix,
    feature_names: list[str],
    config: dict,
) -> pd.DataFrame:
    """Combine metadata columns with transformed features."""

    meta_cols = [
        config["animal_col"],
        config["label_col"],
        *config.get("metadata_columns", []),
    ]
    meta_cols = [col for col in dict.fromkeys(meta_cols) if col in source.columns]
    return pd.concat(
        [
            source[meta_cols].reset_index(drop=True),
            pd.DataFrame(matrix, columns=feature_names),
        ],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config["input_csv"])
    df = add_movement_features(df)
    df = df.dropna(subset=config["numeric_features"] + [config["label_col"]]).copy()

    train_pool = df[df[config["label_col"]] == config["reference_label"]].copy()
    train, val = animal_group_split(
        train_pool,
        group_col=config["animal_col"],
        test_size=config["validation_size"],
        random_state=config["random_state"],
    )
    test = df[~df[config["animal_col"]].isin(train[config["animal_col"]])].copy()

    bundle = fit_feature_bundle(
        train,
        val,
        test,
        numeric_features=config["numeric_features"],
        categorical_features=config.get("categorical_features", []),
    )
    feature_names = bundle_feature_names(bundle)

    model = build_dense_autoencoder(bundle.x_train.shape[1], latent_dim=config["latent_dim"])
    loss_lambda = float(config.get("loss_lambda", 0.5))
    loss_scale = float(config.get("loss_scale", 1.0 + loss_lambda * bundle.x_train.shape[1]))
    model.compile(
        optimizer=config.get("optimizer", "adam"),
        loss=scaled_mse_loss(loss_scale),
    )
    model.fit(
        bundle.x_train,
        bundle.x_train,
        validation_data=(bundle.x_val, bundle.x_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=1,
    )

    train_recon = reconstruct(model, bundle.x_train, batch_size=config["batch_size"])
    val_recon = reconstruct(model, bundle.x_val, batch_size=config["batch_size"])
    test_recon = reconstruct(model, bundle.x_test, batch_size=config["batch_size"])
    train_residual_sq = squared_residuals(bundle.x_train, train_recon)
    val_residual_sq = squared_residuals(bundle.x_val, val_recon)
    test_residual_sq = squared_residuals(bundle.x_test, test_recon)
    train_errors = reconstruction_error_from_residuals(train_residual_sq, loss_scale=loss_scale)
    val_errors = reconstruction_error_from_residuals(val_residual_sq, loss_scale=loss_scale)
    test_errors = reconstruction_error_from_residuals(test_residual_sq, loss_scale=loss_scale)
    threshold = choose_threshold_by_policy(
        train_errors=train_errors,
        val_errors=val_errors,
        val_ids=val[config["animal_col"]].to_numpy(),
        policy=config.get("threshold_policy", "validation_percentile"),
        percentile=config["threshold_percentile"],
        lowest_frac=config.get("lowest_frac", 0.75),
        lowest_quantile=config.get("lowest_quantile", 90),
        train_quantile=config.get("train_quantile", 0.90),
        val_row_fpr=config.get("val_row_fpr", 0.20),
        val_id_rate=config.get("val_id_rate", 0.20),
        min_rows_per_id=config.get("min_rows_per_id", 1),
        mad_k=config.get("mad_k", 3.0),
    )
    test["reconstruction_error"] = test_errors
    test["is_anomaly"] = test_errors > threshold.value

    row_metrics = row_level_metrics(
        test[config["label_col"]].to_numpy(),
        test["is_anomaly"].astype(int),
    )
    animal_flags = animal_level_flags(
        test,
        animal_col=config["animal_col"],
        label_col=config["label_col"],
        flag_col="is_anomaly",
        min_anomalous_fixes=config["min_anomalous_fixes"],
    )

    model.save(output_dir / "autoencoder.keras")
    joblib.dump(feature_names, output_dir / "feature_columns.joblib")
    joblib.dump(bundle.scaler, output_dir / "scaler.joblib")
    joblib.dump(bundle.encoder, output_dir / "encoder.joblib")
    joblib.dump(threshold, output_dir / "threshold.joblib")
    (output_dir / "threshold.json").write_text(
        json.dumps(
            {
                "value": threshold.value,
                "source": threshold.source,
                "percentile": threshold.percentile,
                "policy": threshold.policy,
                "metadata": threshold.metadata,
                "loss_lambda": loss_lambda,
                "loss_scale": loss_scale,
                "input_dim": bundle.x_train.shape[1],
            },
            indent=2,
        )
    )
    processed_frame(train, bundle.x_train, feature_names, config).to_csv(
        output_dir / "train_processed.csv",
        index=False,
    )
    processed_frame(val, bundle.x_val, feature_names, config).to_csv(
        output_dir / "val_processed.csv",
        index=False,
    )
    processed_frame(test, bundle.x_test, feature_names, config).to_csv(
        output_dir / "test_processed.csv",
        index=False,
    )
    pd.DataFrame(test_residual_sq, columns=feature_names).to_csv(
        output_dir / "test_feature_residuals.csv",
        index=False,
    )
    contribution_masks = {
        "all_rows": np.ones(len(test), dtype=bool),
        "flagged_rows": test["is_anomaly"].to_numpy(bool),
    }
    label_col = config["label_col"]
    if label_col in test.columns:
        contribution_masks["comparison_rows"] = (
            test[label_col].ne(config["reference_label"]).to_numpy(bool)
        )
    feature_contribution_table(
        test_residual_sq,
        feature_names,
        contribution_masks,
    ).to_csv(output_dir / "feature_contributions.csv", index=False)
    pd.Series(row_metrics).to_csv(output_dir / "row_level_metrics.csv")
    animal_flags.to_csv(output_dir / "animal_level_flags.csv", index=False)
    test.to_csv(output_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
