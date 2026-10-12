"""Config-driven training entrypoint for the public telemetry anomaly pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.autoencoder import choose_threshold, reconstruction_error
from telemetry_anomaly.autoencoder import build_dense_autoencoder
from telemetry_anomaly.evaluation import animal_level_flags, row_level_metrics
from telemetry_anomaly.preprocessing import add_movement_features, animal_group_split, fit_feature_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/example_config.yml")
    return parser.parse_args()


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
    model = build_dense_autoencoder(bundle.x_train.shape[1], latent_dim=config["latent_dim"])
    model.fit(
        bundle.x_train,
        bundle.x_train,
        validation_data=(bundle.x_val, bundle.x_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=1,
    )

    val_errors = reconstruction_error(model, bundle.x_val, batch_size=config["batch_size"])
    threshold = choose_threshold(
        val_errors,
        percentile=config["threshold_percentile"],
        source="validation",
    )
    test_errors = reconstruction_error(model, bundle.x_test, batch_size=config["batch_size"])
    test["reconstruction_error"] = test_errors
    test["is_anomaly"] = test_errors > threshold.value

    row_metrics = row_level_metrics(test[config["label_col"]].to_numpy(), test["is_anomaly"].astype(int))
    animal_flags = animal_level_flags(
        test,
        animal_col=config["animal_col"],
        label_col=config["label_col"],
        flag_col="is_anomaly",
        min_anomalous_fixes=config["min_anomalous_fixes"],
    )

    model.save(output_dir / "autoencoder.keras")
    joblib.dump(bundle.scaler, output_dir / "scaler.joblib")
    joblib.dump(bundle.encoder, output_dir / "encoder.joblib")
    joblib.dump(threshold, output_dir / "threshold.joblib")
    pd.Series(row_metrics).to_csv(output_dir / "row_level_metrics.csv")
    animal_flags.to_csv(output_dir / "animal_level_flags.csv", index=False)
    test.to_csv(output_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
