"""Evaluate row-level and animal-level anomaly calls from saved predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.evaluation import animal_level_flags, row_level_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/example_run/test_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/example_run")
    parser.add_argument("--animal-col", default="animal_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--flag-col", default="is_anomaly")
    parser.add_argument("--min-anomalous-fixes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    required = {args.animal_col, args.label_col, args.flag_col}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")

    row_metrics = row_level_metrics(
        predictions[args.label_col].astype(int).to_numpy(),
        predictions[args.flag_col].astype(int).to_numpy(),
    )
    animal_flags = animal_level_flags(
        predictions,
        animal_col=args.animal_col,
        label_col=args.label_col,
        flag_col=args.flag_col,
        min_anomalous_fixes=args.min_anomalous_fixes,
    )
    animal_metrics = row_level_metrics(
        animal_flags["true_label"].astype(int).to_numpy(),
        animal_flags["predicted_label"].astype(int).to_numpy(),
    )

    pd.Series(row_metrics).to_csv(output_dir / "row_level_metrics.csv")
    pd.Series(animal_metrics).to_csv(output_dir / "animal_level_metrics.csv")
    animal_flags.to_csv(output_dir / "animal_level_flags.csv", index=False)

    print("Wrote row_level_metrics.csv, animal_level_metrics.csv, and animal_level_flags.csv")


if __name__ == "__main__":
    main()
