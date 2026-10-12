"""Create manuscript-style feature contribution tables from residual outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.feature_contributions import (
    concentration_summary,
    feature_contribution_table,
    per_id_contribution_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/example_run/test_predictions.csv")
    parser.add_argument("--residuals", default="outputs/example_run/test_feature_residuals.csv")
    parser.add_argument("--features", default="outputs/example_run/feature_columns.joblib")
    parser.add_argument("--output-dir", default="outputs/example_run")
    parser.add_argument("--animal-col", default="animal_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--flag-col", default="is_anomaly")
    parser.add_argument("--reference-label", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    residuals = pd.read_csv(args.residuals)
    feature_names = joblib.load(args.features)
    residual_sq = residuals[feature_names].to_numpy(dtype=float)

    required = {args.animal_col, args.label_col, args.flag_col}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")

    masks = {
        "all_rows": np.ones(len(predictions), dtype=bool),
        "flagged_rows": predictions[args.flag_col].astype(bool).to_numpy(),
        "comparison_rows": predictions[args.label_col].ne(args.reference_label).to_numpy(),
    }
    contribution = feature_contribution_table(residual_sq, feature_names, masks)
    contribution = contribution.sort_values("flagged_rows_pct", ascending=False)
    contribution.to_csv(output_dir / "feature_contributions.csv", index=False)

    per_id = per_id_contribution_summary(
        residual_sq,
        feature_names,
        predictions[args.animal_col].to_numpy(),
        masks["flagged_rows"],
    )
    per_id.to_csv(output_dir / "feature_contributions_by_animal.csv", index=False)

    summary = pd.DataFrame(
        [
            {"subset": col.replace("_pct", ""), **concentration_summary(contribution, col)}
            for col in contribution.columns
            if col.endswith("_pct")
        ]
    )
    summary.to_csv(output_dir / "feature_contribution_summary.csv", index=False)
    print("Wrote feature contribution tables")


if __name__ == "__main__":
    main()
