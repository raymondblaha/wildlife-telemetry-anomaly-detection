"""Generate manuscript-shaped synthetic CSV inputs for local artifact demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.manuscript_profiles import get_profile


SEASONS = ["winter", "spring", "summer", "fall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["wtd_primary", "wtd_four_pings", "mule_60_40"])
    parser.add_argument("--output-dir", default="data/sample/manuscript_synthetic")
    parser.add_argument("--train-animals", type=int, default=16)
    parser.add_argument("--test-animals", type=int, default=8)
    parser.add_argument("--rows-per-animal", type=int, default=180)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def seasonal_value(step: int) -> str:
    return SEASONS[(step // 45) % len(SEASONS)]


def numeric_feature_value(
    feature: str,
    rng: np.random.Generator,
    *,
    comparison: bool,
) -> float:
    shift = 1.0 if comparison else 0.0
    if feature in {"velo", "rolling_velo_mean"}:
        return float(rng.gamma(2.2 + shift, 0.35))
    if feature in {"rolling_velo_sd", "rolling_abs_rel_angle_sd"}:
        return float(abs(rng.normal(0.35 + 0.12 * shift, 0.10)))
    if "skew" in feature:
        return float(rng.normal(0.15 * shift, 0.35))
    if "kurt" in feature:
        return float(rng.normal(2.8 + 0.25 * shift, 0.45))
    if feature in {"abs_rel_angle", "rolling_abs_rel_angle_mean"}:
        return float(abs(rng.normal(0.75 + 0.20 * shift, 0.35)))
    if feature in {"sinuosity", "sinuosity_28pt"}:
        return float(abs(rng.normal(1.25 + 0.18 * shift, 0.18)))
    if feature in {"akde_low", "low"}:
        return float(abs(rng.normal(1.8 + 0.4 * shift, 0.3)))
    if feature in {"akde_est", "est"}:
        return float(abs(rng.normal(2.5 + 0.6 * shift, 0.4)))
    if feature in {"akde_high", "high"}:
        return float(abs(rng.normal(3.2 + 0.8 * shift, 0.5)))
    return float(rng.normal(0.0, 1.0))


def build_rows(
    profile_name: str,
    *,
    n_animals: int,
    rows_per_animal: int,
    comparison: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    profile = get_profile(profile_name)
    rows = []
    start = pd.Timestamp("2024-01-01")
    prefix = "cmp" if comparison else "ref"

    for animal_idx in range(n_animals):
        animal_id = f"{profile_name}_{prefix}_{animal_idx:03d}"
        for step in range(rows_per_animal):
            timestamp = start + pd.Timedelta(hours=6 * step)
            row = {
                "id": animal_id,
                profile.datetime_col: timestamp.isoformat(),
                profile.label_col: "positive" if comparison else profile.reference_label,
                "season": seasonal_value(step),
            }
            if "pkey" in profile.metadata_features:
                row["pkey"] = f"{animal_id}_{step:05d}"

            for feature in profile.numeric_features:
                row[feature] = numeric_feature_value(feature, rng, comparison=comparison)

            for drop_col in profile.drop_columns:
                if drop_col not in row:
                    row[drop_col] = rng.normal()

            if "date_std" in profile.drop_columns:
                row["date_std"] = timestamp.date().isoformat()
            if "burst" in profile.drop_columns:
                row["burst"] = animal_idx
            if "seg_key" in profile.drop_columns:
                row["seg_key"] = f"{animal_id}_segment"
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    train = build_rows(
        args.profile,
        n_animals=args.train_animals,
        rows_per_animal=args.rows_per_animal,
        comparison=False,
        rng=rng,
    )
    test = build_rows(
        args.profile,
        n_animals=args.test_animals,
        rows_per_animal=args.rows_per_animal,
        comparison=True,
        rng=rng,
    )

    train_path = output_dir / f"{args.profile}_train.csv"
    test_path = output_dir / f"{args.profile}_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
