"""Prepare manuscript autoencoder feature files with train-only preprocessing."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_anomaly.manuscript_profiles import get_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["wtd_primary", "wtd_four_pings", "mule_60_40"])
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def basic_cleanup(df: pd.DataFrame, profile) -> pd.DataFrame:
    """Match the manuscript notebook cleanup before scaling/encoding."""

    out = df.drop(columns=profile.drop_columns, errors="ignore").copy()
    out[f"{profile.datetime_col}_raw"] = out[profile.datetime_col]
    out[profile.datetime_col] = pd.to_datetime(out[profile.datetime_col], errors="coerce")

    for col in profile.numeric_features:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=profile.numeric_features + profile.categorical_features).copy()
    sort_cols = ["id", profile.datetime_col]
    if "pkey" in out.columns:
        sort_cols.append("pkey")
    return out.sort_values(sort_cols).reset_index(drop=True)


def transform_with_artifacts(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: MinMaxScaler,
    feature_columns: list[str],
    profile,
) -> pd.DataFrame:
    """Apply fitted encoder/scaler and preserve manuscript metadata columns."""

    cat_mat = encoder.transform(df[profile.categorical_features])
    num_mat = scaler.transform(df[profile.numeric_features])
    cat_names = encoder.get_feature_names_out(profile.categorical_features).tolist()
    matrix = np.hstack([num_mat, cat_mat])
    features = pd.DataFrame(matrix, columns=profile.numeric_features + cat_names)
    features = features.reindex(columns=feature_columns, fill_value=0.0)

    meta_cols = [col for col in profile.metadata_features if col in df.columns]
    return pd.concat(
        [df[meta_cols].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    train_pool = pd.read_csv(args.train_csv)
    train_pool = train_pool[train_pool[profile.label_col] == profile.reference_label].copy()
    train_pool = basic_cleanup(train_pool, profile)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_size,
        random_state=args.random_state,
    )
    train_idx, val_idx = next(splitter.split(train_pool, groups=train_pool["id"]))
    train_ids = train_pool.iloc[train_idx]["id"].unique()
    val_ids = train_pool.iloc[val_idx]["id"].unique()
    train = train_pool[train_pool["id"].isin(train_ids)].copy()
    val = train_pool[train_pool["id"].isin(val_ids)].copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = MinMaxScaler()
    encoder.fit(train[profile.categorical_features])
    scaler.fit(train[profile.numeric_features])
    feature_columns = profile.numeric_features + encoder.get_feature_names_out(
        profile.categorical_features
    ).tolist()

    test = basic_cleanup(pd.read_csv(args.test_csv), profile)
    train_out = transform_with_artifacts(train, encoder, scaler, feature_columns, profile)
    val_out = transform_with_artifacts(val, encoder, scaler, feature_columns, profile)
    test_out = transform_with_artifacts(test, encoder, scaler, feature_columns, profile)

    suffix = profile.processed_suffix
    train_out.to_csv(output_dir / f"train_processed_{suffix}.csv", index=False)
    val_out.to_csv(output_dir / f"val_processed_{suffix}.csv", index=False)
    test_out.to_csv(output_dir / f"test_processed_{suffix}.csv", index=False)
    joblib.dump(encoder, output_dir / f"encoder_{suffix}.pkl")
    joblib.dump(scaler, output_dir / f"scaler_{suffix}.pkl")
    joblib.dump(feature_columns, output_dir / f"feature_columns_{suffix}.pkl")
    joblib.dump(
        {
            "train_ids": sorted({str(x).strip() for x in train_ids}),
            "val_ids": sorted({str(x).strip() for x in val_ids}),
            "test_ids": sorted({str(x).strip() for x in test_out["id"].unique()}),
        },
        output_dir / f"split_ids_{suffix}.pkl",
    )

    print(f"Prepared manuscript features in {output_dir}")
    print("Train IDs:", sorted({str(x).strip() for x in train_ids}))
    print("Val IDs:", sorted({str(x).strip() for x in val_ids}))
    print("Test IDs:", sorted({str(x).strip() for x in test_out["id"].unique()}))


if __name__ == "__main__":
    main()
