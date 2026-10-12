"""Telemetry cleaning, feature engineering, and leakage-safe splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


REQUIRED_COLUMNS = {
    "animal_id",
    "timestamp",
    "latitude",
    "longitude",
}


@dataclass(frozen=True)
class FeatureBundle:
    """Frozen preprocessing objects and matrix outputs."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    numeric_features: list[str]
    categorical_features: list[str]
    scaler: MinMaxScaler
    encoder: OneHotEncoder | None


def validate_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise a clear error when expected telemetry fields are missing."""

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required telemetry columns: {missing}")


def add_movement_features(
    df: pd.DataFrame,
    *,
    id_col: str = "animal_id",
    time_col: str = "timestamp",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    rolling_window: int = 3,
) -> pd.DataFrame:
    """Add step distance, speed, turning angle, and rolling movement summaries."""

    validate_schema(
        df.rename(
            columns={
                id_col: "animal_id",
                time_col: "timestamp",
                lat_col: "latitude",
                lon_col: "longitude",
            }
        )
    )
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.sort_values([id_col, time_col]).reset_index(drop=True)

    previous = out.groupby(id_col)[[lat_col, lon_col, time_col]].shift(1)
    out["step_distance_m"] = haversine_m(
        previous[lat_col], previous[lon_col], out[lat_col], out[lon_col]
    )
    elapsed_hours = (out[time_col] - previous[time_col]).dt.total_seconds() / 3600
    out["elapsed_hours"] = elapsed_hours
    out["speed_m_per_hour"] = out["step_distance_m"] / elapsed_hours

    out["bearing_deg"] = bearing_deg(previous[lat_col], previous[lon_col], out[lat_col], out[lon_col])
    out["turning_angle_deg"] = (
        out.groupby(id_col)["bearing_deg"].diff().pipe(lambda s: ((s + 180) % 360) - 180).abs()
    )

    grouped = out.groupby(id_col, group_keys=False)
    out[f"rolling_{rolling_window}_step_mean_m"] = grouped["step_distance_m"].transform(
        lambda s: s.rolling(rolling_window, min_periods=1).mean()
    )
    out[f"rolling_{rolling_window}_speed_mean"] = grouped["speed_m_per_hour"].transform(
        lambda s: s.rolling(rolling_window, min_periods=1).mean()
    )
    out["hour"] = out[time_col].dt.hour
    out["month"] = out[time_col].dt.month
    return out


def animal_group_split(
    df: pd.DataFrame,
    *,
    group_col: str = "animal_id",
    test_size: float = 0.2,
    random_state: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows while keeping each animal entirely in one partition."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def fit_feature_bundle(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> FeatureBundle:
    """Fit preprocessing on training data and apply it unchanged to val/test."""

    categorical_features = categorical_features or []
    scaler = MinMaxScaler()
    x_train_num = scaler.fit_transform(train[numeric_features])
    x_val_num = scaler.transform(val[numeric_features])
    x_test_num = scaler.transform(test[numeric_features])

    encoder = None
    if categorical_features:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x_train_cat = encoder.fit_transform(train[categorical_features])
        x_val_cat = encoder.transform(val[categorical_features])
        x_test_cat = encoder.transform(test[categorical_features])
        x_train = np.hstack([x_train_num, x_train_cat])
        x_val = np.hstack([x_val_num, x_val_cat])
        x_test = np.hstack([x_test_num, x_test_cat])
    else:
        x_train, x_val, x_test = x_train_num, x_val_num, x_test_num

    return FeatureBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaler=scaler,
        encoder=encoder,
    )


def haversine_m(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in meters."""

    radius_m = 6_371_000
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return pd.Series(radius_m * 2 * np.arcsin(np.sqrt(a)), index=lat2.index)


def bearing_deg(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Initial bearing between consecutive fixes in degrees."""

    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return pd.Series((np.degrees(np.arctan2(x, y)) + 360) % 360, index=lat2.index)
