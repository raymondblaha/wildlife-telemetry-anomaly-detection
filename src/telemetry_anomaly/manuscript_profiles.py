"""Column profiles from the manuscript notebooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManuscriptProfile:
    """Feature and metadata columns for a manuscript experiment family."""

    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    metadata_features: list[str]
    drop_columns: list[str]
    datetime_col: str
    label_col: str
    reference_label: str
    processed_suffix: str
    default_learning_rate: float
    default_batch_size: int
    default_count_rule: int
    default_fraction_rule: float = 0.75
    default_aggregation_rule: str = "OR"


WTD_NUMERIC_FEATURES = [
    "velo",
    "abs_rel_angle",
    "sinuosity",
    "rolling_abs_rel_angle_mean",
    "rolling_abs_rel_angle_sd",
    "rolling_abs_rel_angle_skew",
    "rolling_abs_rel_angle_kurt",
    "rolling_velo_mean",
    "rolling_velo_sd",
    "rolling_velo_skew",
    "rolling_velo_kurt",
    "akde_low",
    "akde_est",
    "akde_high",
]

MULE_NUMERIC_FEATURES = [
    "velo",
    "abs_rel_angle",
    "sinuosity_28pt",
    "rolling_abs_rel_angle_mean",
    "rolling_abs_rel_angle_sd",
    "rolling_abs_rel_angle_skew",
    "rolling_abs_rel_angle_kurt",
    "rolling_velo_mean",
    "rolling_velo_sd",
    "rolling_velo_skew",
    "rolling_velo_kurt",
    "low",
    "est",
    "high",
]

WTD_PRIMARY = ManuscriptProfile(
    name="wtd_primary",
    numeric_features=WTD_NUMERIC_FEATURES,
    categorical_features=["season"],
    metadata_features=["id", "date"],
    drop_columns=["date_std", "dx", "dy", "dt", "burst", "seg_key"],
    datetime_col="date",
    label_col="cwd_status",
    reference_label="negative",
    processed_suffix="100",
    default_learning_rate=1e-6,
    default_batch_size=128,
    default_count_rule=550,
)

WTD_FOUR_PINGS = ManuscriptProfile(
    name="wtd_four_pings",
    numeric_features=WTD_NUMERIC_FEATURES,
    categorical_features=["season"],
    metadata_features=["id", "date"],
    drop_columns=["date_std", "dx", "dy", "dt", "burst", "seg_key"],
    datetime_col="date",
    label_col="cwd_status",
    reference_label="negative",
    processed_suffix="100",
    default_learning_rate=1e-4,
    default_batch_size=180,
    default_count_rule=110,
)

MULE_60_40 = ManuscriptProfile(
    name="mule_60_40",
    numeric_features=MULE_NUMERIC_FEATURES,
    categorical_features=["season"],
    metadata_features=["pkey", "id", "datetime"],
    drop_columns=[
        "x_2",
        "x_1",
        "x_2_2",
        "transition_to_migratory",
        "id_1",
        "phases",
        "migration_class_grouped",
        "hour",
        "movement_group",
        "r2n_smoothed",
        "suspect_jump",
        "extreme_jump",
        "SATS",
        "rel.angle",
        "utm_easting",
        "utm_northing",
        "dx",
        "dy",
        "dt",
        "cum_dist",
        "dist_per_hr",
    ],
    datetime_col="datetime",
    label_col="CWD_status",
    reference_label="negative",
    processed_suffix="60_40",
    default_learning_rate=1e-4,
    default_batch_size=128,
    default_count_rule=75,
)

PROFILES = {
    WTD_PRIMARY.name: WTD_PRIMARY,
    WTD_FOUR_PINGS.name: WTD_FOUR_PINGS,
    MULE_60_40.name: MULE_60_40,
}


def get_profile(name: str) -> ManuscriptProfile:
    """Return a named manuscript profile."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown manuscript profile '{name}'. Valid profiles: {valid}") from exc
