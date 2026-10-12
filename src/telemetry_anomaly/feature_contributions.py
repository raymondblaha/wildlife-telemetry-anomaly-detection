"""Feature-level reconstruction residual summaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def contribution_percent(residual_sq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return each feature's share of mean squared residual error."""

    residual_sq = np.asarray(residual_sq, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if residual_sq.ndim != 2:
        raise ValueError("residual_sq must be a 2D array")
    if residual_sq.shape[0] != mask.shape[0]:
        raise ValueError("mask length must match residual row count")
    if residual_sq.size == 0 or not np.any(mask):
        return np.full(residual_sq.shape[1], np.nan, dtype=float)

    mean_residual = np.nanmean(residual_sq[mask], axis=0)
    total = np.nansum(mean_residual)
    if total <= 0:
        return np.full(residual_sq.shape[1], np.nan, dtype=float)
    return mean_residual / total * 100.0


def feature_contribution_table(
    residual_sq: np.ndarray,
    feature_names: list[str],
    masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Build a wide contribution table for named row subsets."""

    residual_sq = np.asarray(residual_sq, dtype=float)
    if residual_sq.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match residual feature count")

    out = pd.DataFrame({"feature": feature_names})
    for name, mask in masks.items():
        out[f"{name}_pct"] = contribution_percent(residual_sq, np.asarray(mask, dtype=bool))
    return out


def per_id_contribution_summary(
    residual_sq: np.ndarray,
    feature_names: list[str],
    ids: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Summarize per-animal residual contribution variability."""

    residual_sq = np.asarray(residual_sq, dtype=float)
    ids = np.asarray(ids)
    mask = np.asarray(mask, dtype=bool)
    rows = []
    for animal_id in pd.unique(ids):
        id_mask = (ids == animal_id) & mask
        if not np.any(id_mask):
            continue
        pct = contribution_percent(residual_sq, id_mask)
        for feature, value in zip(feature_names, pct, strict=True):
            rows.append(
                {
                    "animal_id": animal_id,
                    "feature": feature,
                    "contribution_pct": value,
                }
            )
    return pd.DataFrame(rows)


def concentration_summary(contribution: pd.DataFrame, subset_col: str) -> dict[str, float]:
    """Return top-feature concentration metrics for one contribution column."""

    values = contribution[subset_col].dropna().sort_values(ascending=False).to_numpy(dtype=float)
    if values.size == 0:
        return {"top3_share_pct": np.nan, "top5_share_pct": np.nan}
    return {
        "top3_share_pct": float(values[:3].sum()),
        "top5_share_pct": float(values[:5].sum()),
    }
