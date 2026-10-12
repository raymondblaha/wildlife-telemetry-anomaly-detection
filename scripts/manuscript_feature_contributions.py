"""Manuscript-style AE feature contributions from frozen holdout outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    return parser.parse_args()


def contrib_pct(residual_sq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean per-feature residual, normalized to percent across features."""

    if residual_sq.size == 0 or not np.any(mask):
        return np.full(residual_sq.shape[1], np.nan, dtype=float)
    means = np.nanmean(residual_sq[mask], axis=0)
    total = np.nansum(means)
    return (means / total) * 100.0 if total > 0 else np.full(residual_sq.shape[1], np.nan)


def topk_share_pct(values: np.ndarray, k: int) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    return float(np.sort(finite)[::-1][: min(k, finite.size)].sum())


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    processed_dir = Path(args.processed_dir)
    suffix = profile.processed_suffix

    feature_columns = joblib.load(processed_dir / f"feature_columns_{suffix}.pkl")
    processed = pd.read_csv(processed_dir / f"test_processed_{suffix}.csv")
    flags = pd.read_csv(processed_dir / f"test_processed_{suffix}_rows_with_flags_FROZEN.csv")
    residuals = pd.read_csv(
        processed_dir / f"test_processed_{suffix}_rows_with_recon_and_residuals.csv"
    )

    residual_sq = np.square(residuals[[f"{col}_resid" for col in feature_columns]].to_numpy())
    ids = processed["id"].to_numpy()
    mask_all = np.ones(len(processed), dtype=bool)
    mask_flag = flags["row_flag"].astype(bool).to_numpy()
    mask_eval = mask_all.copy()

    contrib_all = contrib_pct(residual_sq, mask_all)
    contrib_flag = contrib_pct(residual_sq, mask_flag)
    contrib_eval = contrib_pct(residual_sq, mask_eval)

    id_to_rows = defaultdict(list)
    for row_idx, animal_id in enumerate(ids):
        id_to_rows[animal_id].append(row_idx)

    def per_id_contrib_stats(mask: np.ndarray) -> dict[str, float | int]:
        per_id = []
        for animal_id, row_indices in id_to_rows.items():
            idx = np.asarray(row_indices, dtype=int)
            idx = idx[mask[idx]]
            if idx.size == 0:
                continue
            per_id.append(contrib_pct(residual_sq, np.isin(np.arange(len(ids)), idx)))
        if not per_id:
            return {"median_feature_sd_pp": np.nan, "n_ids": 0}
        matrix = np.vstack(per_id)
        return {
            "median_feature_sd_pp": float(np.nanmedian(np.nanstd(matrix, axis=0))),
            "n_ids": len(per_id),
        }

    contribution = pd.DataFrame(
        {
            "feature": feature_columns,
            "All_%": contrib_all,
            "Flagged_%": contrib_flag,
            "Eval_%": contrib_eval,
            "Delta_Flagged_minus_All_pp": contrib_flag - contrib_all,
        }
    ).sort_values("Flagged_%", ascending=False)

    err_all = per_id_contrib_stats(mask_all)
    err_flag = per_id_contrib_stats(mask_flag)
    err_eval = per_id_contrib_stats(mask_eval)
    summary = pd.DataFrame(
        [
            {
                "Group": "All rows",
                "Top3_share_%": topk_share_pct(contrib_all, 3),
                "Top5_share_%": topk_share_pct(contrib_all, 5),
                "Median_feat_SD_pp": err_all["median_feature_sd_pp"],
                "n_ids": err_all["n_ids"],
            },
            {
                "Group": "Flagged rows",
                "Top3_share_%": topk_share_pct(contrib_flag, 3),
                "Top5_share_%": topk_share_pct(contrib_flag, 5),
                "Median_feat_SD_pp": err_flag["median_feature_sd_pp"],
                "n_ids": err_flag["n_ids"],
            },
            {
                "Group": "Eval IDs",
                "Top3_share_%": topk_share_pct(contrib_eval, 3),
                "Top5_share_%": topk_share_pct(contrib_eval, 5),
                "Median_feat_SD_pp": err_eval["median_feature_sd_pp"],
                "n_ids": err_eval["n_ids"],
            },
        ]
    )
    summary["Delta_Flagged_minus_All_Top3_pp"] = (
        summary.loc[1, "Top3_share_%"] - summary.loc[0, "Top3_share_%"]
    )
    summary["Delta_Flagged_minus_All_Top5_pp"] = (
        summary.loc[1, "Top5_share_%"] - summary.loc[0, "Top5_share_%"]
    )
    summary["Delta_Flagged_minus_All_SD_pp"] = (
        summary.loc[1, "Median_feat_SD_pp"] - summary.loc[0, "Median_feat_SD_pp"]
    )

    contribution.to_csv(processed_dir / "ae_feature_contributions_holdout.csv", index=False)
    summary.to_csv(processed_dir / "ae_feature_contribution_summary_holdout.csv", index=False)
    print("Top feature contributions:")
    print(contribution.head(15).to_string(index=False))
    print("Saved manuscript-style feature contribution tables.")


if __name__ == "__main__":
    main()
