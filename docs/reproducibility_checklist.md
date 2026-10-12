# Reproducibility Checklist

Before claiming model results, document each item below.

- Input data version and date range.
- Species, sex, genotype, and CWD-status inclusion criteria.
- Sampling interval and regularization target times.
- Minimum fixes per day and long-gap segmentation threshold.
- Feature list and rolling-window lengths.
- Animal-level train/validation/test split manifest.
- Confirmation that prevalence was calculated by unique animals, not GPS rows.
- Confirmation that encoders, scalers, feature order, and thresholds were fit only on training/validation data.
- Random seeds and repeated-split strategy, if used.
- Threshold statistic, source dataset, and percentile or selection rule.
- Row-level confusion matrix, sensitivity, specificity, precision, F1, and balanced accuracy.
- Animal-level aggregation rule and corresponding animal-level metrics.
- Missing-data handling.
- Whether AKDE-derived fields were computed before or after filtering/regularization.
- Any known limitations, such as small sample size, class imbalance, or duplicated exploratory notebooks.
