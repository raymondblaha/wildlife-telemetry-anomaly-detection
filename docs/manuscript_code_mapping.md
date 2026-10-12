# Manuscript Code Mapping

This repository does not publish protected telemetry, exact animal identifiers, local file paths, or unpublished split manifests. Instead, the exploratory manuscript notebooks are represented as public-safe, reusable pipeline components.

## Source Workflows Reviewed

The private research archive included white-tailed deer, mule deer, and cross-species notebooks/R Markdown workflows. The relevant manuscript logic included:

- train-only `MinMaxScaler` and `OneHotEncoder` fitting;
- frozen feature-column order saved to disk;
- ID-disjoint train/validation/test splits;
- dense autoencoder training with scaled mean squared reconstruction loss;
- threshold policies such as `train_lowest_p95`, validation row false-positive rate, validation ID flag rate, train quantile, and MAD;
- row-level reconstruction errors and row flags;
- animal-level aggregation by count and fraction rules;
- per-feature squared residual contribution percentages for manuscript review.

## Public Repo Implementation

| Manuscript workflow concept | Public-safe repo location |
| --- | --- |
| GPS regularization and minimum fixes/day | `R/01_regularize_fix_schedule.R` |
| Movement feature engineering | `src/telemetry_anomaly/preprocessing.py` |
| Train-only scaler/encoder and frozen feature order | `scripts/train_autoencoder.py` |
| TensorFlow/Keras dense autoencoder | `src/telemetry_anomaly/autoencoder.py` |
| Scaled mean squared reconstruction loss | `src/telemetry_anomaly/autoencoder.py` |
| Frozen threshold policies | `src/telemetry_anomaly/autoencoder.py` |
| Row-level metrics | `src/telemetry_anomaly/evaluation.py` |
| Animal-level aggregation | `src/telemetry_anomaly/evaluation.py`, `scripts/evaluate_holdout.py` |
| Feature residual contribution tables | `src/telemetry_anomaly/feature_contributions.py`, `scripts/feature_contributions.py` |
| Public visual inspection layer | `dashboard/app.py` |

## Reproduction Commands

Generate public-safe sample telemetry:

```bash
python scripts/generate_sample_data.py
```

Run the TensorFlow autoencoder pipeline:

```bash
pip install ".[deep-learning]"
python scripts/train_autoencoder.py --config config/example_config.yml
```

Recompute animal-level metrics from saved predictions:

```bash
python scripts/evaluate_holdout.py \
  --predictions outputs/example_run/test_predictions.csv \
  --output-dir outputs/example_run
```

Recompute manuscript-style feature contribution tables:

```bash
python scripts/feature_contributions.py \
  --predictions outputs/example_run/test_predictions.csv \
  --residuals outputs/example_run/test_feature_residuals.csv \
  --features outputs/example_run/feature_columns.joblib \
  --output-dir outputs/example_run
```

## Important Boundary

The public scripts are designed to show the scientific ML workflow without exposing sensitive wildlife data. They are not a raw release of the mule deer or white-tailed deer study data, and the sample outputs should not be interpreted as biological results.
