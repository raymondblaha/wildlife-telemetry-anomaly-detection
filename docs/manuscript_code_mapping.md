# Manuscript Code Mapping

This repository does not publish protected telemetry, exact animal identifiers, local file paths,
or unpublished split manifests. Instead, the exploratory manuscript notebooks are represented as
public-safe, reusable pipeline components.

## Source Workflows Reviewed

The private research archive included white-tailed deer, mule deer, and cross-species notebooks/R
Markdown workflows. The relevant manuscript logic included:

- train-only `MinMaxScaler` and `OneHotEncoder` fitting;
- frozen feature-column order saved to disk;
- ID-disjoint train/validation/test splits;
- dense autoencoder training with scaled mean squared reconstruction loss;
- threshold policies such as `train_lowest_p95`, validation row false-positive rate,
  validation ID flag rate, train quantile, and MAD;
- row-level reconstruction errors and row flags;
- animal-level aggregation by count and fraction rules;
- per-feature squared residual contribution percentages for manuscript review.

## Faithful Notebook Ports

The files below are the closest public-safe representation of the manuscript
notebooks. They preserve the column profiles, train-only preprocessing, model
shape, threshold policy, frozen ID rule, and feature contribution outputs.

| Notebook behavior | Public-safe port |
| --- | --- |
| WTD primary, WTD six-hour, and mule deer 60:40 column profiles | `src/telemetry_anomaly/manuscript_profiles.py` |
| Drop notebook-specific non-feature columns | `scripts/prepare_manuscript_features.py` |
| Fit `OneHotEncoder` and `MinMaxScaler` on training animals only | `scripts/prepare_manuscript_features.py` |
| Save `encoder_*.pkl`, `scaler_*.pkl`, `feature_columns_*.pkl`, and split IDs | `scripts/prepare_manuscript_features.py` |
| Train dense autoencoder `32 -> 16 -> 4 -> 16 -> 32` | `scripts/train_manuscript_autoencoder.py` |
| Use L2 regularization `1e-4`, early stopping, and checkpoint weights | `scripts/train_manuscript_autoencoder.py` |
| Use scaled MSE loss with `LAMBDA_DEV=0.5` | `scripts/train_manuscript_autoencoder.py`, `src/telemetry_anomaly/autoencoder.py` |
| Freeze reconstruction threshold from training/validation errors | `scripts/train_manuscript_autoencoder.py`, `src/telemetry_anomaly/autoencoder.py` |
| Apply frozen row flags and animal-level count/fraction rule | `scripts/apply_manuscript_frozen_rule.py` |
| Export per-feature residual contribution percentages | `scripts/manuscript_feature_contributions.py` |
| Generate manuscript-shaped synthetic CSV inputs for demos | `scripts/generate_manuscript_synthetic_inputs.py` |

## Manuscript Script Families

| Study family | Public profile | Main public scripts |
| --- | --- | --- |
| Mule deer manuscript pipeline | `mule_60_40` | `prepare_manuscript_features.py`, `train_manuscript_autoencoder.py`, `apply_manuscript_frozen_rule.py`, `manuscript_feature_contributions.py` |
| White-tailed deer primary pipeline | `wtd_primary` | `prepare_manuscript_features.py`, `train_manuscript_autoencoder.py`, `apply_manuscript_frozen_rule.py`, `manuscript_feature_contributions.py` |
| White-tailed deer six-hour / 4-pings pipeline | `wtd_four_pings` | `prepare_manuscript_features.py`, `train_manuscript_autoencoder.py`, `apply_manuscript_frozen_rule.py`, `manuscript_feature_contributions.py` |

## Profile Defaults

| Profile | Source notebook family | Learning rate | Batch size | Frozen ID rule |
| --- | --- | ---: | ---: | --- |
| `wtd_primary` | White-tailed deer primary AE | `1e-6` | `128` | `COUNT=550`, `FRAC=0.75`, `OR` |
| `wtd_four_pings` | White-tailed deer six-hour / 4-pings AE | `1e-4` | `180` | `COUNT=110`, `FRAC=0.75`, `OR` |
| `mule_60_40` | Mule deer 60:40 experiment | `1e-4` | `128` | `COUNT=75`, `FRAC=0.75`, `OR` |

## Shared Public Modules

| Manuscript workflow concept | Public-safe repo location |
| --- | --- |
| GPS regularization and minimum fixes/day | `R/01_regularize_fix_schedule.R` |
| Movement feature engineering | `src/telemetry_anomaly/preprocessing.py` |
| Config-driven demo training | `scripts/train_autoencoder.py` |
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

Generate manuscript-shaped synthetic inputs:

```bash
python scripts/generate_manuscript_synthetic_inputs.py \
  --profile wtd_primary \
  --output-dir data/sample/manuscript_synthetic
```

Run the manuscript-profile TensorFlow autoencoder pipeline:

```bash
pip install ".[deep-learning]"
python scripts/prepare_manuscript_features.py \
  --profile wtd_primary \
  --train-csv data/sample/manuscript_synthetic/wtd_primary_train.csv \
  --test-csv data/sample/manuscript_synthetic/wtd_primary_test.csv \
  --output-dir outputs/wtd_primary

python scripts/train_manuscript_autoencoder.py \
  --profile wtd_primary \
  --processed-dir outputs/wtd_primary

python scripts/apply_manuscript_frozen_rule.py \
  --profile wtd_primary \
  --processed-dir outputs/wtd_primary
```

Recompute animal-level metrics from saved predictions:

```bash
python scripts/evaluate_holdout.py \
  --predictions outputs/example_run/test_predictions.csv \
  --output-dir outputs/example_run
```

Recompute manuscript-style feature contribution tables:

```bash
python scripts/manuscript_feature_contributions.py \
  --profile wtd_primary \
  --processed-dir outputs/wtd_primary
```

## Important Boundary

The public scripts are designed to show the scientific ML workflow without exposing sensitive
wildlife data. They are not a raw release of the mule deer or white-tailed deer study data, and the
sample outputs should not be interpreted as biological results.
