# Dashboard

This Streamlit dashboard is a reduced inspection layer for the manuscript-profile
pipeline. It loads saved local artifacts first and falls back to a small synthetic
demo only when those artifacts are missing.

The main tabs are:

- **Movement Features**: manuscript movement metrics such as velocity, absolute
  relative angle, sinuosity, rolling movement moments, AKDE/home-range estimates,
  and season encodings.
- **Model Learning**: autoencoder architecture, frozen threshold metadata, training
  history, and layer weight summaries when `autoencoder_model.h5` is available.
- **Results Explorer**: frozen row flags, animal-level ID summaries, and feature
  residual contribution tables.

Conceptually, the dashboard follows the secondary public-demo flow:

1. Preprocessing: inspect engineered movement features such as AKDE low/estimate/high,
   rolling velocity mean, rolling velocity standard deviation, turning angle summaries,
   and season encodings.
2. Training AE: inspect the saved dense autoencoder architecture, training history,
   threshold metadata, and layer weight summaries when a trained `.h5` model is available.
3. Results: interact with saved row-level anomaly flags, animal-level calls, and feature
   reconstruction residuals from the frozen manuscript-style evaluation.

Expected artifact directory contents include:

- `feature_columns_*.pkl`
- `train_processed_*.csv`, `val_processed_*.csv`, `test_processed_*.csv`
- `reconstruction_threshold.pkl`
- `id_rule_params.pkl`
- `test_processed_*_rows_with_flags_FROZEN.csv`
- `test_processed_*_id_summary_FROZEN.csv`
- `ae_feature_contributions_holdout.csv`
- optional `training_history.npy` and `autoencoder_model.h5`

Run from the repository root:

```bash
streamlit run dashboard/app.py
```

To generate a public-safe artifact set first:

```bash
python scripts/generate_manuscript_synthetic_inputs.py \
  --profile wtd_primary \
  --output-dir data/sample/manuscript_synthetic

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

python scripts/manuscript_feature_contributions.py \
  --profile wtd_primary \
  --processed-dir outputs/wtd_primary
```

The dashboard does not display raw field-study coordinates, exact animal IDs, or
unpublished collaborator data.

