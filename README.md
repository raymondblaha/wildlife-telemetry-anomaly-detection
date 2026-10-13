# Wildlife Telemetry Anomaly Detection

Machine learning pipeline for detecting CWD-associated movement anomalies from cervid GPS
telemetry.

This repository presents a public-safe, reproducible version of research workflows used to
study whether deep learning can identify chronic wasting disease (CWD) signals from animal
movement. The core story is the ML pipeline: clean and regularize GPS telemetry, engineer
biologically meaningful movement features, split and evaluate by animal ID, train
autoencoder-style anomaly models, freeze thresholds before final evaluation, and summarize
results at both the GPS-fix and animal levels.

Raw GPS coordinates, exact animal IDs, agency file paths, and unpublished split manifests
are intentionally excluded.

The runnable public demo, generated sample files, and dashboard use synthetic,
manuscript-shaped telemetry data. They are designed to demonstrate the workflow without
exposing protected collaborator GPS data.

## Research Context

The mule deer anomaly-detection pipeline represented in this repository was developed to
support the peer-reviewed Movement Ecology publication:

Blaha, R. L. et al. (2026). [Use of deep learning to predict chronic wasting disease
status based on animal movement](https://link.springer.com/article/10.1186/s40462-026-00668-4).
Movement Ecology, 14, Article 57.

The white-tailed deer workflow represented here extends the same public-safe pipeline
concepts for a draft manuscript being prepared for peer review in the same journal. That
manuscript focuses on CWD-associated movement changes in white-tailed deer and is not yet
represented here as a peer-reviewed publication.

This public repository is not a raw-data release for either study. Instead, it demonstrates
the reproducible engineering and analysis patterns behind the work.

Start with [`CASE_STUDY.md`](CASE_STUDY.md) for the employer-facing walkthrough of the
research problem, source workflows, autoencoder design, validation controls, results context,
and public-safe dashboard demo.

## What This Shows

- GPS fix schedule regularization for mule deer and white-tailed deer telemetry.
- Movement feature engineering: speed, turning angle, sinuosity, rolling movement summaries,
  season features, and AKDE/home-range summaries.
- Animal-level train/validation/test splitting to avoid leakage across repeated GPS fixes.
- TensorFlow/Keras autoencoder anomaly detection using frozen preprocessing, frozen feature
  order, and a threshold selected before final test evaluation.
- Manuscript-style threshold policies, including train-lowest thresholds, validation
  percentiles, row false-positive rate, ID flag rate, train quantiles, and MAD.
- Row-level and animal-level reporting so model metrics are not confused with independent
  biological replicates.
- Feature residual contribution tables showing which movement dimensions drive
  reconstruction error.
- Cross-species and genotype-transfer experiment design without publishing sensitive raw
  telemetry.
- A dashboard layer for inspecting autoencoder-style reconstruction error, feature residual
  contributions, and animal-level anomaly calls using synthetic data.

## Actual Manuscript Scripts

The original manuscript work lives in notebooks and R Markdown files, including WTD
autoencoder notebooks, mule deer infection/prevalence experiments, AKDE review files,
relocation diagnostics, manual trajectory review, and genotype-transfer experiments.
Public-safe source snapshots are preserved under
`research_archive/original_manuscript_code/`; see `docs/original_archive_inventory.md` for
the source-file inventory from the supplied archive.

Those files are the research provenance layer. Notebook outputs were cleared, local machine
paths were redacted, and the mule deer `60:40_Experiment` folder name was normalized to
`60_40_Experiment` for cross-platform Git compatibility. The files are not expected to
rerun without the protected raw telemetry and local study files.

The public manuscript-profile scripts are cleaned ports of that work. They preserve the
notebook decisions that matter for scientific review:

- WTD primary profile: `velo`, `abs_rel_angle`, `sinuosity`, rolling angle/velocity
  moments, `akde_low`, `akde_est`, `akde_high`, and `season`.
- WTD 4-pings / six-hour profile: same WTD feature set with the six-hour notebook defaults.
- Mule deer 60:40 profile: `velo`, `abs_rel_angle`, `sinuosity_28pt`, rolling angle/velocity
  moments, `low`, `est`, `high`, and `season`.
- Exact dense autoencoder shape: `32 -> 16 -> 4 -> 16 -> 32`.
- Scaled row loss: `mean_squared_error * (1 + 0.5 * input_dim)`.
- Default frozen threshold policy: `train_lowest_p95` with the notebook low-error subset
  settings.
- Frozen animal-level rules: WTD primary `COUNT=550`, WTD six-hour `COUNT=110`, mule deer
  60:40 `COUNT=75`, all with `FRAC=0.75` and `OR` aggregation.

The core scripts are:

| Step | Script | What it preserves from the notebooks |
| --- | --- | --- |
| 1 | `scripts/prepare_manuscript_features.py` | train-only scaler/encoder fitting, ID-disjoint validation split, frozen feature order |
| 2 | `scripts/train_manuscript_autoencoder.py` | TensorFlow/Keras AE, `32 -> 16 -> 4 -> 16 -> 32`, scaled MSE, checkpointing |
| 3 | `scripts/apply_manuscript_frozen_rule.py` | frozen row threshold and animal-level count/fraction rule |
| 4 | `scripts/manuscript_feature_contributions.py` | squared-residual feature contribution tables |

To create a local public-safe artifact run with manuscript-shaped synthetic data, use the
one-command runner:

```bash
pip install ".[deep-learning]"
python scripts/run_synthetic_manuscript_pipeline.py \
  --profile wtd_primary \
  --output-dir outputs/wtd_primary \
  --epochs 40
```

That command runs the full public demo flow:

1. generate manuscript-shaped synthetic train/test CSVs;
2. preprocess features with a train-only scaler and season encoder;
3. train the dense autoencoder;
4. apply the frozen row and animal-level rule;
5. write feature residual contribution tables.

The individual commands are:

```bash
python scripts/generate_manuscript_synthetic_inputs.py \
  --profile wtd_primary \
  --output-dir data/sample/manuscript_synthetic

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

python scripts/manuscript_feature_contributions.py \
  --profile wtd_primary \
  --processed-dir outputs/wtd_primary
```

Supported profiles are `wtd_primary`, `wtd_four_pings`, and `mule_60_40`.

## Intended Workflow

1. Regularize GPS fixes to comparable daily target schedules.
2. Add segment IDs after long gaps so movement metrics do not bridge biologically invalid
   intervals.
3. Engineer movement and temporal features.
4. Split by animal ID, not by row.
5. Fit scalers/encoders only on the training set.
6. Train an autoencoder on the reference movement distribution.
7. Freeze the reconstruction-error threshold using training/validation data.
8. Evaluate on held-out animals and report both row-level and animal-level results.
9. Summarize feature-level squared residual contributions for model interpretation.

## Reduced Dashboard

The Streamlit dashboard is a supporting inspection layer, not the centerpiece of the
project. It is designed to load saved local artifacts from the manuscript scripts first:
processed CSVs, `feature_columns_*.pkl`, `reconstruction_threshold.pkl`, `id_rule_params.pkl`,
training history, model weights, row flags, ID summaries, and feature contribution tables.

It follows the presentation flow:

- **1. Preprocessing**: movement metrics used by the manuscript profiles, including
  velocity, turning, sinuosity, rolling movement moments, AKDE/home-range estimates, and
  season features.
- **2. AE Model**: the dense autoencoder architecture, bottleneck, training-loss curves,
  frozen threshold metadata, and layer weight summaries when saved model files are
  available.
- **3. AE Results**: frozen row flags, animal-level calls, reconstruction-error
  distributions, and feature residual contributions.

Run from the repository root:

```bash
streamlit run dashboard/app.py
```

If manuscript artifacts are missing, the dashboard falls back to a reduced synthetic demo so
the public project still runs without protected data. For portfolio screenshots, lead with
**Movement Features**, **Model Learning**, or **Results Explorer**.

## Project Layout

```text
CASE_STUDY.md                         # employer-facing project walkthrough
R/
  01_regularize_fix_schedule.R        # public-safe telemetry regularization utilities
src/telemetry_anomaly/
  preprocessing.py                    # feature engineering and grouped splits
  autoencoder.py                      # TensorFlow/Keras reconstruction model helpers
  manuscript_profiles.py              # WTD and mule deer manuscript column profiles
  evaluation.py                       # row-level and animal-level metrics
  feature_contributions.py            # squared-residual contribution summaries
scripts/
  prepare_manuscript_features.py      # train-only scaler/encoder notebook port
  train_manuscript_autoencoder.py     # 32-16-4-16-32 AE notebook port
  apply_manuscript_frozen_rule.py     # frozen row and animal ID rule application
  manuscript_feature_contributions.py # holdout feature residual contribution tables
  generate_manuscript_synthetic_inputs.py # manuscript-shaped demo inputs
  generate_sample_data.py             # synthetic telemetry for demos/tests
  train_autoencoder.py                # config-driven demo training and artifact export
  evaluate_holdout.py                 # row-level and animal-level metric regeneration
  feature_contributions.py            # generic residual contribution tables
dashboard/
  app.py                              # Streamlit analysis dashboard
docs/
  data_schema.md                      # expected public-safe input columns
  methods.md                          # modeling and validation design
  manuscript_code_mapping.md          # mapping from private notebooks to public scripts
  original_archive_inventory.md       # inventory of supplied manuscript notebooks/Rmds
  reproducibility_checklist.md        # controls needed before claiming results
  sensitive_data_policy.md            # what must stay out of GitHub
config/
  example_config.yml
data/sample/
  README.md
research_archive/
  README.md                           # notes on public-safe source-code snapshots
  original_manuscript_code/           # stripped-output source snapshots from supplied archive
```

## Why This Matters

Telemetry projects can look deceptively simple if they are reduced to "train a model on GPS
rows." The harder and more valuable work is preventing leakage, preserving biological
interpretation, validating thresholds honestly, and making the pipeline reproducible enough
for collaborators to audit.

This project is designed to make that work visible: not just a model, but a defensible
scientific ML workflow.

## Data Note

The sample data in this repository is synthetic. It is only intended to exercise the
pipeline shape. It should not be used for biological inference.
