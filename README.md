# Wildlife Telemetry Anomaly Detection

Machine learning pipeline for detecting CWD-associated movement anomalies from cervid GPS telemetry.

This repository presents a public-safe, reproducible version of research workflows used to study whether deep learning can identify chronic wasting disease (CWD) signals from animal movement. The core story is the ML pipeline: clean and regularize GPS telemetry, engineer biologically meaningful movement features, split and evaluate by animal ID, train autoencoder-style anomaly models, freeze thresholds before final evaluation, and summarize results at both the GPS-fix and animal levels.

Raw GPS coordinates, exact animal IDs, agency file paths, and unpublished split manifests are intentionally excluded.

## Research Context

The mule deer anomaly-detection pipeline represented in this repository was developed to support the peer-reviewed Movement Ecology publication:

Blaha, R. L. et al. (2026). [Use of deep learning to predict chronic wasting disease status based on animal movement](https://link.springer.com/article/10.1186/s40462-026-00668-4). Movement Ecology, 14, Article 57.

The white-tailed deer workflow represented here extends the same public-safe pipeline concepts for a draft manuscript being prepared for peer review in the same journal. That manuscript focuses on CWD-associated movement changes in white-tailed deer and is not yet represented here as a peer-reviewed publication.

This public repository is not a raw-data release for either study. Instead, it demonstrates the reproducible engineering and analysis patterns behind the work.

## What This Shows

- GPS fix schedule regularization for mule deer and white-tailed deer telemetry.
- Movement feature engineering: step length, speed, turning angle, rolling movement summaries, season/time features, and AKDE-ready outputs.
- Animal-level train/validation/test splitting to avoid leakage across repeated GPS fixes.
- TensorFlow/Keras autoencoder anomaly detection using frozen preprocessing, frozen feature order, and a threshold selected before final test evaluation.
- Manuscript-style threshold policies, including validation percentiles, train-lowest thresholds, row false-positive rate, ID flag rate, train quantiles, and MAD.
- Row-level and animal-level reporting so model metrics are not confused with independent biological replicates.
- Feature residual contribution tables showing which movement dimensions drive reconstruction error.
- Cross-species and genotype-transfer experiment design without publishing sensitive raw telemetry.
- A dashboard layer for inspecting autoencoder-style reconstruction error, feature residual contributions, and animal-level anomaly calls using synthetic data.

## Intended Workflow

1. Regularize GPS fixes to comparable daily target schedules.
2. Add segment IDs after long gaps so movement metrics do not bridge biologically invalid intervals.
3. Engineer movement and temporal features.
4. Split by animal ID, not by row.
5. Fit scalers/encoders only on the training set.
6. Train an autoencoder on the reference movement distribution.
7. Freeze the reconstruction-error threshold using training/validation data.
8. Evaluate on held-out animals and report both row-level and animal-level results.
9. Summarize feature-level squared residual contributions for model interpretation.

## Public-Safe Analysis Dashboard

The Streamlit dashboard is a supporting inspection layer, not the centerpiece of the project. It uses synthetic telemetry to demonstrate how the ML workflow can be audited and communicated without exposing protected wildlife locations or unpublished animal labels.

It is designed to show:

- a lightweight autoencoder-style reconstruction model trained on synthetic reference animals;
- reconstruction-error scoring and frozen validation-threshold behavior;
- feature residual contribution percentages, mirroring the publication-style review logic;
- row-level model metrics;
- animal-level anomaly summaries;
- telemetry quality and movement-feature review before modeling.

Run from the repository root:

```bash
python scripts/generate_sample_data.py
streamlit run dashboard/app.py
```

For the strongest ML/CWD framing in portfolio materials, use a screenshot from the **ML Model**, **Feature Contributions**, or **Animal Flags** tab rather than leading with the Telemetry Quality view.

## Reproducible Pipeline Commands

Generate synthetic public-safe telemetry:

```bash
python scripts/generate_sample_data.py
```

Run the manuscript-derived TensorFlow autoencoder workflow:

```bash
pip install ".[deep-learning]"
python scripts/train_autoencoder.py --config config/example_config.yml
```

Recompute animal-level evaluation from saved predictions:

```bash
python scripts/evaluate_holdout.py \
  --predictions outputs/example_run/test_predictions.csv \
  --output-dir outputs/example_run
```

Recompute manuscript-style feature contribution tables from saved residuals:

```bash
python scripts/feature_contributions.py \
  --predictions outputs/example_run/test_predictions.csv \
  --residuals outputs/example_run/test_feature_residuals.csv \
  --features outputs/example_run/feature_columns.joblib \
  --output-dir outputs/example_run
```

## Project Layout

```text
R/
  01_regularize_fix_schedule.R      # public-safe telemetry regularization utilities
src/telemetry_anomaly/
  preprocessing.py                  # feature engineering and grouped splits
  autoencoder.py                    # TensorFlow/Keras reconstruction model helpers
  evaluation.py                     # row-level and animal-level metrics
  feature_contributions.py          # squared-residual contribution summaries
scripts/
  generate_sample_data.py           # synthetic telemetry for demos/tests
  train_autoencoder.py              # config-driven training and artifact export
  evaluate_holdout.py               # row-level and animal-level metric regeneration
  feature_contributions.py          # manuscript-style residual contribution tables
dashboard/
  app.py                            # Streamlit analysis dashboard
docs/
  data_schema.md                    # expected public-safe input columns
  methods.md                        # modeling and validation design
  manuscript_code_mapping.md        # mapping from private notebooks to public scripts
  reproducibility_checklist.md      # controls needed before claiming results
  sensitive_data_policy.md          # what must stay out of GitHub
config/
  example_config.yml
data/sample/
  README.md
research_archive/
  README.md                         # notes on excluded exploratory notebooks
```

## Why This Matters

Telemetry projects can look deceptively simple if they are reduced to "train a model on GPS rows." The harder and more valuable work is preventing leakage, preserving biological interpretation, validating thresholds honestly, and making the pipeline reproducible enough for collaborators to audit.

This project is designed to make that work visible: not just a model, but a defensible scientific ML workflow.

## Data Note

The sample data in this repository is synthetic. It is only intended to exercise the pipeline shape. It should not be used for biological inference.
