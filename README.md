# Wildlife Telemetry Anomaly Detection

Public-safe scaffold for cervid movement anomaly detection in chronic wasting disease (CWD) telemetry studies.

This repository refactors research notebooks and R Markdown workflows into a reproducible portfolio project. It demonstrates how GPS telemetry can be regularized, converted into movement features, split by animal ID, and evaluated with an autoencoder-style reconstruction-error workflow while keeping sensitive wildlife locations and animal identifiers out of the public repository.

## What This Shows

- GPS fix schedule regularization for mule deer and white-tailed deer telemetry.
- Movement feature engineering: step length, speed, turning angle, rolling movement summaries, season/time features, and AKDE-ready outputs.
- Animal-level train/validation/test splitting to avoid leakage across repeated GPS fixes.
- Autoencoder anomaly detection using frozen preprocessing, frozen feature order, and a threshold selected before final test evaluation.
- Row-level and animal-level reporting so model metrics are not confused with independent biological replicates.
- Cross-species and genotype-transfer experiment design without publishing sensitive raw telemetry.

## Repository Status

This is a public-facing project scaffold. The original research notebooks are being refactored into reusable scripts and modules. Raw GPS coordinates, exact animal IDs, agency file paths, and unpublished split manifests are intentionally excluded.

## Project Layout

```text
R/
  01_regularize_fix_schedule.R      # public-safe telemetry regularization utilities
src/telemetry_anomaly/
  preprocessing.py                  # feature engineering and grouped splits
  autoencoder.py                    # TensorFlow/Keras reconstruction model helpers
  evaluation.py                     # row-level and animal-level metrics
scripts/
  train_autoencoder.py              # config-driven training entrypoint
  generate_sample_data.py           # synthetic telemetry for demos/tests
dashboard/
  app.py                            # Streamlit analysis dashboard
docs/
  data_schema.md                    # expected public-safe input columns
  methods.md                        # modeling and validation design
  reproducibility_checklist.md      # controls needed before claiming results
  sensitive_data_policy.md          # what must stay out of GitHub
config/
  example_config.yml
data/sample/
  README.md
research_archive/
  README.md                         # notes on excluded exploratory notebooks
```

## Intended Workflow

1. Regularize GPS fixes to comparable daily target schedules.
2. Add segment IDs after long gaps so movement metrics do not bridge biologically invalid intervals.
3. Engineer movement and temporal features.
4. Split by animal ID, not by row.
5. Fit scalers/encoders only on the training set.
6. Train an autoencoder on the reference movement distribution.
7. Freeze the reconstruction-error threshold using training/validation data.
8. Evaluate on held-out animals and report both row-level and animal-level results.

## Dashboard

The repository includes a Streamlit dashboard for analyst-friendly review of the public-safe workflow:

```bash
python scripts/generate_sample_data.py
streamlit run dashboard/app.py
```

The dashboard uses synthetic telemetry to show data-quality checks, movement-feature distributions, frozen-threshold evaluation, and animal-level anomaly review.

## Why This Matters

Telemetry projects can look deceptively simple if they are reduced to "train a model on GPS rows." The harder and more valuable work is preventing leakage, preserving biological interpretation, validating thresholds honestly, and making the pipeline reproducible enough for collaborators to audit.

This project is designed to make that work visible.

## Data Note

The sample data in this repository is synthetic. It is only intended to exercise the pipeline shape. It should not be used for biological inference.
