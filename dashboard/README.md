# Dashboard

This Streamlit dashboard is a public-safe analysis layer for the telemetry anomaly-detection project.

It uses synthetic telemetry by default and is intended to demonstrate:

- an autoencoder-style reconstruction model trained on reference movement;
- reconstruction-error scoring for held-out GPS fixes;
- feature residual contributions, expressed as percent of reconstruction error;
- row-level and animal-level anomaly summaries;
- telemetry data-quality checks;
- retained fixes by animal and day;
- movement-feature distributions.

The sidebar controls are active model controls:

- the threshold percentile changes the reconstruction-error cutoff and row-level flags;
- the minimum anomalous-fixes rule changes animal-level calls.

Run from the repository root:

```bash
python scripts/generate_sample_data.py
streamlit run dashboard/app.py
```

The dashboard does not display raw field-study coordinates, exact animal IDs, or unpublished collaborator data.
The public app uses a lightweight scikit-learn reconstruction model so it runs easily in a portfolio
environment. The protected research workflow uses TensorFlow/Keras autoencoders with saved scalers,
frozen feature columns, and held-out animal cohorts.
