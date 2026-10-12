# Dashboard

This Streamlit dashboard is a public-safe analysis layer for the telemetry anomaly-detection project.

It uses synthetic telemetry by default and is intended to demonstrate:

- telemetry data-quality checks;
- retained fixes by animal and day;
- movement-feature distributions;
- validation-threshold thinking;
- row-level and animal-level anomaly summaries.

Run from the repository root:

```bash
python scripts/generate_sample_data.py
streamlit run dashboard/app.py
```

The dashboard does not display raw field-study coordinates, exact animal IDs, or unpublished collaborator data.
