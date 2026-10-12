# Portfolio Audit

## Uploaded Bundle

The uploaded archive contains a strong research foundation:

- mule deer CWD anomaly-detection notebooks;
- white-tailed deer preprocessing, AKDE review, and autoencoder notebooks;
- cross-species and genotype-transfer experiments;
- detailed notes explaining study design and reproducibility controls.

## Public-Readiness Assessment

The bundle should not be published directly.

Main issues:

- hard-coded local paths across R Markdown and notebooks;
- exploratory duplicate filenames such as `copy` and `copy 2`;
- Mac metadata files;
- likely-sensitive animal IDs, disease-status lists, and GPS-derived details;
- stale saved notebook outputs unless rerun from a clean kernel.

## Recommended Public Repo Strategy

Publish a refactored scaffold first:

- `README.md` for the employer-facing story;
- `docs/` for methods, schema, reproducibility, and sensitive-data rules;
- `R/` for reusable telemetry regularization;
- `src/` and `scripts/` for Python modeling workflows;
- `data/sample/` with synthetic data only.

Then selectively promote one canonical experiment after the data policy is confirmed.

## Canonical Project Framing

Suggested repo name:

`wildlife-telemetry-anomaly-detection`

Suggested one-line description:

`Public-safe reproducible pipeline for cervid GPS telemetry regularization, movement-feature engineering, and autoencoder anomaly detection in CWD movement studies.`

Suggested GitHub topics:

`wildlife`, `telemetry`, `gps`, `anomaly-detection`, `autoencoder`, `movement-ecology`, `cwd`, `python`, `r`, `machine-learning`
