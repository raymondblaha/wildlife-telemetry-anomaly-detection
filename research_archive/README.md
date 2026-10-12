# Research Archive

This folder preserves source-code snapshots from the manuscript archive that produced the
mule deer publication workflow and the white-tailed deer draft workflow.

## Contents

- `original_manuscript_code/white_tailed_deer/`: WTD preprocessing, AKDE review,
  relocation diagnostics, manual review, and autoencoder notebooks.
- `original_manuscript_code/mule_deer/60_40_Experiment/`: mule deer 60:40,
  baseline, infection-prevalence, and visual inspection workflows.
- `original_manuscript_code/cross_species/`: WTD/mule schedule normalization and
  genotype-transfer experiments.
- `original_manuscript_code/notes/Notes.txt`: supplied archive notes and file mapping.

## Public-Safe Handling

The files here are source-code snapshots, not a raw-data release. Notebook outputs were
cleared before publication, and machine-specific paths were replaced with `<LOCAL_PATH>`.
The folder name `60:40_Experiment` was normalized to `60_40_Experiment` for cross-platform
Git compatibility.

These files should be read as research provenance: they show the actual analytical work
used for the manuscript workflows. The reusable public pipeline lives in `scripts/`,
`src/telemetry_anomaly/`, and `R/`.

