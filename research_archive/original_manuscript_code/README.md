# Original Manuscript Code

This folder contains public-safe source snapshots from the supplied manuscript archive.

The point of this directory is provenance. These notebooks and R Markdown files are the
workflows Raymond actually used for the mule deer publication pipeline and the white-tailed
deer draft manuscript work. The cleaner scripts in `scripts/` and `src/` are public
refactors of the same workflow, but they are not pretending to be the original research
workspace.

## What Was Preserved

- White-tailed deer preprocessing, AKDE review, relocation diagnostics, manual review, and
  autoencoder notebooks.
- Mule deer 60:40, baseline, infection-prevalence, and visual inspection workflows.
- Cross-species schedule-normalization and genotype-transfer notebooks.
- Supplied archive notes.

## What Was Changed For GitHub

- Notebook outputs were cleared so stale results and rendered private data are not
  published.
- Local machine and external-drive paths were replaced with `<LOCAL_PATH>`.
- The original `60:40_Experiment` folder name was normalized to `60_40_Experiment` for
  cross-platform Git compatibility.

These files are not expected to run end-to-end without the protected raw telemetry and
local study files. For a runnable public demonstration, use the synthetic-data workflow in
the repository root README and inspect the resulting artifacts with the Streamlit dashboard.
