# Original Manuscript Archive Inventory

This document inventories the source archive that Raymond supplied for the manuscript
workflow:

`Combined_Anomaly_Detection/telemetry_project_code/`

The public repository should be read as a cleaned, public-safe refactor and map of these
source workflows. It is not meant to pretend that the refactored Python scripts were the
original research notebooks.

Public-safe source snapshots from this archive are preserved in
`research_archive/original_manuscript_code/`. Notebook outputs were cleared before
publication, local machine paths were replaced with `<LOCAL_PATH>`, and the original mule
deer `60:40_Experiment` folder was normalized to `60_40_Experiment` for cross-platform Git
compatibility.

## Primary Manuscript Source Files

| Study area | Original file | Role in the research workflow |
| --- | --- | --- |
| White-tailed deer | `white_tailed_deer/autoencoder_100.ipynb` | Main WTD autoencoder analysis using CWD-negative training data and CWD-positive test data. Includes train-only preprocessing, autoencoder training, frozen row/ID rules, and later movement/AKDE follow-up analyses. |
| White-tailed deer | `white_tailed_deer/autoencoder_100_4pings.ipynb` | WTD six-hour / four-fixes-per-day sensitivity analysis with frozen row and animal flags plus feature-contribution outputs. |
| White-tailed deer | `white_tailed_deer/WTD_Autoencoder_4hours.ipynb` | WTD four-hour sampling/regularization sensitivity workflow. |
| Mule deer | `mule_deer/60:40_Experiment/60_40_Experiment.ipynb` | Mule deer 60:40 autoencoder experiment using prepared train/test files and a grouped validation split from CWD-negative training animals. |
| Mule deer | `mule_deer/60:40_Experiment/100_Experiment_2 copy.ipynb` | Baseline mule deer experiment trained on CWD-free animals and tested on CWD-positive animals. |
| Mule deer | `mule_deer/60:40_Experiment/combined_MF_Infection_10_2 copy.ipynb` | Combined male/female mule deer infection experiment with 10% CWD prevalence in training and testing datasets. |
| Mule deer | `mule_deer/60:40_Experiment/Combined_MF_Infection_20 copy.ipynb` | Combined-sex conditional-autoencoder experiment at the 20% infection/prevalence setting. |
| Mule deer | `mule_deer/60:40_Experiment/Combined_MF_Infection_30 copy.ipynb` | Combined-sex conditional-autoencoder experiment at the 30% infection/prevalence setting. |
| Mule deer | `mule_deer/60:40_Experiment/Combined_MF_Infection_40 copy.ipynb` | Combined-sex conditional-autoencoder experiment at the 40% infection/prevalence setting. |

## Preprocessing, Movement Review, and AKDE Source Files

| Study area | Original file | Role in the research workflow |
| --- | --- | --- |
| White-tailed deer | `white_tailed_deer/White_Tailed_Deer_REDO3.Rmd` | WTD telemetry preprocessing: import, field checks, timestamp parsing, coordinate cleaning, and downstream preparation. |
| White-tailed deer | `white_tailed_deer/White_Tailed_Deer_Manual_Review.Rmd` | Manual review and visualization of individual WTD trajectories and labels. |
| White-tailed deer | `white_tailed_deer/Relocation_Analysis.Rmd` | GPS fix-success diagnostic before trajectory regularization, including fixes per deer-day and retention criteria. |
| White-tailed deer | `white_tailed_deer/Combo_AKDE_Results.Rmd` | Joins migratory and resident seven-day AKDE estimates and confidence limits back to cleaned records. |
| White-tailed deer | `white_tailed_deer/AKDE_REVIEW.Rmd` | Deep AKDE diagnostics for selected CWD-positive and CWD-negative animals. |
| Mule deer | `mule_deer/60:40_Experiment/Visual_Inspection_Mule_Deer.Rmd` | Manual-label review and static/interactive trajectory visualization for mule deer. |
| Mule deer | `mule_deer/60:40_Experiment/White_Tailed_Deer_REDO3.Rmd` | Preprocessing workflow copied into the mule deer experiment directory; verify species/input paths before reuse. |
| Cross-species | `cross_species/Mule_WTD_Normalization.Rmd` | Regularizes WTD and mule deer GPS schedules for cross-species comparison and starts new segments after long gaps. |

## Genotype and Transfer Experiments

| Original file | Interpretation |
| --- | --- |
| `cross_species/Genotype_CWD_Free_Vs_Free_WTD_GS_vs_GG copy.ipynb` | Train/reference genotype = CWD-free GS; test/comparison genotype = CWD-free GG. |
| `cross_species/Genotype_CWD_Free_Vs_Free_WTD_GG_vs_SS copy.ipynb` | Train/reference genotype = CWD-free GG; test/comparison genotype = CWD-free SS. |
| `cross_species/Genotype_CWD_Free_Vs_Free_WTD_GS_vs_SS copy.ipynb` | Train/reference genotype = CWD-free GS; test/comparison genotype = CWD-free SS. |
| `cross_species/Genotype_CWD_Free_Vs_Free_WTD_SS_vs_GS copy.ipynb` | Train/reference genotype = CWD-free SS; test/comparison genotype = CWD-free GS. |
| `cross_species/Genotype_CWD_Free_Vs_Free_WTD_SS_vs_GS copy 2.ipynb` | Later copied SS-vs-GS genotype-transfer notebook. |
| `cross_species/Genotype_CWD_Free_Vs_Free copy.ipynb` | Earlier CWD-free genotype comparison using a separate SS/SF/FF vocabulary. |
| `cross_species/Genotype_CWD_Free_Vs_Free copy 2.ipynb` | Later copy of the separate SS/SF/FF genotype comparison workflow. |

## Notebook Constants Confirmed From the Archive

| Original notebook | Code cells | Confirmed constants / controls |
| --- | ---: | --- |
| `white_tailed_deer/autoencoder_100.ipynb` | 32 | `FROZEN_COUNT=550`, `LAMBDA_DEV=0.5`, `LOWEST_FRAC=0.75`, `LOWEST_Q=90`; uses grouped split, train-only encoder/scaler, dense AE, thresholding, ID rule, AKDE and rolling movement features. |
| `white_tailed_deer/autoencoder_100_4pings.ipynb` | 16 | `FROZEN_COUNT=110`, `LAMBDA_DEV=0.5`, `LOWEST_FRAC=0.75`, `LOWEST_Q=90`; same core controls for six-hour/four-fix sensitivity analysis. |
| `white_tailed_deer/WTD_Autoencoder_4hours.ipynb` | 10 | `FROZEN_COUNT=75`, `LAMBDA_DEV=0.5`, `LOWEST_FRAC=0.75`, `LOWEST_Q=90`; four-hour WTD sensitivity workflow. |
| `mule_deer/60:40_Experiment/60_40_Experiment.ipynb` | 24 | `FROZEN_COUNT=75`, `LAMBDA_DEV=0.5`, `LOWEST_FRAC=0.75`, `LOWEST_Q=90`; grouped validation split, train-only preprocessing, dense AE, frozen row/ID rule, rolling movement features. |
| `mule_deer/60:40_Experiment/100_Experiment_2 copy.ipynb` | 21 | `FROZEN_COUNT=75`, `LAMBDA_DEV=0.5`, `LOWEST_FRAC=0.75`, `LOWEST_Q=90`; baseline CWD-free-to-CWD-positive experiment. |
| `mule_deer/60:40_Experiment/combined_MF_Infection_10_2 copy.ipynb` | 44 | `LAMBDA_DEV=0.75`; combined-sex 10% infection/prevalence experiment with autoencoder, thresholds, and residual analyses. |
| `mule_deer/60:40_Experiment/Combined_MF_Infection_20 copy.ipynb` | 24 | `LAMBDA_DEV=0.75`; combined-sex 20% experiment. |
| `mule_deer/60:40_Experiment/Combined_MF_Infection_30 copy.ipynb` | 21 | `LAMBDA_DEV=0.75`; combined-sex 30% experiment. |
| `mule_deer/60:40_Experiment/Combined_MF_Infection_40 copy.ipynb` | 27 | `LAMBDA_DEV=0.75`; combined-sex 40% experiment. |

## Public Repo Boundary

The public repo should emphasize three layers:

1. **Original manuscript work**: the notebooks and R Markdown files listed above.
2. **Public-safe refactor**: reusable Python/R scripts that preserve the workflow without
   exposing raw telemetry, exact animal IDs, agency paths, or unpublished split manifests.
3. **Reduced dashboard**: a viewer for saved artifacts plus a synthetic fallback for public
   demonstration.

The dashboard is not the scientific centerpiece. The manuscript notebooks and their public-safe
ports are the centerpiece.
