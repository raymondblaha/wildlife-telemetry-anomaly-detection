# Methods

This project is organized around a leakage-safe anomaly detection workflow for repeated GPS telemetry observations.

## 1. Telemetry Regularization

GPS fixes are first aligned to comparable target times. This prevents models from learning artifacts caused by irregular fix schedules rather than movement behavior. Days with too few fixes are removed, and long temporal gaps start new segments so movement features do not bridge invalid intervals.

The current R utility supports:

- target-hour matching, such as `00:00`, `08:00`, and `16:00`;
- retaining animal-days with at least a minimum number of fixes;
- segmenting trajectories after gaps longer than a configured threshold.

## 2. Movement Features

The Python preprocessing module derives movement summaries from ordered fixes:

- step distance;
- elapsed time;
- speed;
- bearing;
- turning angle;
- rolling movement means;
- time-of-day and seasonal indicators.

Feature definitions should remain stable across train, validation, and test data.

## 3. Animal-Level Splitting

Rows from the same animal are not independent. The pipeline splits by `animal_id` so that fixes from one animal do not appear in both training and evaluation partitions.

This is essential for an honest portfolio claim. A high row-level score is weak evidence if the same animals were leaked across splits.

## 4. Autoencoder Anomaly Scoring

The model learns to reconstruct a reference movement distribution, such as CWD-negative animals or one genotype group. Reconstruction error becomes the anomaly score.

The threshold must be selected from training or validation errors before final test evaluation. Do not tune the threshold on final test outcomes.

The public training entrypoint supports manuscript-style threshold policies:

- `validation_percentile`;
- `train_lowest_p95`;
- `train_quantile`;
- `val_row_fpr`;
- `val_id_flag_rate`;
- `mad`.

The default public config uses a validation percentile for simplicity, while the manuscript notebooks also used train-lowest and ID-rate policies in sensitivity experiments.

## 5. Reporting

Report two levels of results:

- row-level metrics for GPS fixes;
- animal-level metrics for biological interpretation.

Animal-level reporting requires a documented rule, such as "flag an animal if at least N fixes exceed the frozen threshold."

The pipeline also saves per-feature squared reconstruction residuals so feature contribution tables can be regenerated without retraining. These tables report the percent of mean squared residual assigned to each model input feature for all rows, flagged rows, and comparison rows.

## 6. Cross-Species and Genotype Transfer

Transfer experiments should be described as similarity tests unless the design directly supports disease-effect inference. For example, training on CWD-free genotype GG and testing on CWD-free genotype SS evaluates whether the held-out genotype is reconstructed similarly by the learned reference distribution.
