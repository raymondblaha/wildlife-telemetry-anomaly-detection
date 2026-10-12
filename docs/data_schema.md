# Data Schema

The public pipeline expects one row per GPS fix. Raw study data should be transformed into this schema before modeling.

| Column | Type | Required | Notes |
|---|---:|---:|---|
| `animal_id` | string | yes | Use anonymized IDs in public outputs. |
| `timestamp` | datetime | yes | UTC or explicitly documented timezone. |
| `latitude` | numeric | yes | Do not publish exact protected-location coordinates. |
| `longitude` | numeric | yes | Do not publish exact protected-location coordinates. |
| `species` | string | recommended | Example: `mule_deer`, `white_tailed_deer`. |
| `sex` | string | recommended | Standardize values before modeling. |
| `genotype` | string | optional | Use documented mappings, for example WTD PRNP labels. |
| `label` | integer | required for supervised evaluation | Public demos use `0 = reference`, `1 = comparison/anomaly`. |
| `segment_id` | integer | recommended | Prevent movement metrics across long telemetry gaps. |

## Derived Feature Examples

| Feature | Meaning |
|---|---|
| `step_distance_m` | Distance between consecutive fixes for the same animal. |
| `elapsed_hours` | Hours between consecutive fixes. |
| `speed_m_per_hour` | Step distance divided by elapsed hours. |
| `turning_angle_deg` | Absolute change in movement bearing. |
| `rolling_3_step_mean_m` | Short-window movement smoothing. |
| `rolling_3_speed_mean` | Short-window speed smoothing. |
| `hour`, `month` | Temporal context for movement patterns. |

## Public Data Rule

Use synthetic or spatially masked data for public demos. Do not commit raw field-study coordinates, exact animal IDs, unpublished disease labels, or collaborator-owned files.
