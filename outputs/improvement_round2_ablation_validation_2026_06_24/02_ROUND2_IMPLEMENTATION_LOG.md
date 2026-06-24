# Round 2 Implementation Log

## Implemented

- Added fixed ablation feature sets.
- Added time-split, rolling-origin, leave-one-region-out, and leave-one-crop-out stress validation.
- Added group permutation importance for paper-safe and operational feature sets.
- Generated paper tables and figures 12-16.

## Feature Set Validation

| feature_set | n_features | uses_production_kt | uses_area_000ha | uses_lag_feature |
| --- | --- | --- | --- | --- |
| identity_time_only | 3 | False | False | False |
| weather_stage_only | 33 | False | False | False |
| weather_plus_anomaly | 51 | False | False | False |
| weather_plus_soil | 55 | False | False | False |
| weather_anomaly_soil_no_lag | 73 | False | False | False |
| lag_yield_only | 6 | False | False | True |
| full_operational | 76 | False | False | True |

## Output Tables

- `round2_ablation_metrics.csv`
- `round2_stress_validation_metrics.csv`
- `round2_feature_group_importance.csv`
- `round2_paper_table_lead_time.csv`
- `round2_paper_table_ablation.csv`
- `round2_paper_table_validation.csv`

## Output Figures

- `fig12_ablation_rmse_by_window.png`
- `fig13_ablation_r2_delta_by_feature_set.png`
- `fig14_stress_validation_heatmap.png`
- `fig15_feature_group_importance.png`
- `fig16_paper_safe_vs_operational_model.png`
