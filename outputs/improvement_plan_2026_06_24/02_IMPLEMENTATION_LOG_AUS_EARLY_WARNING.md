# Implementation Log: AUS Early-Warning Improvement

## What Changed

- Added train-only region-window weather anomaly features for rainfall, heat, dry-spell, evaporation, and radiation signals.
- Added past-year and rolling historical yield features by crop-region using only previous years.
- Installed/detected CatBoost and LightGBM where available.
- Added validation-only threshold tuning for classification.
- Added improved quantile and conformal interval reporting.
- Added new comparison and decision-support figures.

## Package Availability

| package | available |
| --- | --- |
| catboost | True |
| lightgbm | True |
| interpret | False |

## Output Files

- `data/processed/model_ready_panel_improved.csv`
- `outputs/tables/improved_model_metrics_by_window.csv`
- `outputs/tables/improved_test_predictions.csv`
- `outputs/tables/classification_threshold_analysis.csv`
- `outputs/tables/classification_metrics_tuned_thresholds.csv`
- `outputs/tables/improved_interval_summary.csv`
- `outputs/figures/fig08_pr_curve_by_window.png`
- `outputs/figures/fig09_threshold_tradeoff.png`
- `outputs/figures/fig10_interval_coverage_by_lead_time.png`
- `outputs/figures/fig11_old_vs_improved_rmse_r2.png`

## Acceptance Checks

- Improved panel shape: 4830 rows, 966 unique yield observations
- Missing forecast-window rows: 0
- Test years were not used for anomaly baselines, thresholds, calibration, or conformal residuals.
- Leakage fields `production_kt` and `area_000ha` remain excluded from training feature matrices.
