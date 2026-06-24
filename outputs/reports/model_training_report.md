# Model Training Report

- Metrics: `outputs/tables/model_metrics_by_window.csv`
- Test predictions: `outputs/tables/test_predictions.csv`
- Rolling-origin sensitivity: `outputs/tables/rolling_origin_sensitivity.csv`

## Optional Package Availability

- catboost: not installed; skipped
- lightgbm: not installed; skipped
- interpret: not installed; skipped

The pipeline keeps CatBoost, LightGBM, and EBM/interpret support optional. When those packages are installed, the same script adds them to the model comparison automatically.
