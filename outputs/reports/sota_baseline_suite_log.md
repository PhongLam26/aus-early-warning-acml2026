# SOTA-Style Baseline Suite Log

## Availability

| package | available |
| --- | --- |
| xgboost | True |
| lightgbm | True |
| catboost | True |
| interpret | False |
| pygam | True |
| torch | True |

## Best Comparator Per Window And Regime

| forecast_window | feature_regime | model_group | model | RMSE | R2 | status |
| --- | --- | --- | --- | --- | --- | --- |
| May-Aug | daily_weather_sequence_no_yield_history | Sequence | DailyWeather-GRU | 1.042 | 0.322 | ok |
| May-Aug | daily_weather_sequence_operational | Sequence | DailyWeather-GRU | 0.846 | 0.554 | ok |
| May-Aug | no_yield_history_weather_soil | Strong tabular ML | HistGradientBoosting | 0.968 | 0.416 | ok |
| May-Aug | operational_with_yield_history | Classical ML | ElasticNet | 0.724 | 0.673 | ok |
| May-Aug | past_only_yield_history | Historical | 3-year rolling mean | 0.779 | 0.622 | ok |
| May-Jul | daily_weather_sequence_no_yield_history | Sequence | DailyWeather-GRU | 1.154 | 0.168 | ok |
| May-Jul | daily_weather_sequence_operational | Sequence | DailyWeather-GRU | 0.718 | 0.679 | ok |
| May-Jul | no_yield_history_weather_soil | Strong tabular ML | CatBoost | 0.951 | 0.436 | ok |
| May-Jul | operational_with_yield_history | Classical ML | Ridge | 0.716 | 0.68 | ok |
| May-Jul | past_only_yield_history | Historical | 3-year rolling mean | 0.779 | 0.622 | ok |
| May-Jun | daily_weather_sequence_no_yield_history | Sequence | DailyWeather-GRU | 1.14 | 0.189 | ok |
| May-Jun | daily_weather_sequence_operational | Sequence | DailyWeather-GRU | 0.784 | 0.616 | ok |
| May-Jun | no_yield_history_weather_soil | Strong tabular ML | XGBoost | 0.914 | 0.478 | ok |
| May-Jun | operational_with_yield_history | Classical ML | ElasticNet | 0.719 | 0.677 | ok |
| May-Jun | past_only_yield_history | Historical | 3-year rolling mean | 0.779 | 0.622 | ok |
| May-Oct | daily_weather_sequence_no_yield_history | Sequence | DailyWeather-GRU | 1.02 | 0.35 | ok |
| May-Oct | daily_weather_sequence_operational | Sequence | DailyWeather-GRU | 0.718 | 0.678 | ok |
| May-Oct | no_yield_history_weather_soil | Classical ML | SVR-RBF | 0.821 | 0.579 | ok |
| May-Oct | operational_with_yield_history | Classical ML | Ridge | 0.66 | 0.728 | ok |
| May-Oct | past_only_yield_history | Historical | 3-year rolling mean | 0.779 | 0.622 | ok |
| May-Sep | daily_weather_sequence_no_yield_history | Sequence | DailyWeather-GRU | 0.998 | 0.378 | ok |
| May-Sep | daily_weather_sequence_operational | Sequence | DailyWeather-GRU | 0.721 | 0.676 | ok |
| May-Sep | no_yield_history_weather_soil | Classical ML | SVR-RBF | 0.881 | 0.516 | ok |
| May-Sep | operational_with_yield_history | Classical ML | Ridge | 0.704 | 0.691 | ok |
| May-Sep | past_only_yield_history | Historical | 3-year rolling mean | 0.779 | 0.622 | ok |

## Leakage Checks

- `production_kt`, `area_000ha`, target columns, and test-derived quantities are excluded from feature matrices.
- Tabular preprocessing is fit inside train-only pipelines.
- Hyperparameters are selected by validation RMSE, then test metrics are reported once.
- Sequence models use raw daily weather only up to the forecast-window cutoff and static features fit/scaled from train only.
- GRU sequence models use PyTorch; sequence comparator rows are completed GRU runs.
