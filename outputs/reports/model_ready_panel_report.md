# Model Ready Panel Report

- Output: `data/processed/model_ready_panel.csv`
- Yield observations: `966`
- Forecast windows per observation: `5`
- Expected panel rows: `4830`
- Actual panel rows: `4830`
- Missing weather-merge rows: `0`

Targets are created from yield data only. `expected_yield_t_ha` is fit using train years only, and `low_yield_risk` thresholds are computed from train shortfall distributions by crop.

## Split Counts

| split | forecast_window | rows |
| --- | --- | --- |
| test | May-Aug | 149 |
| test | May-Jul | 149 |
| test | May-Jun | 149 |
| test | May-Oct | 149 |
| test | May-Sep | 149 |
| train | May-Aug | 701 |
| train | May-Jul | 701 |
| train | May-Jun | 701 |
| train | May-Oct | 701 |
| train | May-Sep | 701 |
| validation | May-Aug | 116 |
| validation | May-Jul | 116 |
| validation | May-Jun | 116 |
| validation | May-Oct | 116 |
| validation | May-Sep | 116 |

## Low-Yield Thresholds

| crop | low_yield_shortfall_threshold |
| --- | --- |
| Barley | 0.3361143523082599 |
| Canola | 0.23538168855564479 |
| Lupins | 0.2969567573004563 |
| Oats | 0.23121848822063432 |
| Wheat | 0.3639273368215884 |

## Leakage Policy

`production_kt` and `area_000ha` are retained as metadata for auditing, but training code excludes them from feature matrices by default.
