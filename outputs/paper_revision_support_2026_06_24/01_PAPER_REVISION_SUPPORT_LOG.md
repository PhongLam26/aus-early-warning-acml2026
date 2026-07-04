# Paper Revision Support Log

Generated support artifacts for the ACML Australia early-warning manuscript revision.

## Output Summary

- `naive_baselines`: 20 rows
- `fixed_model_lead_time`: 20 rows
- `threshold_sensitivity`: 15 rows
- `threshold_sensitivity_summary`: 3 rows
- `confusion_matrix`: 1 rows
- `watch_list`: 10 rows
- `uncertainty_summary`: 5 rows
- `feature_importance_split`: 14 rows

## Key Checks

- Naive historical baselines use train-only means/trends or past-only chronological yield values.
- Confusion matrix and watch-list use May-Oct validation-selected classification threshold only.
- Uncertainty summary prioritizes conformal intervals and keeps coverage/width as diagnostics.
- Feature-importance figure separates operational lag-history importance from no-yield-history weather-soil importance.

## Preview: May-Oct Watch List

| rank | region | crop | year_start | forecast_window | y_proba | predicted_alert | low_yield_risk | yield_t_ha | expected_yield_t_ha | yield_shortfall | y_low | y_high | interval_width | suggested_use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Victoria | Oats | 2018 | May-Oct | 1.0 | 1 | 1 | 1.232 | 1.697 | 0.465 | 1.099 | 2.224 | 1.125 | Analyst review watch-list |
| 2 | New South Wales | Lupins | 2018 | May-Oct | 1.0 | 1 | 1 | 0.406 | 0.816 | 0.41 | 0.053 | 1.179 | 1.125 | Analyst review watch-list |
| 3 | Victoria | Lupins | 2018 | May-Oct | 1.0 | 1 | 0 | 0.491 | 0.765 | 0.274 | 0.348 | 1.473 | 1.125 | Analyst review watch-list |
| 4 | Victoria | Canola | 2018 | May-Oct | 1.0 | 1 | 0 | 1.232 | 1.245 | 0.012 | 0.591 | 1.717 | 1.125 | Analyst review watch-list |
| 5 | New South Wales | Wheat | 2018 | May-Oct | 0.273 | 1 | 1 | 0.777 | 1.756 | 0.979 | 0.974 | 2.1 | 1.125 | Analyst review watch-list |
| 6 | Queensland | Wheat | 2019 | May-Oct | 0.273 | 1 | 1 | 0.948 | 1.905 | 0.957 | 0.818 | 1.943 | 1.125 | Analyst review watch-list |
| 7 | Queensland | Wheat | 2018 | May-Oct | 0.273 | 1 | 1 | 1.0 | 1.88 | 0.88 | 1.085 | 2.21 | 1.125 | Analyst review watch-list |
| 8 | Queensland | Barley | 2019 | May-Oct | 0.273 | 1 | 1 | 0.995 | 1.859 | 0.865 | 1.034 | 2.159 | 1.125 | Analyst review watch-list |
| 9 | Tasmania | Oats | 2019 | May-Oct | 0.273 | 1 | 1 | 1.18 | 1.91 | 0.73 | 1.345 | 2.47 | 1.125 | Analyst review watch-list |
| 10 | Queensland | Canola | 2018 | May-Oct | 0.273 | 1 | 1 | 0.3 | 0.946 | 0.646 | 0.295 | 1.421 | 1.125 | Analyst review watch-list |
