# Round 3 Results And Decision

## Headline

Best residual paper-safe: May-Oct Ridge RMSE 0.948, R2 0.439. Best direct paper-safe: May-Oct Ridge RMSE 0.890, R2 0.505. Best identity/time: May-Aug Ridge RMSE 1.009, R2 0.364.

## Decision

Round 3 does not improve the paper-safe model. Keep Round 2 as the main result and report that much of the state-level predictability is carried by identity/time and lag-yield history.

## Best Per Window And Experiment

| forecast_window | experiment | model | alpha | n_train | n_val | n_test | MAE | R2 | RMSE | identity_RMSE | identity_R2 | RMSE_delta_vs_identity | R2_delta_vs_identity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| May-Aug | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.617039 | 0.399571 | 0.980829 | 1.0092 | 0.364338 | -0.0283668 | 0.0352325 |
| May-Aug | identity_time_only | Ridge | -1 | 701 | 116 | 149 | 0.686566 | 0.364338 | 1.0092 | 1.0092 | 0.364338 | 0 | 0 |
| May-Aug | residual_paper_safe | Ridge | 0.95 | 701 | 116 | 149 | 0.694872 | 0.320089 | 1.04373 | 1.0092 | 0.364338 | 0.034535 | -0.0442495 |
| May-Jul | direct_paper_safe | ElasticNet | -1 | 701 | 116 | 149 | 0.648463 | 0.409548 | 0.972645 | 1.0092 | 0.364338 | -0.0365502 | 0.04521 |
| May-Jul | identity_time_only | Ridge | -1 | 701 | 116 | 149 | 0.686566 | 0.364338 | 1.0092 | 1.0092 | 0.364338 | 0 | 0 |
| May-Jul | residual_paper_safe | LightGBM | 0.6 | 701 | 116 | 149 | 0.678255 | 0.375002 | 1.0007 | 1.0092 | 0.364338 | -0.00850052 | 0.0106633 |
| May-Jun | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.665866 | 0.457317 | 0.932471 | 1.0092 | 0.364338 | -0.0767242 | 0.0929785 |
| May-Jun | identity_time_only | Ridge | -1 | 701 | 116 | 149 | 0.686566 | 0.364338 | 1.0092 | 1.0092 | 0.364338 | 0 | 0 |
| May-Jun | residual_paper_safe | ElasticNet | 0.4 | 701 | 116 | 149 | 0.67578 | 0.374628 | 1.00099 | 1.0092 | 0.364338 | -0.00820124 | 0.0102895 |
| May-Oct | direct_paper_safe | Ridge | -1 | 701 | 116 | 149 | 0.539033 | 0.505207 | 0.890378 | 1.0092 | 0.364338 | -0.118818 | 0.140868 |
| May-Oct | identity_time_only | Ridge | -1 | 701 | 116 | 149 | 0.686566 | 0.364338 | 1.0092 | 1.0092 | 0.364338 | 0 | 0 |
| May-Oct | residual_paper_safe | Ridge | 1.1 | 701 | 116 | 149 | 0.58328 | 0.438891 | 0.94817 | 1.0092 | 0.364338 | -0.061026 | 0.0745525 |
| May-Sep | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.621225 | 0.424341 | 0.960384 | 1.0092 | 0.364338 | -0.0488112 | 0.0600024 |
| May-Sep | identity_time_only | Ridge | -1 | 701 | 116 | 149 | 0.686566 | 0.364338 | 1.0092 | 1.0092 | 0.364338 | 0 | 0 |
| May-Sep | residual_paper_safe | LightGBM | 0.65 | 701 | 116 | 149 | 0.657892 | 0.360984 | 1.01185 | 1.0092 | 0.364338 | 0.00265895 | -0.00335399 |

## Best Overall Rows

| forecast_window | experiment | model | alpha | n_train | n_val | n_test | MAE | R2 | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| May-Oct | direct_paper_safe | Ridge | -1 | 701 | 116 | 149 | 0.539033 | 0.505207 | 0.890378 |
| May-Oct | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.582032 | 0.497908 | 0.89692 |
| May-Oct | direct_paper_safe | ElasticNet | -1 | 701 | 116 | 149 | 0.566329 | 0.478232 | 0.914326 |
| May-Jun | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.665866 | 0.457317 | 0.932471 |
| May-Oct | residual_paper_safe | Ridge | 1.1 | 701 | 116 | 149 | 0.58328 | 0.438891 | 0.94817 |
| May-Sep | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.621225 | 0.424341 | 0.960384 |
| May-Sep | direct_paper_safe | Ridge | -1 | 701 | 116 | 149 | 0.617758 | 0.421322 | 0.962899 |
| May-Sep | direct_paper_safe | ElasticNet | -1 | 701 | 116 | 149 | 0.620396 | 0.416938 | 0.96654 |
| May-Jul | direct_paper_safe | ElasticNet | -1 | 701 | 116 | 149 | 0.648463 | 0.409548 | 0.972645 |
| May-Oct | residual_paper_safe | ElasticNet | 1.2 | 701 | 116 | 149 | 0.615914 | 0.40086 | 0.979775 |
| May-Aug | direct_paper_safe | LightGBM | -1 | 701 | 116 | 149 | 0.617039 | 0.399571 | 0.980829 |
| May-Jun | direct_paper_safe | ElasticNet | -1 | 701 | 116 | 149 | 0.662294 | 0.394429 | 0.985019 |
| May-Oct | residual_paper_safe | LightGBM | 0.75 | 701 | 116 | 149 | 0.625616 | 0.382849 | 0.994393 |
| May-Aug | direct_paper_safe | ElasticNet | -1 | 701 | 116 | 149 | 0.653347 | 0.380483 | 0.996298 |
| May-Jul | direct_paper_safe | Ridge | -1 | 701 | 116 | 149 | 0.671222 | 0.378587 | 0.997821 |
| May-Jul | residual_paper_safe | LightGBM | 0.6 | 701 | 116 | 149 | 0.678255 | 0.375002 | 1.0007 |
| May-Jun | residual_paper_safe | ElasticNet | 0.4 | 701 | 116 | 149 | 0.67578 | 0.374628 | 1.00099 |
| May-Jul | residual_paper_safe | ElasticNet | 0.65 | 701 | 116 | 149 | 0.675443 | 0.36877 | 1.00567 |
| May-Jun | residual_paper_safe | Ridge | 0.2 | 701 | 116 | 149 | 0.6833 | 0.367017 | 1.00707 |
| May-Jun | identity_time_only | Ridge | -1 | 701 | 116 | 149 | 0.686566 | 0.364338 | 1.0092 |

## Interpretation For Paper

This test is intentionally conservative. It asks whether paper-safe weather/anomaly/soil information explains the residual left after a crop/region/year baseline. If the residual model does not beat the identity/time baseline, the manuscript should avoid claiming that weather alone drives most predictive skill. It can still report weather/anomaly/soil as monitoring evidence, while presenting lag-yield history as an operational forecasting enhancement.
