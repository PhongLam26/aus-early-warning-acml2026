# Project Method Plan

## Title

Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall Using Daily Weather and Soil Data

## Problem Statement

This project treats Australian winter-crop yield monitoring as an early-warning and decision-support problem. Instead of only predicting average yield after the season, the workflow estimates the probability that a crop-region-year will fall into a low-yield or shortfall state using partial-season daily weather indicators and regional soil background features.

The analysis unit is:

```text
region + crop + year_start
```

The project is state-level only. It does not claim farm-level prediction or causal attribution.

## Research Questions

1. How accurately can partial-season daily weather features predict Australian winter crop yield shortfall at state level?
2. How early in the May-October growing season can low-yield risk be detected?
3. Do regional soil characteristics improve yield-risk prediction and vulnerability profiling?
4. Which stage-specific weather indicators are most useful for policy and business decision support?

## Data

- Yield: `yield_panel.csv`, 966 observations from 1989 to 2021.
- Crops: Barley, Canola, Lupins, Oats, Wheat.
- Regions: New South Wales, Queensland, South Australia, Tasmania, Victoria, Western Australia.
- Weather: daily SILO weather with rainfall, temperature, radiation, vapor pressure, and evaporation.
- Soil: regional soil attributes aggregated into topsoil and subsoil background indicators.

## Feature Engineering

Daily weather is aggregated into cumulative forecast windows:

- May-Jun
- May-Jul
- May-Aug
- May-Sep
- May-Oct

Feature groups:

- Rainfall: seasonal rain, rain days, dry days, heavy-rain days, rolling rainfall maxima, longest dry spell.
- Heat/cold: mean and maximum temperature, heat days, heat degree days, frost days, cold days.
- Energy/dryness: radiation, evaporation, vapor pressure.
- Compound stress: hot-dry days, high-evaporation dry days.

Soil features are aggregated by region as topsoil 0-30 cm and subsoil 30-100 cm summaries for water availability, texture, organic carbon, nutrients, pH, cation exchange capacity, and depth.

## Targets

- `yield_t_ha`: direct yield regression target.
- `expected_yield_t_ha`: train-only crop-region trend baseline.
- `yield_shortfall`: expected yield minus actual yield.
- `low_yield_risk`: binary risk label where shortfall exceeds the crop-specific train percentile threshold.

## Models

The default pipeline runs with sklearn and automatically adds optional packages when installed.

- Baseline: historical mean/risk rate.
- Linear: Ridge, ElasticNet, Logistic Regression.
- Performance model: HistGradientBoosting.
- Optional: CatBoost, LightGBM, EBM/interpret.
- Probabilistic: sklearn quantile gradient boosting.
- Uncertainty: split conformal prediction using validation residuals.

## Evaluation

Primary split:

- Train: 1989-2012.
- Validation: 2013-2016.
- Test: 2017-2021.

Metrics:

- Regression: MAE, RMSE, R2.
- Classification: ROC-AUC, PR-AUC, F1, recall, Brier score.
- Probabilistic: pinball loss, P10-P90 interval coverage and width.
- Uncertainty: conformal coverage, width, and qhat.
- Robustness: rolling-origin sensitivity with three folds.

## Decision-Support Outputs

The project produces lead-time skill charts, calibration curves, prediction interval plots, crop-state vulnerability matrices, empirical response curves, and a dashboard-style risk snapshot.

Allowed claim: the workflow supports state-level agricultural risk monitoring, drought preparedness, procurement planning, and supply-risk assessment.

Avoided claim: the workflow does not decide policy, prove weather causality, or replace farm-level agronomic monitoring.
