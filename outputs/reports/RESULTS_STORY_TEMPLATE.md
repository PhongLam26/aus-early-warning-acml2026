# Results Story Template

## 1. Data Readiness

- Yield observations:
- Model-ready panel rows:
- Missing merge rows:
- Crops covered:
- Regions covered:

Key sentence:

> The final panel preserves the state-crop-year structure of the Australian winter crop dataset and expands each observation into multiple partial-season forecast windows.

## 2. Lead-Time Skill

Fill from `outputs/tables/model_metrics_by_window.csv` and `outputs/reports/lead_time_summary.md`.

- Best early window:
- Best overall window:
- Earliest useful warning point:
- Trade-off between lead time and accuracy:

Key sentence:

> Forecast skill improves as more seasonal weather becomes available, but the early windows show whether useful risk signals emerge before the full May-October season is observed.

## 3. Low-Yield Risk Classification

Fill from ROC-AUC, PR-AUC, F1, recall, and Brier score.

- Best classifier:
- Best calibrated window:
- Weakest crop/region:
- Risk threshold behavior:

Key sentence:

> The classification task translates yield forecasting into a decision-support signal by estimating the probability of crop-region yield shortfall.

## 4. Uncertainty And Calibration

Fill from calibration curve, interval coverage, interval width, and conformal qhat.

- Calibration quality:
- Interval coverage:
- Interval width:
- Whether intervals are useful for decision support:

Key sentence:

> Prediction intervals and calibration diagnostics show whether the model's risk statements are reliable enough for monitoring rather than only point prediction.

## 5. Crop-State Vulnerability

Fill from `crop_region_risk_summary.csv` and `fig05_crop_state_vulnerability_matrix.png`.

- Highest risk crop-region pairs:
- Lowest risk crop-region pairs:
- Crop-specific patterns:
- Region-specific patterns:

Key sentence:

> The crop-state vulnerability matrix converts model and historical risk into a practical state-level monitoring view.

## 6. Weather Response Patterns

Fill from `fig06_response_curves.png`.

- Rainfall response:
- Heat response:
- Dry-spell response:
- Compound stress response:

Key sentence:

> Stage-specific weather indicators help explain why the early-warning framing is more useful than a single full-season average.

## 7. Discussion Claims

Use:

- State-level yield-risk monitoring.
- Early warning and drought preparedness.
- Procurement and supply-risk planning.
- Soil as regional vulnerability context.

Avoid:

- Farm-level prediction.
- Causal proof of yield loss.
- Policy automation.
- Claiming all crop failures can be predicted early.
