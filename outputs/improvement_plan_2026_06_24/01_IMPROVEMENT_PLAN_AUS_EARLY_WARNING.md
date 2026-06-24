# Improvement Plan: AUS Early-Warning Project

## Goal

Improve the current Australia stage-aware early-warning baseline so the results are stronger for paper writing. The improvement pass keeps the same Australia-only scope and focuses on:

- better classification thresholding and recall,
- stronger tabular models when CatBoost/LightGBM are available,
- more credible uncertainty and interval reporting,
- refined weather anomaly and lag-yield features,
- clear before/after reporting in a separate folder.

## Baseline To Beat

Current baseline outputs are preserved under `AUS_project/outputs/tables/`.

- Model-ready panel: 4,830 rows, representing 966 yield observations across 5 forecast windows.
- Current best yield regression: May-Oct HistGradientBoosting, RMSE about 0.868 and R2 about 0.530.
- Current low-yield classification has usable ROC/PR signal but weak recall at fixed 0.5 threshold.
- Current sklearn quantile intervals under-cover, while conformal intervals are closer to the 80% target.

## Improvement Tasks

1. Install or detect model packages:
   - CatBoost and LightGBM are priority packages.
   - interpret/EBM remains optional.

2. Add refined model features:
   - weather anomaly features by region and forecast window using train-year baselines only,
   - prior-year yield and expanding rolling yield features by crop-region using past years only.

3. Add classification threshold tuning:
   - choose thresholds on validation predictions only,
   - evaluate best-F1, recall-at-least-0.5, and precision-constrained thresholds on test,
   - keep fixed 0.5 results for comparison but do not use them as the main decision threshold.

4. Improve interval reporting:
   - compare quantile and conformal intervals,
   - keep conformal as the paper-facing interval if quantile coverage remains weak.

5. Generate improved output files:
   - `outputs/tables/improved_model_metrics_by_window.csv`
   - `outputs/tables/improved_test_predictions.csv`
   - `outputs/tables/classification_threshold_analysis.csv`
   - `outputs/tables/classification_metrics_tuned_thresholds.csv`
   - `outputs/tables/improved_interval_summary.csv`

6. Generate new figures:
   - `fig08_pr_curve_by_window.png`
   - `fig09_threshold_tradeoff.png`
   - `fig10_interval_coverage_by_lead_time.png`
   - `fig11_old_vs_improved_rmse_r2.png`

7. Write final improvement reports in this folder:
   - `02_IMPLEMENTATION_LOG_AUS_EARLY_WARNING.md`
   - `03_RESULTS_COMPARISON_AFTER_IMPROVEMENT.md`
   - `04_PAPER_READY_FINDINGS_AUS_EARLY_WARNING.md`

## Acceptance Checks

- `model_ready_panel.csv` remains 4,830 rows with 966 unique yield observations.
- No missing forecast-window merge rows.
- Improved metric values have no missing values in key metrics.
- Tuned classification thresholds use validation only and are evaluated on test only.
- Test years 2017-2021 are not used for scaler fitting, target thresholds, calibration, conformal residuals, anomaly baselines, or lag feature construction.
- The final report states the safe paper claims and the claims to avoid.
