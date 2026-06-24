# Round 3 Plan: Paper-Safe Residual Early-Warning Test

## Motivation

Round 2 showed that the operational model is strong, but much of that strength comes from lag-yield/history features. The next paper-safe question is whether stage-aware weather, weather anomalies, and soil explain yield variation beyond a crop/region/year baseline.

## Frozen Baseline

The accepted Round 2 version is frozen at:

`AUS_project/outputs/safe_versions/round2_safe_2026_06_24/`

This Round 3 experiment must not overwrite that snapshot or the Round 2 result tables.

## Experiment

- Use `data/processed/model_ready_panel_improved.csv`.
- Keep the same split: train `1989-2012`, validation `2013-2016`, test `2017-2021`.
- Run each forecast window from May-Jun to May-Oct.
- Fit a baseline model using only `crop`, `region`, and `year_start`.
- Fit a residual model on train residuals using paper-safe features only:
  - stage weather features,
  - train-derived weather anomaly features already in the improved panel,
  - soil background features,
  - no lag-yield/history features,
  - no `production_kt`,
  - no `area_000ha`.
- Predict final yield as `baseline_prediction + residual_prediction`.

## Outputs

- `outputs/tables/round3_paper_safe_residual_metrics.csv`
- `outputs/tables/round3_paper_safe_residual_comparison.csv`
- `outputs/figures/fig17_round3_residual_vs_direct_rmse.png`
- `outputs/figures/fig18_round3_incremental_weather_signal.png`
- `outputs/improvement_round3_paper_safe_residual_2026_06_24/02_ROUND3_IMPLEMENTATION_LOG.md`
- `outputs/improvement_round3_paper_safe_residual_2026_06_24/03_ROUND3_RESULTS_AND_DECISION.md`

## Decision Rule

- If residual paper-safe models improve over identity/time-only and direct paper-safe models, Round 3 can strengthen the paper's early-warning evidence.
- If they do not improve, keep Round 2 as the main result and use Round 3 as honest negative evidence that history/identity effects dominate much of the state-level signal.
