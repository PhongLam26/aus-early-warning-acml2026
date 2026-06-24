# Round 3 Implementation Log

## Implemented

- Froze the accepted Round 2 version before experimentation.
- Added a paper-safe residual experiment that fits `crop + region + year_start` first.
- Added direct paper-safe and residual paper-safe comparisons.
- Tuned residual shrinkage `alpha` on validation years only.
- Wrote new Round 3 tables and figures without overwriting Round 2 outputs.

## Feature Counts

| group | n_features |
| --- | ---: |
| identity_time_only | 3 |
| direct_paper_safe | 78 |
| residual_paper_safe | 75 |

## Leakage Checks

- No `production_kt`.
- No `area_000ha`.
- No lag-yield, rolling-yield, or expanding-yield features in paper-safe features.
- Test years 2017-2021 are not used to tune residual shrinkage.
