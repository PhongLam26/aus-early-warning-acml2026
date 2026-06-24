# Results Comparison After Improvement

## Regression

Baseline best result:

| Window | Model | RMSE | R2 |
|---|---|---:|---:|
| May-Oct | HistGradientBoosting | 0.868 | 0.530 |

Improved best result:

| Window | Model | RMSE | R2 |
|---|---|---:|---:|
| May-Oct | Ridge | 0.658 | 0.730 |

RMSE change: `-0.210`. R2 change: `0.200`.

## Tuned Classification Highlights

| forecast_window | model | strategy | threshold | test_precision | test_recall | test_f1 | test_pr_auc | test_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| May-Oct | LogisticRegression | precision_ge_0_3 | 0.27 | 0.46875 | 0.5555555555555556 | 0.5084745762711864 | 0.3904661429803322 | 0.1348246872102882 |
| May-Oct | LogisticRegression | best_f1 | 0.25 | 0.30864197530864196 | 0.9259259259259259 | 0.46296296296296297 | 0.3904661429803322 | 0.1348246872102882 |
| May-Oct | LogisticRegression | recall_ge_0_5 | 0.25 | 0.30864197530864196 | 0.9259259259259259 | 0.46296296296296297 | 0.3904661429803322 | 0.1348246872102882 |
| May-Oct | CatBoostClassifier | best_f1 | 0.33 | 0.3409090909090909 | 0.5555555555555556 | 0.4225352112676056 | 0.2857838263319248 | 0.15384615908574348 |
| May-Oct | CatBoostClassifier | recall_ge_0_5 | 0.33 | 0.3409090909090909 | 0.5555555555555556 | 0.4225352112676056 | 0.2857838263319248 | 0.15384615908574348 |
| May-Sep | HistGradientBoostingClassifier | best_f1 | 0.63 | 0.28169014084507044 | 0.7407407407407407 | 0.40816326530612246 | 0.2694019731348358 | 0.19740211874393754 |
| May-Sep | HistGradientBoostingClassifier | recall_ge_0_5 | 0.63 | 0.28169014084507044 | 0.7407407407407407 | 0.40816326530612246 | 0.2694019731348358 | 0.19740211874393754 |
| May-Sep | HistGradientBoostingClassifier | precision_ge_0_3 | 0.63 | 0.28169014084507044 | 0.7407407407407407 | 0.40816326530612246 | 0.2694019731348358 | 0.19740211874393754 |
| May-Oct | LightGBMClassifier | best_f1 | 0.25 | 0.3333333333333333 | 0.5185185185185185 | 0.4057971014492754 | 0.27488786548814315 | 0.14698622387647436 |
| May-Oct | LightGBMClassifier | recall_ge_0_5 | 0.25 | 0.3333333333333333 | 0.5185185185185185 | 0.4057971014492754 | 0.27488786548814315 | 0.14698622387647436 |

## Interval Highlights

| forecast_window | model | coverage | width | pinball_p10 | pinball_p50 | pinball_p90 |
| --- | --- | --- | --- | --- | --- | --- |
| May-Jun | ConformalBestPoint_ElasticNet | 0.7986577181208053 | 1.56468844398832 | 0.10660735893087198 | 0.27244548725496376 | 0.14170729033283455 |
| May-Jul | ConformalBestPoint_ElasticNet | 0.7919463087248322 | 1.422939978503379 | 0.10203671225080123 | 0.2573021310352364 | 0.13884210861160004 |
| May-Sep | ConformalBestPoint_ElasticNet | 0.7583892617449665 | 1.251473912652093 | 0.0925930112953208 | 0.23903590667037736 | 0.1531915199368908 |
| May-Oct | ConformalBestPoint_ElasticNet | 0.7516778523489933 | 1.1253429913850335 | 0.08499001403159452 | 0.21938971673707702 | 0.1490195452007513 |
| May-Aug | ConformalBestPoint_ElasticNet | 0.7114093959731543 | 1.25878874862055 | 0.09548110065445133 | 0.2617234388819227 | 0.16215224209866433 |
| May-Jul | SklearnQuantileGBR | 0.5033557046979866 | 1.035495776479107 | 0.10235004972790354 | 0.2982605206801325 | 0.24384032898037022 |
| May-Sep | SklearnQuantileGBR | 0.4697986577181208 | 0.8208368648696764 | 0.09481257011041691 | 0.3086565988212993 | 0.3162617218968332 |
| May-Aug | SklearnQuantileGBR | 0.4697986577181208 | 0.8690170065155698 | 0.09526163846262109 | 0.3079330642226077 | 0.2742930049659967 |
| May-Oct | CatBoostQuantile | 0.436241610738255 | 0.6932048286126439 | 0.0935978287828185 | 0.26597451545104367 | 0.2593880509825763 |
| May-Sep | CatBoostQuantile | 0.42953020134228187 | 0.6760843570727627 | 0.09182349928398767 | 0.2932741999485811 | 0.32047758316958486 |

## Interpretation

The improvement pass should be read as a model-selection and reporting upgrade. If the strongest regression gain is modest, the main value still comes from tuned risk thresholds, stronger calibrated classification views, and conformal uncertainty that is easier to defend in the paper.
