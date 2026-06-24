# Round 2 Paper-Ready Tables And Claims

## Recommended Reporting Split

- **Paper-safe scientific model:** `weather_anomaly_soil_no_lag`.
- **Operational forecasting model:** `full_operational`.

Operational best RMSE is 0.658 (May-Oct, Ridge); paper-safe best RMSE is 0.889 (May-Oct, Ridge).

## Lead-Time Table

| forecast_window | feature_set | model | MAE | R2 | RMSE |
| --- | --- | --- | --- | --- | --- |
| May-Aug | full_operational | ElasticNet | 0.523443604866299 | 0.6638366980318446 | 0.7339010096291269 |
| May-Jul | full_operational | ElasticNet | 0.5146227792386395 | 0.6927508065379347 | 0.7016292896596281 |
| May-Jun | full_operational | ElasticNet | 0.5448993400953628 | 0.675600500714467 | 0.7209454782432281 |
| May-Oct | full_operational | Ridge | 0.4236564868153168 | 0.7301075135523823 | 0.657593833293215 |
| May-Sep | full_operational | ElasticNet | 0.4781146305008212 | 0.6904791112590918 | 0.7042183161806282 |
| May-Aug | weather_anomaly_soil_no_lag | LightGBM | 0.6211426869933483 | 0.40663460697241394 | 0.9750422648289532 |
| May-Jul | weather_anomaly_soil_no_lag | CatBoost | 0.6689432539830402 | 0.42261861804555234 | 0.961819820617607 |
| May-Jun | weather_anomaly_soil_no_lag | LightGBM | 0.6584855637474262 | 0.4701969304215309 | 0.9213392432804131 |
| May-Oct | weather_anomaly_soil_no_lag | Ridge | 0.5362645525274213 | 0.5071056332109963 | 0.888667427208413 |
| May-Sep | weather_anomaly_soil_no_lag | LightGBM | 0.621128939100965 | 0.4411609657742742 | 0.9462495836271952 |

## Ablation Table

| forecast_window | feature_set | model | MAE | R2 | RMSE |
| --- | --- | --- | --- | --- | --- |
| May-Aug | full_operational | ElasticNet | 0.523443604866299 | 0.6638366980318446 | 0.7339010096291269 |
| May-Aug | lag_yield_only | CatBoost | 0.5927607418045674 | 0.600402837863486 | 0.8001539110049523 |
| May-Aug | identity_time_only | LightGBM | 0.6273298807549812 | 0.543174224767377 | 0.855534711152426 |
| May-Aug | weather_plus_soil | CatBoost | 0.6040036738610873 | 0.47360115691611004 | 0.918374460541949 |
| May-Aug | weather_stage_only | LightGBM | 0.5960735285441191 | 0.43976159596714615 | 0.9474335785799931 |
| May-Aug | weather_anomaly_soil_no_lag | LightGBM | 0.6211426869933483 | 0.40663460697241394 | 0.9750422648289532 |
| May-Aug | weather_plus_anomaly | LightGBM | 0.6277988276789556 | 0.400148199199316 | 0.9803571445077324 |
| May-Jul | full_operational | ElasticNet | 0.5146227792386395 | 0.6927508065379347 | 0.7016292896596281 |
| May-Jul | lag_yield_only | CatBoost | 0.5927607418045674 | 0.600402837863486 | 0.8001539110049523 |
| May-Jul | identity_time_only | LightGBM | 0.6273298807549812 | 0.543174224767377 | 0.855534711152426 |
| May-Jul | weather_plus_soil | Ridge | 0.6275396631578994 | 0.45325310505908756 | 0.9359561392795682 |
| May-Jul | weather_stage_only | Ridge | 0.6334934054891468 | 0.44313703535127846 | 0.9445751203265406 |
| May-Jul | weather_anomaly_soil_no_lag | CatBoost | 0.6689432539830402 | 0.42261861804555234 | 0.961819820617607 |
| May-Jul | weather_plus_anomaly | ElasticNet | 0.6642889363385945 | 0.38615406892264736 | 0.9917267447951618 |
| May-Jun | full_operational | ElasticNet | 0.5448993400953628 | 0.675600500714467 | 0.7209454782432281 |
| May-Jun | lag_yield_only | CatBoost | 0.5927607418045674 | 0.600402837863486 | 0.8001539110049523 |
| May-Jun | identity_time_only | LightGBM | 0.6273298807549812 | 0.543174224767377 | 0.855534711152426 |
| May-Jun | weather_plus_soil | CatBoost | 0.6623200206678131 | 0.48002844635175235 | 0.9127506002866078 |
| May-Jun | weather_stage_only | LightGBM | 0.6485371344535367 | 0.4719079520202275 | 0.9198502879656464 |
| May-Jun | weather_anomaly_soil_no_lag | LightGBM | 0.6584855637474262 | 0.4701969304215309 | 0.9213392432804131 |

## Validation Table

| protocol | feature_set | model | mean_RMSE | mean_R2 | folds |
| --- | --- | --- | --- | --- | --- |
| leave_one_crop_out | full_operational | Ridge | 0.7313359629349558 | 0.5199283881231307 | 5 |
| leave_one_crop_out | full_operational | LightGBM | 0.8752455556538844 | 0.34902323263741253 | 5 |
| leave_one_crop_out | weather_anomaly_soil_no_lag | Ridge | 1.0707664379489523 | 0.066546245623804 | 5 |
| leave_one_crop_out | weather_anomaly_soil_no_lag | LightGBM | 1.2080753995904567 | -0.1923909986161295 | 5 |
| leave_one_region_out | full_operational | Ridge | 0.8231445610085084 | -0.3562192922010015 | 6 |
| leave_one_region_out | full_operational | LightGBM | 0.9023849836569174 | -0.06350835507489062 | 6 |
| leave_one_region_out | weather_anomaly_soil_no_lag | LightGBM | 0.9743781255690226 | -0.2850686446079508 | 6 |
| leave_one_region_out | weather_anomaly_soil_no_lag | Ridge | 1.1703994597598883 | -1.5784348370572647 | 6 |
| rolling_origin | full_operational | Ridge | 0.5157908076985456 | 0.6978578553135245 | 3 |
| rolling_origin | full_operational | LightGBM | 0.52671671116801 | 0.6891627536802319 | 3 |
| rolling_origin | weather_anomaly_soil_no_lag | LightGBM | 0.5777036001895967 | 0.6310894650109088 | 3 |
| rolling_origin | weather_anomaly_soil_no_lag | Ridge | 0.6775096359504507 | 0.48757650480712733 | 3 |
| time_split | full_operational | Ridge | 0.7159027161042709 | 0.6794144890761725 | 1 |
| time_split | full_operational | LightGBM | 0.8623097481238784 | 0.5356227571112495 | 1 |
| time_split | weather_anomaly_soil_no_lag | LightGBM | 0.9626596851158837 | 0.4199182373793978 | 1 |
| time_split | weather_anomaly_soil_no_lag | Ridge | 0.9772234160526748 | 0.4025436642229761 | 1 |

## Safe Claims

- Stage-aware weather and soil features can be evaluated separately from historical yield memory.
- Operational performance improves when lag-yield features are allowed, but that should be labelled as operational forecasting rather than pure weather early warning.
- Stress validation identifies which crops and regions are harder to transfer across.

## Limitations To Write

- Leave-one-crop-out is a hard unseen-crop test and may underperform.
- Region-level soil features should be treated as background vulnerability indicators, not causal farm-level explanations.
- The framework remains state-level and should not be presented as farm-level prediction.
