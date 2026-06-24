# Round 2 Results: Ablation And Stress Validation

## Best Ablation Results

| forecast_window | feature_set | model | MAE | R2 | RMSE |
| --- | --- | --- | --- | --- | --- |
| May-Aug | full_operational | ElasticNet | 0.523443604866299 | 0.6638366980318446 | 0.7339010096291269 |
| May-Aug | full_operational | Ridge | 0.5353467768961767 | 0.6488399373870672 | 0.7500926217754506 |
| May-Aug | lag_yield_only | CatBoost | 0.5927607418045674 | 0.600402837863486 | 0.8001539110049523 |
| May-Aug | lag_yield_only | ElasticNet | 0.5922311724784769 | 0.595615408157687 | 0.8049328179680206 |
| May-Aug | lag_yield_only | Ridge | 0.5909066538095353 | 0.5926338568497613 | 0.8078947767539951 |
| May-Aug | lag_yield_only | LightGBM | 0.6169334432950075 | 0.5737968224265425 | 0.8263626436261627 |
| May-Aug | full_operational | CatBoost | 0.5704811801623108 | 0.5550016422896371 | 0.844387000783053 |
| May-Aug | identity_time_only | LightGBM | 0.6273298807549812 | 0.543174224767377 | 0.855534711152426 |
| May-Aug | identity_time_only | CatBoost | 0.6190362032817508 | 0.5409979424631546 | 0.8575701401396568 |
| May-Aug | full_operational | LightGBM | 0.5790286710538318 | 0.5146632555002522 | 0.8818280741235607 |
| May-Aug | weather_plus_soil | CatBoost | 0.6040036738610873 | 0.47360115691611004 | 0.918374460541949 |
| May-Aug | weather_plus_soil | ElasticNet | 0.6049924508114155 | 0.4543437746099086 | 0.9350221344676164 |

## Stress Validation Summary

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

## Top Feature Group Importance

| forecast_window | feature_set | model | feature_group | n_features | baseline_rmse | rmse_delta_mean | rmse_delta_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| May-Jun | full_operational | Ridge | lag_yield | 3 | 0.7470145760044083 | 0.6276302833086375 | 0.025080919129723173 |
| May-Jun | full_operational | Ridge | soil | 22 | 0.7470145760044083 | 0.10765892609433816 | 0.005585182926701757 |
| May-Jun | full_operational | Ridge | categorical_time | 3 | 0.7470145760044083 | 0.035123071108040804 | 0.005944777770908509 |
| May-Jun | full_operational | Ridge | weather_anomaly | 18 | 0.7470145760044083 | 0.02012823292078101 | 0.011690393312590301 |
| May-Jun | full_operational | Ridge | heat_cold | 11 | 0.7470145760044083 | 0.015522823850144673 | 0.016912632200692702 |
| May-Jun | weather_anomaly_soil_no_lag | Ridge | soil | 22 | 1.0124951656655747 | 0.24683064192802323 | 0.020781925343475803 |
| May-Jun | weather_anomaly_soil_no_lag | Ridge | categorical_time | 3 | 1.0124951656655747 | 0.175308669564667 | 0.018309740597877502 |
| May-Jun | weather_anomaly_soil_no_lag | Ridge | rainfall | 12 | 1.0124951656655747 | 0.03482690204851786 | 0.010414498693970594 |
| May-Jun | weather_anomaly_soil_no_lag | Ridge | weather_anomaly | 18 | 1.0124951656655747 | 0.0011866778806157497 | 0.011993579927822675 |
| May-Jun | weather_anomaly_soil_no_lag | Ridge | compound_stress | 3 | 1.0124951656655747 | 0.000625428295083097 | 0.002757933789263548 |
| May-Oct | full_operational | Ridge | lag_yield | 3 | 0.657593833293215 | 0.5803085227853051 | 0.032279144486707795 |
| May-Oct | full_operational | Ridge | heat_cold | 11 | 0.657593833293215 | 0.2618189149308495 | 0.032645071983999505 |
| May-Oct | full_operational | Ridge | energy_dryness | 8 | 0.657593833293215 | 0.06779833986024567 | 0.016106131388024435 |
| May-Oct | full_operational | Ridge | soil | 22 | 0.657593833293215 | 0.04425368284768205 | 0.007603176166187086 |
| May-Oct | full_operational | Ridge | weather_anomaly | 18 | 0.657593833293215 | 0.03833255405543727 | 0.011485986483767316 |

## Interpretation

The ablation table separates paper-safe weather/anomaly/soil evidence from operational forecasts that additionally use historical lag-yield features. If the operational model materially outperforms the paper-safe model, the paper should present both as different claims rather than mixing them.
