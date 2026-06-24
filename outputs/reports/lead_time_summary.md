# Lead-Time Summary

HistGradientBoosting is used as the default sklearn performance model in this environment.

| forecast_window | R2 | RMSE |
| --- | --- | --- |
| May-Aug | 0.4397173453835336 | 0.9474709944913827 |
| May-Jul | 0.3058575735868152 | 1.054597237736393 |
| May-Jun | 0.4603013120137047 | 0.9299037858624375 |
| May-Oct | 0.5303026028784041 | 0.8675038511230183 |
| May-Sep | 0.4693476023387947 | 0.9220774476950602 |

- Lowest RMSE window: `May-Oct`
- Highest R2 window: `May-Oct`

Interpretation note: earlier windows preserve more lead time; later windows should be treated as more information-rich near-harvest benchmarks.
