# Paper-Ready Findings: AUS Early-Warning Project

## Recommended Framing

Use the project as a stage-aware, probabilistic, state-level early-warning framework. The strongest paper angle is not just point-yield prediction; it is the trade-off between early warning, calibrated low-yield risk, and uncertainty.

## Model Choice For Paper

Top improved regression candidates:

| forecast_window | model | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| May-Oct | Ridge | 0.42365648681531765 | 0.6575938332932154 | 0.730107513552382 |
| May-Oct | ElasticNet | 0.43877943347415405 | 0.6710226527657663 | 0.7189719345552661 |
| May-Jul | ElasticNet | 0.5146042620704728 | 0.7016084241913878 | 0.6927690805839501 |
| May-Sep | ElasticNet | 0.4780718133407547 | 0.7041555048716905 | 0.6905343229589305 |
| May-Sep | Ridge | 0.46714806279507015 | 0.7048987663980523 | 0.689880673803648 |
| May-Jul | Ridge | 0.5298376071396558 | 0.7199137830502319 | 0.6765282877508391 |
| May-Jun | ElasticNet | 0.5448909745099275 | 0.7209526037960298 | 0.6755940881985308 |
| May-Aug | ElasticNet | 0.5234468777638454 | 0.7339301704753795 | 0.6638099832542701 |

For low-yield risk, the best tuned threshold result is:

- Window: `May-Oct`
- Model: `LogisticRegression`
- Strategy: `precision_ge_0_3`
- Threshold: `0.270`
- Test recall: `0.556`
- Test F1: `0.508`
- Test PR-AUC: `0.390`

For uncertainty, use conformal intervals if quantile coverage is weaker:

- Best conformal window: `May-Jun`
- Model: `ConformalBestPoint_ElasticNet`
- Coverage: `0.799`
- Width: `1.565`

## Safe Claims

- The framework supports state-level winter-crop yield-risk monitoring.
- Partial-season weather features contain useful early-warning signal before full May-Oct weather is observed.
- Threshold tuning makes low-yield risk outputs more useful than a fixed 0.5 decision threshold.
- Conformal intervals provide a defensible uncertainty layer for decision support.

## Claims To Avoid

- Do not claim farm-level prediction.
- Do not claim the model proves heat, drought, or soil causally caused yield loss.
- Do not claim the model decides policy or replaces agronomic expertise.
- Do not claim all crop failures can be predicted early.
