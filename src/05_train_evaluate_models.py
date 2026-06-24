from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    add_metric,
    configured_path,
    ensure_output_dirs,
    finite_or_nan,
    load_config,
    markdown_table,
    safe_metric,
    write_markdown,
)


TARGET_COLUMNS = {
    "yield_t_ha",
    "expected_yield_t_ha",
    "yield_shortfall",
    "low_yield_risk",
    "low_yield_shortfall_threshold",
}
LEAKAGE_COLUMNS = {"production_kt", "area_000ha"}
META_COLUMNS = {
    "row_id",
    "panel_key",
    "season",
    "split",
    "forecast_window",
    "window_start_month",
    "window_end_month",
    "window_order",
    "lat",
    "lon",
    "soil_lat",
    "soil_lon",
}
CATEGORICAL_COLUMNS = ["region", "crop"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def optional_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_preprocessor(feature_columns: list[str], df: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = TARGET_COLUMNS | LEAKAGE_COLUMNS | META_COLUMNS
    features = []
    for column in df.columns:
        if column in excluded:
            continue
        if column.startswith("soil_") or column in CATEGORICAL_COLUMNS or column == "year_start":
            features.append(column)
        elif column in {
            "n_weather_days",
            "rain_sum",
            "rain_mean",
            "rain_days",
            "dry_days",
            "max_3day_rain",
            "max_7day_rain",
            "max_consecutive_dry_days",
            "heavy_rain_days_10",
            "heavy_rain_days_25",
            "tmax_mean",
            "tmax_max",
            "tmin_mean",
            "tmin_min",
            "heat_days_25",
            "heat_days_30",
            "heat_days_35",
            "heat_degree_days_30",
            "frost_days_0",
            "cold_days_5",
            "radiation_sum",
            "radiation_mean",
            "evap_sum",
            "evap_mean",
            "vp_mean",
            "vp_min",
            "vp_max",
            "hot_dry_days_30_rain_lt_1",
            "high_evap_dry_days",
            "heat_and_low_rain_days",
        }:
            features.append(column)
    return features


def historical_mean_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    global_mean = float(train[target].mean())
    crop_mean = train.groupby("crop")[target].mean().to_dict()
    crop_region_mean = train.groupby(["crop", "region"])[target].mean().to_dict()
    preds = []
    for _, row in test.iterrows():
        key = (row["crop"], row["region"])
        preds.append(crop_region_mean.get(key, crop_mean.get(row["crop"], global_mean)))
    return np.asarray(preds, dtype=float)


def historical_risk_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_rate = float(train["low_yield_risk"].mean())
    crop_rate = train.groupby("crop")["low_yield_risk"].mean().to_dict()
    crop_region_rate = train.groupby(["crop", "region"])["low_yield_risk"].mean().to_dict()
    preds = []
    for _, row in test.iterrows():
        key = (row["crop"], row["region"])
        preds.append(crop_region_rate.get(key, crop_rate.get(row["crop"], global_rate)))
    return np.asarray(preds, dtype=float)


def regression_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }
    if optional_module("lightgbm"):
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=10,
            random_state=random_state,
            verbose=-1,
        )
    if optional_module("catboost"):
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            depth=4,
            learning_rate=0.04,
            random_seed=random_state,
            verbose=False,
            loss_function="RMSE",
        )
    if optional_module("interpret"):
        from interpret.glassbox import ExplainableBoostingRegressor

        models["EBM"] = ExplainableBoostingRegressor(random_state=random_state)
    return models


def classification_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }
    if optional_module("lightgbm"):
        from lightgbm import LGBMClassifier

        models["LightGBMClassifier"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=10,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    if optional_module("catboost"):
        from catboost import CatBoostClassifier

        models["CatBoostClassifier"] = CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.04,
            random_seed=random_state,
            verbose=False,
            loss_function="Logloss",
            auto_class_weights="Balanced",
        )
    return models


def evaluate_regression(rows, forecast_window, target, model_name, y_true, y_pred) -> None:
    n = len(y_true)
    add_metric(rows, forecast_window, target, "regression", model_name, "MAE", mean_absolute_error(y_true, y_pred), n)
    add_metric(rows, forecast_window, target, "regression", model_name, "RMSE", rmse(y_true, y_pred), n)
    add_metric(rows, forecast_window, target, "regression", model_name, "R2", safe_metric(r2_score, y_true, y_pred), n)


def evaluate_classification(rows, forecast_window, model_name, y_true, proba) -> None:
    label = (proba >= 0.5).astype(int)
    n = len(y_true)
    add_metric(rows, forecast_window, "low_yield_risk", "classification", model_name, "ROC_AUC", safe_metric(roc_auc_score, y_true, proba), n)
    add_metric(rows, forecast_window, "low_yield_risk", "classification", model_name, "PR_AUC", safe_metric(average_precision_score, y_true, proba), n)
    add_metric(rows, forecast_window, "low_yield_risk", "classification", model_name, "F1_at_0.5", safe_metric(f1_score, y_true, label), n)
    add_metric(rows, forecast_window, "low_yield_risk", "classification", model_name, "Recall_at_0.5", safe_metric(recall_score, y_true, label), n)
    add_metric(rows, forecast_window, "low_yield_risk", "classification", model_name, "Brier", safe_metric(brier_score_loss, y_true, proba), n)


def calibrate_with_validation(raw_val: np.ndarray, y_val: pd.Series, raw_test: np.ndarray) -> np.ndarray:
    if y_val.nunique() < 2 or len(np.unique(np.round(raw_val, 6))) < 3:
        return raw_test
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_val, y_val)
    return np.asarray(calibrator.transform(raw_test), dtype=float)


def train_standard_models(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_state = int(config["project"]["random_state"])
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    availability_rows = [
        {"package": "catboost", "available": optional_module("catboost")},
        {"package": "lightgbm", "available": optional_module("lightgbm")},
        {"package": "interpret", "available": optional_module("interpret")},
    ]

    for window in [w["name"] for w in config["forecast_windows"]]:
        window_df = panel[panel["forecast_window"] == window].copy()
        train = window_df[window_df["split"] == "train"].copy()
        val = window_df[window_df["split"] == "validation"].copy()
        test = window_df[window_df["split"] == "test"].copy()
        feature_cols = get_feature_columns(window_df)
        preprocessor = build_preprocessor(feature_cols, window_df)

        for target in ["yield_t_ha", "yield_shortfall"]:
            y_test = test[target]
            baseline_pred = historical_mean_predictions(train, test, target)
            evaluate_regression(metrics_rows, window, target, "HistoricalMean", y_test, baseline_pred)
            for row, pred in zip(test.itertuples(index=False), baseline_pred):
                if target == "yield_t_ha":
                    prediction_rows.append(
                        {
                            "row_id": row.row_id,
                            "forecast_window": window,
                            "model": "HistoricalMean",
                            "prediction_type": "point",
                            "target": target,
                            "y_true": getattr(row, target),
                            "y_pred": float(pred),
                            "y_low": np.nan,
                            "y_high": np.nan,
                            "y_proba": np.nan,
                        }
                    )

            for model_name, estimator in regression_models(random_state).items():
                pipeline = Pipeline(
                    steps=[("preprocess", clone(preprocessor)), ("model", clone(estimator))]
                )
                pipeline.fit(train[feature_cols], train[target])
                pred = pipeline.predict(test[feature_cols])
                evaluate_regression(metrics_rows, window, target, model_name, y_test, pred)
                if target == "yield_t_ha":
                    for row, value in zip(test.itertuples(index=False), pred):
                        prediction_rows.append(
                            {
                                "row_id": row.row_id,
                                "forecast_window": window,
                                "model": model_name,
                                "prediction_type": "point",
                                "target": target,
                                "y_true": getattr(row, target),
                                "y_pred": float(value),
                                "y_low": np.nan,
                                "y_high": np.nan,
                                "y_proba": np.nan,
                            }
                        )

        y_test_class = test["low_yield_risk"]
        baseline_proba = historical_risk_predictions(train, test)
        evaluate_classification(metrics_rows, window, "HistoricalRiskRate", y_test_class, baseline_proba)
        for row, proba in zip(test.itertuples(index=False), baseline_proba):
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "forecast_window": window,
                    "model": "HistoricalRiskRate",
                    "prediction_type": "classification",
                    "target": "low_yield_risk",
                    "y_true": row.low_yield_risk,
                    "y_pred": float(proba >= 0.5),
                    "y_low": np.nan,
                    "y_high": np.nan,
                    "y_proba": float(proba),
                }
            )

        for model_name, estimator in classification_models(random_state).items():
            pipeline = Pipeline(
                steps=[("preprocess", clone(preprocessor)), ("model", clone(estimator))]
            )
            pipeline.fit(train[feature_cols], train["low_yield_risk"])
            raw_val = pipeline.predict_proba(val[feature_cols])[:, 1]
            raw_test = pipeline.predict_proba(test[feature_cols])[:, 1]
            proba = calibrate_with_validation(raw_val, val["low_yield_risk"], raw_test)
            evaluate_classification(metrics_rows, window, model_name, y_test_class, proba)
            for row, value in zip(test.itertuples(index=False), proba):
                prediction_rows.append(
                    {
                        "row_id": row.row_id,
                        "forecast_window": window,
                        "model": model_name,
                        "prediction_type": "classification",
                        "target": "low_yield_risk",
                        "y_true": row.low_yield_risk,
                        "y_pred": float(value >= 0.5),
                        "y_low": np.nan,
                        "y_high": np.nan,
                        "y_proba": float(value),
                    }
                )

        add_quantile_and_conformal(
            metrics_rows,
            prediction_rows,
            window,
            train,
            val,
            test,
            feature_cols,
            preprocessor,
            config,
        )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(availability_rows),
    )


def add_quantile_and_conformal(
    metrics_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    window: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    preprocessor: ColumnTransformer,
    config: dict,
) -> None:
    random_state = int(config["project"]["random_state"])
    quantile_preds = {}
    for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        model = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=alpha,
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.04,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(train[feature_cols], train["yield_t_ha"])
        quantile_preds[label] = model.predict(test[feature_cols])
        add_metric(
            metrics_rows,
            window,
            "yield_t_ha",
            "probabilistic",
            "SklearnQuantileGBR",
            f"Pinball_{label}",
            mean_pinball_loss(test["yield_t_ha"], quantile_preds[label], alpha=alpha),
            len(test),
        )

    p10 = np.minimum(quantile_preds["p10"], quantile_preds["p90"])
    p90 = np.maximum(quantile_preds["p10"], quantile_preds["p90"])
    p50 = quantile_preds["p50"]
    coverage = ((test["yield_t_ha"].to_numpy() >= p10) & (test["yield_t_ha"].to_numpy() <= p90)).mean()
    width = np.mean(p90 - p10)
    add_metric(metrics_rows, window, "yield_t_ha", "probabilistic", "SklearnQuantileGBR", "IntervalCoverage_P10_P90", coverage, len(test))
    add_metric(metrics_rows, window, "yield_t_ha", "probabilistic", "SklearnQuantileGBR", "IntervalWidth_P10_P90", width, len(test))

    for row, lo, mid, hi in zip(test.itertuples(index=False), p10, p50, p90):
        prediction_rows.append(
            {
                "row_id": row.row_id,
                "forecast_window": window,
                "model": "SklearnQuantileGBR",
                "prediction_type": "interval",
                "target": "yield_t_ha",
                "y_true": row.yield_t_ha,
                "y_pred": float(mid),
                "y_low": float(lo),
                "y_high": float(hi),
                "y_proba": np.nan,
            }
        )

    point_model = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )
    point_model.fit(train[feature_cols], train["yield_t_ha"])
    val_pred = point_model.predict(val[feature_cols])
    residual = np.abs(val["yield_t_ha"].to_numpy() - val_pred)
    alpha = float(config["thresholds"]["conformal_alpha"])
    qhat = float(np.quantile(residual, min(1.0, (1.0 - alpha) * (len(residual) + 1) / len(residual)), method="higher"))
    test_pred = point_model.predict(test[feature_cols])
    low = test_pred - qhat
    high = test_pred + qhat
    coverage = ((test["yield_t_ha"].to_numpy() >= low) & (test["yield_t_ha"].to_numpy() <= high)).mean()
    width = np.mean(high - low)
    add_metric(metrics_rows, window, "yield_t_ha", "uncertainty", "ConformalHGB", "ConformalCoverage", coverage, len(test))
    add_metric(metrics_rows, window, "yield_t_ha", "uncertainty", "ConformalHGB", "ConformalWidth", width, len(test))
    add_metric(metrics_rows, window, "yield_t_ha", "uncertainty", "ConformalHGB", "ConformalQhat", qhat, len(test))

    for row, pred, lo, hi in zip(test.itertuples(index=False), test_pred, low, high):
        prediction_rows.append(
            {
                "row_id": row.row_id,
                "forecast_window": window,
                "model": "ConformalHGB",
                "prediction_type": "interval",
                "target": "yield_t_ha",
                "y_true": row.yield_t_ha,
                "y_pred": float(pred),
                "y_low": float(lo),
                "y_high": float(hi),
                "y_proba": np.nan,
            }
        )


def rolling_origin(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    random_state = int(config["project"]["random_state"])
    rows = []
    for fold in config["rolling_origin_folds"]:
        for window in [w["name"] for w in config["forecast_windows"]]:
            window_df = panel[panel["forecast_window"] == window].copy()
            train = window_df[window_df["year_start"] <= int(fold["train_end"])].copy()
            test = window_df[
                (window_df["year_start"] >= int(fold["test_start"]))
                & (window_df["year_start"] <= int(fold["test_end"]))
            ].copy()
            feature_cols = get_feature_columns(window_df)
            preprocessor = build_preprocessor(feature_cols, window_df)
            for model_name, estimator in {
                "Ridge": Ridge(alpha=1.0),
                "HistGradientBoosting": HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.05,
                    random_state=random_state,
                ),
            }.items():
                pipeline = Pipeline(
                    steps=[("preprocess", clone(preprocessor)), ("model", clone(estimator))]
                )
                pipeline.fit(train[feature_cols], train["yield_t_ha"])
                pred = pipeline.predict(test[feature_cols])
                rows.append(
                    {
                        "fold": fold["fold"],
                        "forecast_window": window,
                        "model": model_name,
                        "train_end": fold["train_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        "n_test": len(test),
                        "MAE": mean_absolute_error(test["yield_t_ha"], pred),
                        "RMSE": rmse(test["yield_t_ha"], pred),
                        "R2": finite_or_nan(safe_metric(r2_score, test["yield_t_ha"], pred)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_lead_time(metrics: pd.DataFrame) -> str:
    hgb = metrics[
        (metrics["target"] == "yield_t_ha")
        & (metrics["task"] == "regression")
        & (metrics["model"] == "HistGradientBoosting")
        & (metrics["metric"].isin(["RMSE", "R2"]))
    ].copy()
    pivot = hgb.pivot_table(index="forecast_window", columns="metric", values="value", aggfunc="mean")
    if pivot.empty:
        return "No HistGradientBoosting lead-time metrics were available."
    best_rmse_window = pivot["RMSE"].idxmin() if "RMSE" in pivot else "NA"
    best_r2_window = pivot["R2"].idxmax() if "R2" in pivot else "NA"
    table = markdown_table(pivot.reset_index())
    return f"""
# Lead-Time Summary

HistGradientBoosting is used as the default sklearn performance model in this environment.

{table}

- Lowest RMSE window: `{best_rmse_window}`
- Highest R2 window: `{best_r2_window}`

Interpretation note: earlier windows preserve more lead time; later windows should be treated as more information-rich near-harvest benchmarks.
"""


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)

    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel.csv")
    metrics, predictions, availability = train_standard_models(panel, config)
    rolling = rolling_origin(panel, config)

    tables_dir = configured_path(config, "tables_dir")
    reports_dir = configured_path(config, "reports_dir")
    metrics.to_csv(tables_dir / "model_metrics_by_window.csv", index=False)
    predictions.to_csv(tables_dir / "test_predictions.csv", index=False)
    availability.to_csv(tables_dir / "model_availability.csv", index=False)
    rolling.to_csv(tables_dir / "rolling_origin_sensitivity.csv", index=False)
    write_markdown(reports_dir / "lead_time_summary.md", summarize_lead_time(metrics))

    availability_notes = []
    for row in availability.itertuples(index=False):
        availability_notes.append(f"- {row.package}: {'available' if row.available else 'not installed; skipped'}")
    write_markdown(
        reports_dir / "model_training_report.md",
        f"""
# Model Training Report

- Metrics: `outputs/tables/model_metrics_by_window.csv`
- Test predictions: `outputs/tables/test_predictions.csv`
- Rolling-origin sensitivity: `outputs/tables/rolling_origin_sensitivity.csv`

## Optional Package Availability

{chr(10).join(availability_notes)}

The pipeline keeps CatBoost, LightGBM, and EBM/interpret support optional. When those packages are installed, the same script adds them to the model comparison automatically.
""",
    )
    print(f"Wrote {tables_dir / 'model_metrics_by_window.csv'} ({len(metrics)} rows)")
    print(f"Wrote {tables_dir / 'test_predictions.csv'} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
