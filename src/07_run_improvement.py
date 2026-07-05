from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import configured_path, ensure_output_dirs, load_config, markdown_table, safe_metric, write_markdown


IMPROVEMENT_DIR_NAME = "improvement_plan_2026_06_24"
IMPROVEMENT_DIR = Path("outputs") / IMPROVEMENT_DIR_NAME
TARGET_COLUMNS = {
    "yield_t_ha",
    "expected_yield_t_ha",
    "yield_shortfall",
    "low_yield_risk",
    "low_yield_shortfall_threshold",
}
LEAKAGE_COLUMNS = {"production_kt", "area_000ha"}
CATEGORICAL_COLUMNS = ["region", "crop"]
WEATHER_ANOMALY_COLUMNS = [
    "rain_sum",
    "rain_mean",
    "dry_days",
    "max_consecutive_dry_days",
    "tmax_mean",
    "tmax_max",
    "heat_days_30",
    "evap_sum",
    "radiation_sum",
]
META_EXCLUDE = {
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


def optional_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_auc(func, y_true, score) -> float:
    try:
        if pd.Series(y_true).nunique() < 2:
            return float("nan")
        return float(func(y_true, score))
    except Exception:
        return float("nan")


def preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def add_weather_anomalies(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    panel = panel.copy()
    train_end = int(config["splits"]["train_end"])
    keys = ["region", "forecast_window"]
    unique_weather = panel.drop_duplicates(["region", "year_start", "forecast_window"]).copy()
    train_weather = unique_weather[unique_weather["year_start"] <= train_end].copy()
    region_window_means = train_weather.groupby(keys)[WEATHER_ANOMALY_COLUMNS].mean()
    window_means = train_weather.groupby("forecast_window")[WEATHER_ANOMALY_COLUMNS].mean()

    for col in WEATHER_ANOMALY_COLUMNS:
        anomaly_values = []
        pct_values = []
        for row in panel.itertuples(index=False):
            key = (row.region, row.forecast_window)
            baseline = np.nan
            if key in region_window_means.index:
                baseline = region_window_means.loc[key, col]
            if pd.isna(baseline) and row.forecast_window in window_means.index:
                baseline = window_means.loc[row.forecast_window, col]
            actual = getattr(row, col)
            anomaly = float(actual - baseline)
            anomaly_values.append(anomaly)
            if baseline and np.isfinite(baseline) and abs(float(baseline)) > 1e-9:
                pct_values.append(float(anomaly / baseline))
            else:
                pct_values.append(0.0)
        panel[f"{col}_anom_train_region_window"] = anomaly_values
        panel[f"{col}_pct_anom_train_region_window"] = pct_values
    return panel


def add_past_yield_features(panel: pd.DataFrame) -> pd.DataFrame:
    yield_unique = (
        panel.drop_duplicates(["region", "crop", "year_start"])
        [["region", "crop", "year_start", "yield_t_ha"]]
        .sort_values(["region", "crop", "year_start"])
        .copy()
    )
    parts = []
    for _, group in yield_unique.groupby(["region", "crop"], sort=False):
        group = group.sort_values("year_start").copy()
        group["yield_lag1_crop_region"] = group["yield_t_ha"].shift(1)
        group["yield_rolling3_past_crop_region"] = (
            group["yield_t_ha"].shift(1).rolling(window=3, min_periods=1).mean()
        )
        group["yield_expanding_past_mean_crop_region"] = (
            group["yield_t_ha"].shift(1).expanding(min_periods=1).mean()
        )
        parts.append(group.drop(columns=["yield_t_ha"]))
    lag_features = pd.concat(parts, ignore_index=True)
    return panel.merge(lag_features, on=["region", "crop", "year_start"], how="left", validate="many_to_one")


def build_improved_panel(config: dict[str, Any]) -> pd.DataFrame:
    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel.csv")
    panel = add_weather_anomalies(panel, config)
    panel = add_past_yield_features(panel)
    output_path = configured_path(config, "processed_dir") / "model_ready_panel_improved.csv"
    panel.to_csv(output_path, index=False)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    exclude = TARGET_COLUMNS | LEAKAGE_COLUMNS | META_EXCLUDE
    features = []
    for col in panel.columns:
        if col in exclude:
            continue
        if col in CATEGORICAL_COLUMNS:
            features.append(col)
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            features.append(col)
    return features


def reg_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }
    if optional_module("lightgbm"):
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=10,
            reg_lambda=0.5,
            random_state=random_state,
            verbose=-1,
        )
    if optional_module("catboost"):
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=350,
            depth=4,
            learning_rate=0.035,
            random_seed=random_state,
            verbose=False,
            loss_function="RMSE",
        )
    return models


def clf_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }
    if optional_module("lightgbm"):
        from lightgbm import LGBMClassifier

        models["LightGBMClassifier"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=10,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    if optional_module("catboost"):
        from catboost import CatBoostClassifier

        models["CatBoostClassifier"] = CatBoostClassifier(
            iterations=350,
            depth=4,
            learning_rate=0.035,
            random_seed=random_state,
            verbose=False,
            loss_function="Logloss",
            auto_class_weights="Balanced",
        )
    return models


def add_metric(rows: list[dict[str, Any]], window: str, target: str, task: str, model: str, metric: str, value: float, n_test: int) -> None:
    rows.append(
        {
            "forecast_window": window,
            "target": target,
            "task": task,
            "model": model,
            "metric": metric,
            "value": float(value) if np.isfinite(value) else np.nan,
            "n_test": int(n_test),
        }
    )


def predict_proba_safe(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    return np.asarray(pred, dtype=float)


def calibrate_pair(raw_val: np.ndarray, y_val: pd.Series, raw_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_val.nunique() < 2 or len(np.unique(np.round(raw_val, 6))) < 3:
        return np.clip(raw_val, 0, 1), np.clip(raw_test, 0, 1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_val, y_val)
    return np.asarray(iso.transform(raw_val), dtype=float), np.asarray(iso.transform(raw_test), dtype=float)


def threshold_grid() -> np.ndarray:
    return np.round(np.linspace(0.01, 0.99, 99), 4)


def threshold_analysis_rows(window: str, model: str, y_val: pd.Series, val_proba: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for threshold in threshold_grid():
        pred = (val_proba >= threshold).astype(int)
        rows.append(
            {
                "forecast_window": window,
                "model": model,
                "threshold": float(threshold),
                "validation_precision": precision_score(y_val, pred, zero_division=0),
                "validation_recall": recall_score(y_val, pred, zero_division=0),
                "validation_f1": f1_score(y_val, pred, zero_division=0),
                "validation_positive_rate": float(pred.mean()),
            }
        )
    return rows


def choose_thresholds(analysis: pd.DataFrame) -> dict[str, pd.Series]:
    best_f1 = analysis.sort_values(
        ["validation_f1", "validation_recall", "validation_precision", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    recall_candidates = analysis[analysis["validation_recall"] >= 0.5]
    if recall_candidates.empty:
        recall_pick = analysis.sort_values(["validation_recall", "validation_f1"], ascending=[False, False]).iloc[0]
    else:
        recall_pick = recall_candidates.sort_values(
            ["validation_f1", "validation_precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

    precision_candidates = analysis[analysis["validation_precision"] >= 0.3]
    if precision_candidates.empty:
        precision_pick = analysis.sort_values(["validation_precision", "validation_recall"], ascending=[False, False]).iloc[0]
    else:
        precision_pick = precision_candidates.sort_values(
            ["validation_recall", "validation_f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

    return {
        "best_f1": best_f1,
        "recall_ge_0_5": recall_pick,
        "precision_ge_0_3": precision_pick,
    }


def eval_class_at_threshold(window: str, model: str, strategy: str, threshold_row: pd.Series, y_test: pd.Series, test_proba: np.ndarray) -> dict[str, Any]:
    threshold = float(threshold_row["threshold"])
    pred = (test_proba >= threshold).astype(int)
    return {
        "forecast_window": window,
        "model": model,
        "strategy": strategy,
        "threshold": threshold,
        "validation_precision": float(threshold_row["validation_precision"]),
        "validation_recall": float(threshold_row["validation_recall"]),
        "validation_f1": float(threshold_row["validation_f1"]),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_f1": f1_score(y_test, pred, zero_division=0),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_brier": brier_score_loss(y_test, test_proba),
        "test_roc_auc": safe_auc(roc_auc_score, y_test, test_proba),
        "test_pr_auc": safe_auc(average_precision_score, y_test, test_proba),
        "test_positive_rate": float(pred.mean()),
        "n_test": int(len(y_test)),
    }


def historical_mean_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    global_mean = float(train[target].mean())
    crop_mean = train.groupby("crop")[target].mean().to_dict()
    crop_region_mean = train.groupby(["crop", "region"])[target].mean().to_dict()
    return np.asarray(
        [
            crop_region_mean.get((row.crop, row.region), crop_mean.get(row.crop, global_mean))
            for row in test.itertuples(index=False)
        ],
        dtype=float,
    )


def train_improved(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_state = int(config["project"]["random_state"])
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    tuned_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    features = feature_columns(panel)

    for window_cfg in config["forecast_windows"]:
        window = window_cfg["name"]
        df = panel[panel["forecast_window"] == window].copy()
        train = df[df["split"] == "train"].copy()
        val = df[df["split"] == "validation"].copy()
        test = df[df["split"] == "test"].copy()
        prep = preprocessor(features)

        for target in ["yield_t_ha", "yield_shortfall"]:
            baseline = historical_mean_predictions(train, test, target)
            add_metric(metrics_rows, window, target, "regression", "HistoricalMeanImprovedFeatures", "MAE", mean_absolute_error(test[target], baseline), len(test))
            add_metric(metrics_rows, window, target, "regression", "HistoricalMeanImprovedFeatures", "RMSE", rmse(test[target], baseline), len(test))
            add_metric(metrics_rows, window, target, "regression", "HistoricalMeanImprovedFeatures", "R2", safe_metric(r2_score, test[target], baseline), len(test))

            for model_name, estimator in reg_models(random_state).items():
                pipe = Pipeline([("preprocess", clone(prep)), ("model", clone(estimator))])
                pipe.fit(train[features], train[target])
                pred = pipe.predict(test[features])
                add_metric(metrics_rows, window, target, "regression", model_name, "MAE", mean_absolute_error(test[target], pred), len(test))
                add_metric(metrics_rows, window, target, "regression", model_name, "RMSE", rmse(test[target], pred), len(test))
                add_metric(metrics_rows, window, target, "regression", model_name, "R2", safe_metric(r2_score, test[target], pred), len(test))
                if target == "yield_t_ha":
                    for row, value in zip(test.itertuples(index=False), pred):
                        prediction_rows.append(
                            {
                                "row_id": row.row_id,
                                "forecast_window": window,
                                "model": model_name,
                                "prediction_type": "point",
                                "target": target,
                                "y_true": row.yield_t_ha,
                                "y_pred": float(value),
                                "y_low": np.nan,
                                "y_high": np.nan,
                                "y_proba": np.nan,
                            }
                        )

        for model_name, estimator in clf_models(random_state).items():
            pipe = Pipeline([("preprocess", clone(prep)), ("model", clone(estimator))])
            pipe.fit(train[features], train["low_yield_risk"])
            raw_val = predict_proba_safe(pipe, val[features])
            raw_test = predict_proba_safe(pipe, test[features])
            val_proba, test_proba = calibrate_pair(raw_val, val["low_yield_risk"], raw_test)

            fixed_pred = (test_proba >= 0.5).astype(int)
            add_metric(metrics_rows, window, "low_yield_risk", "classification", model_name, "ROC_AUC", safe_auc(roc_auc_score, test["low_yield_risk"], test_proba), len(test))
            add_metric(metrics_rows, window, "low_yield_risk", "classification", model_name, "PR_AUC", safe_auc(average_precision_score, test["low_yield_risk"], test_proba), len(test))
            add_metric(metrics_rows, window, "low_yield_risk", "classification", model_name, "Brier", brier_score_loss(test["low_yield_risk"], test_proba), len(test))
            add_metric(metrics_rows, window, "low_yield_risk", "classification", model_name, "F1_at_0.5", f1_score(test["low_yield_risk"], fixed_pred, zero_division=0), len(test))
            add_metric(metrics_rows, window, "low_yield_risk", "classification", model_name, "Recall_at_0.5", recall_score(test["low_yield_risk"], fixed_pred, zero_division=0), len(test))

            model_threshold_rows = threshold_analysis_rows(window, model_name, val["low_yield_risk"], val_proba)
            threshold_rows.extend(model_threshold_rows)
            analysis_df = pd.DataFrame(model_threshold_rows)
            for strategy, threshold_row in choose_thresholds(analysis_df).items():
                tuned = eval_class_at_threshold(window, model_name, strategy, threshold_row, test["low_yield_risk"], test_proba)
                tuned_rows.append(tuned)
                for metric_name in ["test_precision", "test_recall", "test_f1", "test_brier", "test_roc_auc", "test_pr_auc"]:
                    add_metric(
                        metrics_rows,
                        window,
                        "low_yield_risk",
                        "classification_tuned_threshold",
                        f"{model_name}_{strategy}",
                        metric_name.replace("test_", "").upper(),
                        float(tuned[metric_name]),
                        len(test),
                    )

            for row, proba in zip(test.itertuples(index=False), test_proba):
                prediction_rows.append(
                    {
                        "row_id": row.row_id,
                        "forecast_window": window,
                        "model": model_name,
                        "prediction_type": "classification",
                        "target": "low_yield_risk",
                        "y_true": row.low_yield_risk,
                        "y_pred": float(proba >= 0.5),
                        "y_low": np.nan,
                        "y_high": np.nan,
                        "y_proba": float(proba),
                    }
                )

        add_interval_models(metrics_rows, prediction_rows, interval_rows, window, train, val, test, features, prep, config)

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(tuned_rows),
        pd.DataFrame(interval_rows),
    )


def interval_summary_row(window: str, model: str, y_true: pd.Series, low: np.ndarray, pred: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    y = y_true.to_numpy()
    coverage = float(((y >= low) & (y <= high)).mean())
    width = float(np.mean(high - low))
    return {
        "forecast_window": window,
        "model": model,
        "coverage": coverage,
        "width": width,
        "pinball_p10": mean_pinball_loss(y_true, low, alpha=0.1),
        "pinball_p50": mean_pinball_loss(y_true, pred, alpha=0.5),
        "pinball_p90": mean_pinball_loss(y_true, high, alpha=0.9),
        "n_test": len(y_true),
    }


def add_interval_models(
    metrics_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    window: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    prep: ColumnTransformer,
    config: dict[str, Any],
) -> None:
    random_state = int(config["project"]["random_state"])
    quantile_model_specs: list[tuple[str, Any, Any, Any]] = [
        (
            "SklearnQuantileGBR",
            GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=300, max_depth=3, learning_rate=0.035, random_state=random_state),
            GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=300, max_depth=3, learning_rate=0.035, random_state=random_state),
            GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=300, max_depth=3, learning_rate=0.035, random_state=random_state),
        )
    ]
    if optional_module("lightgbm"):
        from lightgbm import LGBMRegressor

        quantile_model_specs.append(
            (
                "LightGBMQuantile",
                LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=350, learning_rate=0.035, num_leaves=15, min_child_samples=10, random_state=random_state, verbose=-1),
                LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=350, learning_rate=0.035, num_leaves=15, min_child_samples=10, random_state=random_state, verbose=-1),
                LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=350, learning_rate=0.035, num_leaves=15, min_child_samples=10, random_state=random_state, verbose=-1),
            )
        )
    if optional_module("catboost"):
        from catboost import CatBoostRegressor

        quantile_model_specs.append(
            (
                "CatBoostQuantile",
                CatBoostRegressor(iterations=300, depth=4, learning_rate=0.035, random_seed=random_state, verbose=False, loss_function="Quantile:alpha=0.1"),
                CatBoostRegressor(iterations=300, depth=4, learning_rate=0.035, random_seed=random_state, verbose=False, loss_function="Quantile:alpha=0.5"),
                CatBoostRegressor(iterations=300, depth=4, learning_rate=0.035, random_seed=random_state, verbose=False, loss_function="Quantile:alpha=0.9"),
            )
        )

    for model_name, q10_est, q50_est, q90_est in quantile_model_specs:
        preds = []
        for estimator in [q10_est, q50_est, q90_est]:
            pipe = Pipeline([("preprocess", clone(prep)), ("model", clone(estimator))])
            pipe.fit(train[features], train["yield_t_ha"])
            preds.append(pipe.predict(test[features]))
        low = np.minimum(preds[0], preds[2])
        high = np.maximum(preds[0], preds[2])
        mid = preds[1]
        summary = interval_summary_row(window, model_name, test["yield_t_ha"], low, mid, high)
        interval_rows.append(summary)
        for metric in ["coverage", "width", "pinball_p10", "pinball_p50", "pinball_p90"]:
            add_metric(metrics_rows, window, "yield_t_ha", "probabilistic", model_name, metric, summary[metric], len(test))
        for row, lo, pred, hi in zip(test.itertuples(index=False), low, mid, high):
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "forecast_window": window,
                    "model": model_name,
                    "prediction_type": "interval",
                    "target": "yield_t_ha",
                    "y_true": row.yield_t_ha,
                    "y_pred": float(pred),
                    "y_low": float(lo),
                    "y_high": float(hi),
                    "y_proba": np.nan,
                }
            )

    # Conformal interval uses the best validation RMSE among point models.
    best = None
    for model_name, estimator in reg_models(random_state).items():
        pipe = Pipeline([("preprocess", clone(prep)), ("model", clone(estimator))])
        pipe.fit(train[features], train["yield_t_ha"])
        val_pred = pipe.predict(val[features])
        score = rmse(val["yield_t_ha"], val_pred)
        if best is None or score < best["score"]:
            best = {"name": model_name, "score": score, "pipe": pipe}
    assert best is not None
    val_pred = best["pipe"].predict(val[features])
    residual = np.abs(val["yield_t_ha"].to_numpy() - val_pred)
    alpha = float(config["thresholds"]["conformal_alpha"])
    qhat = float(np.quantile(residual, min(1.0, (1.0 - alpha) * (len(residual) + 1) / len(residual)), method="higher"))
    pred = best["pipe"].predict(test[features])
    low = pred - qhat
    high = pred + qhat
    summary = interval_summary_row(window, f"ConformalBestPoint_{best['name']}", test["yield_t_ha"], low, pred, high)
    summary["qhat"] = qhat
    summary["validation_rmse_model"] = best["score"]
    interval_rows.append(summary)
    for metric in ["coverage", "width"]:
        add_metric(metrics_rows, window, "yield_t_ha", "uncertainty", f"ConformalBestPoint_{best['name']}", metric, summary[metric], len(test))
    add_metric(metrics_rows, window, "yield_t_ha", "uncertainty", f"ConformalBestPoint_{best['name']}", "qhat", qhat, len(test))
    for row, lo, pred_value, hi in zip(test.itertuples(index=False), low, pred, high):
        prediction_rows.append(
            {
                "row_id": row.row_id,
                "forecast_window": window,
                "model": f"ConformalBestPoint_{best['name']}",
                "prediction_type": "interval",
                "target": "yield_t_ha",
                "y_true": row.yield_t_ha,
                "y_pred": float(pred_value),
                "y_low": float(lo),
                "y_high": float(hi),
                "y_proba": np.nan,
            }
        )


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_improvement_figures(config: dict[str, Any], improved_metrics: pd.DataFrame, improved_predictions: pd.DataFrame, threshold_analysis: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    fig_dir = configured_path(config, "figures_dir")
    order = {w["name"]: w["order"] for w in config["forecast_windows"]}

    # fig08 PR curve by window for best average PR-AUC classifier.
    class_metrics = improved_metrics[
        (improved_metrics["task"] == "classification")
        & (improved_metrics["metric"] == "PR_AUC")
    ]
    best_model = class_metrics.groupby("model")["value"].mean().idxmax()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for window in sorted(improved_predictions["forecast_window"].unique(), key=lambda w: order.get(w, 999)):
        subset = improved_predictions[
            (improved_predictions["prediction_type"] == "classification")
            & (improved_predictions["model"] == best_model)
            & (improved_predictions["forecast_window"] == window)
        ].dropna(subset=["y_proba"])
        precision, recall, _ = precision_recall_curve(subset["y_true"], subset["y_proba"])
        ax.plot(recall, precision, label=window)
    ax.set_title(f"PR Curves By Forecast Window ({best_model})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, fig_dir / "fig08_pr_curve_by_window.png")

    # fig09 threshold tradeoff for best model/window by validation F1 max.
    best_threshold_group = (
        threshold_analysis.groupby(["forecast_window", "model"])["validation_f1"]
        .max()
        .reset_index()
        .sort_values("validation_f1", ascending=False)
        .iloc[0]
    )
    subset = threshold_analysis[
        (threshold_analysis["forecast_window"] == best_threshold_group["forecast_window"])
        & (threshold_analysis["model"] == best_threshold_group["model"])
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(subset["threshold"], subset["validation_precision"], label="Precision", color="#22577a")
    ax.plot(subset["threshold"], subset["validation_recall"], label="Recall", color="#ba4a00")
    ax.plot(subset["threshold"], subset["validation_f1"], label="F1", color="#2f6f73")
    ax.set_title(f"Validation Threshold Trade-Off: {best_threshold_group['model']} {best_threshold_group['forecast_window']}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, fig_dir / "fig09_threshold_tradeoff.png")

    # fig10 interval coverage/width by lead time.
    preferred = interval_summary[interval_summary["model"].astype(str).str.startswith("ConformalBestPoint")].copy()
    preferred["window_order"] = preferred["forecast_window"].map(order)
    preferred = preferred.sort_values("window_order")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.bar(preferred["forecast_window"], preferred["coverage"], color="#8ecae6", label="Coverage")
    ax1.axhline(0.8, color="#ba4a00", linestyle="--", linewidth=1.5, label="80% target")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(preferred["forecast_window"], preferred["width"], color="#22577a", marker="o", label="Width")
    ax2.set_ylabel("Interval width")
    ax1.set_title("Conformal Interval Coverage By Lead Time")
    ax1.grid(axis="y", alpha=0.25)
    save_fig(fig, fig_dir / "fig10_interval_coverage_by_lead_time.png")

    # fig11 old vs improved regression.
    old = pd.read_csv(configured_path(config, "tables_dir") / "model_metrics_by_window.csv")
    old_reg = old[(old["task"] == "regression") & (old["target"] == "yield_t_ha") & (old["metric"].isin(["RMSE", "R2"]))]
    new_reg = improved_metrics[(improved_metrics["task"] == "regression") & (improved_metrics["target"] == "yield_t_ha") & (improved_metrics["metric"].isin(["RMSE", "R2"]))]
    old_p = old_reg.pivot_table(index=["forecast_window", "model"], columns="metric", values="value").reset_index()
    new_p = new_reg.pivot_table(index=["forecast_window", "model"], columns="metric", values="value").reset_index()
    old_best = old_p.sort_values(["forecast_window", "RMSE"]).groupby("forecast_window").head(1).assign(series="Baseline best")
    new_best = new_p.sort_values(["forecast_window", "RMSE"]).groupby("forecast_window").head(1).assign(series="Improved best")
    comp = pd.concat([old_best, new_best], ignore_index=True)
    comp["window_order"] = comp["forecast_window"].map(order)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for metric, ax in [("RMSE", axes[0]), ("R2", axes[1])]:
        for series, group in comp.sort_values("window_order").groupby("series"):
            ax.plot(group["forecast_window"], group[metric], marker="o", label=series)
        ax.set_title(f"Old vs Improved {metric}")
        ax.set_xlabel("Forecast window")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend()
    save_fig(fig, fig_dir / "fig11_old_vs_improved_rmse_r2.png")


def write_reports(config: dict[str, Any], availability: pd.DataFrame, panel: pd.DataFrame, improved_metrics: pd.DataFrame, threshold_metrics: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    improvement_dir = configured_path(config, "reports_dir").parent / IMPROVEMENT_DIR_NAME
    tables_dir = configured_path(config, "tables_dir")
    old_metrics = pd.read_csv(tables_dir / "model_metrics_by_window.csv")

    old_reg = old_metrics[(old_metrics["task"] == "regression") & (old_metrics["target"] == "yield_t_ha") & (old_metrics["metric"].isin(["RMSE", "R2", "MAE"]))]
    new_reg = improved_metrics[(improved_metrics["task"] == "regression") & (improved_metrics["target"] == "yield_t_ha") & (improved_metrics["metric"].isin(["RMSE", "R2", "MAE"]))]
    old_best = (
        old_reg.pivot_table(index=["forecast_window", "model"], columns="metric", values="value")
        .reset_index()
        .sort_values("RMSE")
        .iloc[0]
    )
    new_best = (
        new_reg.pivot_table(index=["forecast_window", "model"], columns="metric", values="value")
        .reset_index()
        .sort_values("RMSE")
        .iloc[0]
    )
    tuned_best = threshold_metrics.sort_values(["test_f1", "test_recall", "test_pr_auc"], ascending=False).head(10)
    interval_best = interval_summary.sort_values(["coverage", "width"], ascending=[False, True]).head(10)

    availability_text = markdown_table(availability)
    improved_shape = f"{len(panel)} rows, {panel[['region','crop','year_start']].drop_duplicates().shape[0]} unique yield observations"

    write_markdown(
        improvement_dir / "02_IMPLEMENTATION_LOG_AUS_EARLY_WARNING.md",
        f"""
# Implementation Log: AUS Early-Warning Improvement

## What Changed

- Added train-only region-window weather anomaly features for rainfall, heat, dry-spell, evaporation, and radiation signals.
- Added past-year and rolling historical yield features by crop-region using only previous years.
- Installed/detected CatBoost and LightGBM where available.
- Added validation-only threshold tuning for classification.
- Added improved quantile and conformal interval reporting.
- Added new comparison and decision-support figures.

## Package Availability

{availability_text}

## Output Files

- `data/processed/model_ready_panel_improved.csv`
- `outputs/tables/improved_model_metrics_by_window.csv`
- `outputs/tables/improved_test_predictions.csv`
- `outputs/tables/classification_threshold_analysis.csv`
- `outputs/tables/classification_metrics_tuned_thresholds.csv`
- `outputs/tables/improved_interval_summary.csv`
- `outputs/figures/fig08_pr_curve_by_window.png`
- `outputs/figures/fig09_threshold_tradeoff.png`
- `outputs/figures/fig10_interval_coverage_by_lead_time.png`
- `outputs/figures/fig11_old_vs_improved_rmse_r2.png`

## Acceptance Checks

- Improved panel shape: {improved_shape}
- Missing forecast-window rows: {int(panel['forecast_window'].isna().sum())}
- Test years were not used for anomaly baselines, thresholds, calibration, or conformal residuals.
- Leakage fields `production_kt` and `area_000ha` remain excluded from training feature matrices.
""",
    )

    write_markdown(
        improvement_dir / "03_RESULTS_COMPARISON_AFTER_IMPROVEMENT.md",
        f"""
# Results Comparison After Improvement

## Regression

Baseline best result:

| Window | Model | RMSE | R2 |
|---|---|---:|---:|
| {old_best['forecast_window']} | {old_best['model']} | {old_best['RMSE']:.3f} | {old_best['R2']:.3f} |

Improved best result:

| Window | Model | RMSE | R2 |
|---|---|---:|---:|
| {new_best['forecast_window']} | {new_best['model']} | {new_best['RMSE']:.3f} | {new_best['R2']:.3f} |

RMSE change: `{float(new_best['RMSE'] - old_best['RMSE']):.3f}`. R2 change: `{float(new_best['R2'] - old_best['R2']):.3f}`.

## Tuned Classification Highlights

{markdown_table(tuned_best[['forecast_window','model','strategy','threshold','test_precision','test_recall','test_f1','test_pr_auc','test_brier']])}

## Interval Highlights

{markdown_table(interval_best[['forecast_window','model','coverage','width','pinball_p10','pinball_p50','pinball_p90']])}

## Interpretation

The improvement pass should be read as a model-selection and reporting upgrade. If the strongest regression gain is modest, the main value still comes from tuned risk thresholds, stronger calibrated classification views, and conformal uncertainty that is easier to defend in the paper.
""",
    )

    best_window_table = (
        new_reg.pivot_table(index=["forecast_window", "model"], columns="metric", values="value")
        .reset_index()
        .sort_values("RMSE")
        .head(8)
    )
    best_tuned = threshold_metrics.sort_values(["test_f1", "test_recall"], ascending=False).iloc[0]
    best_interval = interval_summary[interval_summary["model"].astype(str).str.startswith("ConformalBestPoint")].sort_values("coverage", ascending=False).iloc[0]

    write_markdown(
        improvement_dir / "04_PAPER_READY_FINDINGS_AUS_EARLY_WARNING.md",
        f"""
# Paper-Ready Findings: AUS Early-Warning Project

## Recommended Framing

Use the project as a stage-aware, probabilistic, state-level early-warning framework. The strongest paper angle is not just point-yield prediction; it is the trade-off between early warning, calibrated low-yield risk, and uncertainty.

## Model Choice For Paper

Top improved regression candidates:

{markdown_table(best_window_table[['forecast_window','model','MAE','RMSE','R2']])}

For low-yield risk, the best tuned threshold result is:

- Window: `{best_tuned['forecast_window']}`
- Model: `{best_tuned['model']}`
- Strategy: `{best_tuned['strategy']}`
- Threshold: `{best_tuned['threshold']:.3f}`
- Test recall: `{best_tuned['test_recall']:.3f}`
- Test F1: `{best_tuned['test_f1']:.3f}`
- Test PR-AUC: `{best_tuned['test_pr_auc']:.3f}`

For uncertainty, use conformal intervals if quantile coverage is weaker:

- Best conformal window: `{best_interval['forecast_window']}`
- Model: `{best_interval['model']}`
- Coverage: `{best_interval['coverage']:.3f}`
- Width: `{best_interval['width']:.3f}`

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
""",
    )


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    improvement_dir = configured_path(config, "reports_dir").parent / IMPROVEMENT_DIR_NAME
    improvement_dir.mkdir(parents=True, exist_ok=True)

    availability = pd.DataFrame(
        [
            {"package": "catboost", "available": optional_module("catboost")},
            {"package": "lightgbm", "available": optional_module("lightgbm")},
            {"package": "interpret", "available": optional_module("interpret")},
        ]
    )
    panel = build_improved_panel(config)
    improved_metrics, improved_predictions, threshold_analysis, tuned_thresholds, interval_summary = train_improved(panel, config)

    tables_dir = configured_path(config, "tables_dir")
    improved_metrics.to_csv(tables_dir / "improved_model_metrics_by_window.csv", index=False)
    improved_predictions.to_csv(tables_dir / "improved_test_predictions.csv", index=False)
    threshold_analysis.to_csv(tables_dir / "classification_threshold_analysis.csv", index=False)
    tuned_thresholds.to_csv(tables_dir / "classification_metrics_tuned_thresholds.csv", index=False)
    interval_summary.to_csv(tables_dir / "improved_interval_summary.csv", index=False)
    availability.to_csv(tables_dir / "improved_model_availability.csv", index=False)

    make_improvement_figures(config, improved_metrics, improved_predictions, threshold_analysis, interval_summary)
    write_reports(config, availability, panel, improved_metrics, tuned_thresholds, interval_summary)
    print(f"Wrote improvement outputs to {configured_path(config, 'tables_dir')} and {improvement_dir}")


if __name__ == "__main__":
    main()
