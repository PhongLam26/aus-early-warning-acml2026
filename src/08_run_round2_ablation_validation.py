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
from matplotlib.colors import TwoSlopeNorm

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import configured_path, ensure_output_dirs, load_config, markdown_table, write_markdown


ROUND2_DIR_NAME = "improvement_round2_ablation_validation_2026_06_24"
ROUND2_IMPORTANCE_WINDOWS = ["May-Jun", "May-Sep", "May-Oct"]
FEATURE_LABELS = {
    "identity_time_only": "Identity+Time",
    "weather_stage_only": "Weather",
    "weather_plus_anomaly": "Weather+Dev",
    "weather_plus_soil": "Weather+Soil",
    "weather_anomaly_soil_no_lag": "No-yield-history weather-soil",
    "lag_yield_only": "Yield history",
    "full_operational": "Operational with yield history",
}
PROTOCOL_LABELS = {
    "time_split": "Held-out test years",
    "rolling_origin": "Rolling-origin",
    "leave_one_region_out": "Leave-one-region-out",
    "leave_one_crop_out": "Leave-one-crop-out",
}
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
CATEGORICAL_TIME_COLUMNS = ["region", "crop", "year_start"]
BASE_WEATHER_COLUMNS = [
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
]


def optional_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_auc(func, y_true, score, default: float = 0.5) -> float:
    try:
        if pd.Series(y_true).nunique() < 2:
            return default
        value = float(func(y_true, score))
        return value if np.isfinite(value) else default
    except Exception:
        return default


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in ["region", "crop"] if c in features]
    numeric = [c for c in features if c not in categorical]
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


def column_groups(panel: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(panel.columns)
    groups = {
        "categorical_time": [c for c in CATEGORICAL_TIME_COLUMNS if c in cols],
        "weather_stage": [c for c in BASE_WEATHER_COLUMNS if c in cols],
        "weather_anomaly": [c for c in cols if "anom_train_region_window" in c],
        "soil": [c for c in cols if c.startswith("soil_")],
        "lag_yield": [
            c
            for c in cols
            if c.startswith("yield_lag")
            or "rolling" in c
            or "expanding" in c
        ],
    }
    groups["rainfall"] = [c for c in groups["weather_stage"] if "rain" in c or "dry" in c]
    groups["heat_cold"] = [
        c
        for c in groups["weather_stage"]
        if "tmax" in c or "tmin" in c or "heat" in c or "frost" in c or "cold" in c
    ]
    groups["energy_dryness"] = [
        c
        for c in groups["weather_stage"]
        if "radiation" in c or "evap" in c or c.startswith("vp_")
    ]
    groups["compound_stress"] = [
        c
        for c in groups["weather_stage"]
        if "hot_dry" in c or "high_evap" in c or "heat_and" in c
    ]
    return groups


def feature_set_columns(panel: pd.DataFrame, feature_set: str) -> list[str]:
    groups = column_groups(panel)
    mapping = {
        "identity_time_only": groups["categorical_time"],
        "weather_stage_only": groups["categorical_time"] + groups["weather_stage"],
        "weather_plus_anomaly": groups["categorical_time"] + groups["weather_stage"] + groups["weather_anomaly"],
        "weather_plus_soil": groups["categorical_time"] + groups["weather_stage"] + groups["soil"],
        "weather_anomaly_soil_no_lag": groups["categorical_time"]
        + groups["weather_stage"]
        + groups["weather_anomaly"]
        + groups["soil"],
        "lag_yield_only": groups["categorical_time"] + groups["lag_yield"],
        "full_operational": groups["categorical_time"]
        + groups["weather_stage"]
        + groups["weather_anomaly"]
        + groups["soil"]
        + groups["lag_yield"],
    }
    features = mapping[feature_set]
    blocked = TARGET_COLUMNS | LEAKAGE_COLUMNS | META_COLUMNS
    return sorted(dict.fromkeys([c for c in features if c in panel.columns and c not in blocked]))


def validate_feature_sets(panel: pd.DataFrame, feature_sets: list[str]) -> pd.DataFrame:
    rows = []
    for feature_set in feature_sets:
        features = feature_set_columns(panel, feature_set)
        rows.append(
            {
                "feature_set": feature_set,
                "n_features": len(features),
                "uses_production_kt": "production_kt" in features,
                "uses_area_000ha": "area_000ha" in features,
                "uses_lag_feature": any(("yield_lag" in c or "rolling" in c or "expanding" in c) for c in features),
            }
        )
    return pd.DataFrame(rows)


def regression_models(config: dict[str, Any], stress: bool = False) -> dict[str, Any]:
    random_state = int(config["project"]["random_state"])
    names = config["round2_primary_models"]["stress_regression" if stress else "regression"]
    models: dict[str, Any] = {}
    if "Ridge" in names:
        models["Ridge"] = Ridge(alpha=1.0)
    if "ElasticNet" in names:
        models["ElasticNet"] = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=random_state)
    if "LightGBM" in names and optional_module("lightgbm"):
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=10,
            reg_lambda=0.5,
            random_state=random_state,
            verbose=-1,
        )
    if "CatBoost" in names and optional_module("catboost"):
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=220,
            depth=4,
            learning_rate=0.04,
            random_seed=random_state,
            verbose=False,
            loss_function="RMSE",
        )
    return models


def classification_models(config: dict[str, Any], stress: bool = False) -> dict[str, Any]:
    random_state = int(config["project"]["random_state"])
    names = config["round2_primary_models"]["stress_classification" if stress else "classification"]
    models: dict[str, Any] = {}
    if "LogisticRegression" in names:
        models["LogisticRegression"] = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if "LightGBMClassifier" in names and optional_module("lightgbm"):
        from lightgbm import LGBMClassifier

        models["LightGBMClassifier"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=10,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    if "CatBoostClassifier" in names and optional_module("catboost"):
        from catboost import CatBoostClassifier

        models["CatBoostClassifier"] = CatBoostClassifier(
            iterations=220,
            depth=4,
            learning_rate=0.04,
            random_seed=random_state,
            verbose=False,
            loss_function="Logloss",
            auto_class_weights="Balanced",
        )
    return models


def fit_predict_regression(train: pd.DataFrame, test: pd.DataFrame, features: list[str], estimator: Any) -> np.ndarray:
    pipe = Pipeline([("preprocess", make_preprocessor(features)), ("model", clone(estimator))])
    pipe.fit(train[features], train["yield_t_ha"])
    return pipe.predict(test[features])


def classify_with_threshold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    estimator: Any,
) -> tuple[np.ndarray, float]:
    pipe = Pipeline([("preprocess", make_preprocessor(features)), ("model", clone(estimator))])
    pipe.fit(train[features], train["low_yield_risk"])
    val_proba = pipe.predict_proba(validation[features])[:, 1]
    test_proba = pipe.predict_proba(test[features])[:, 1]
    precision, recall, thresholds = precision_recall_curve(validation["low_yield_risk"], val_proba)
    if len(thresholds) == 0:
        return test_proba, 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return test_proba, float(thresholds[idx])


def regression_metric_rows(context: dict[str, Any], y_true: pd.Series, y_pred: np.ndarray) -> list[dict[str, Any]]:
    base = dict(context)
    return [
        {**base, "task": "regression", "metric": "MAE", "value": mean_absolute_error(y_true, y_pred), "n_test": len(y_true)},
        {**base, "task": "regression", "metric": "RMSE", "value": rmse(y_true, y_pred), "n_test": len(y_true)},
        {**base, "task": "regression", "metric": "R2", "value": r2_score(y_true, y_pred), "n_test": len(y_true)},
    ]


def classification_metric_rows(context: dict[str, Any], y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    y_pred = (y_proba >= threshold).astype(int)
    base = dict(context)
    return [
        {**base, "task": "classification", "metric": "ROC_AUC", "value": safe_auc(roc_auc_score, y_true, y_proba), "threshold": threshold, "n_test": len(y_true)},
        {**base, "task": "classification", "metric": "PR_AUC", "value": safe_auc(average_precision_score, y_true, y_proba, default=float(y_true.mean())), "threshold": threshold, "n_test": len(y_true)},
        {**base, "task": "classification", "metric": "Brier", "value": brier_score_loss(y_true, y_proba), "threshold": threshold, "n_test": len(y_true)},
        {**base, "task": "classification", "metric": "F1", "value": f1_score(y_true, y_pred, zero_division=0), "threshold": threshold, "n_test": len(y_true)},
        {**base, "task": "classification", "metric": "Recall", "value": recall_score(y_true, y_pred, zero_division=0), "threshold": threshold, "n_test": len(y_true)},
        {**base, "task": "classification", "metric": "Precision", "value": precision_score(y_true, y_pred, zero_division=0), "threshold": threshold, "n_test": len(y_true)},
        {**base, "task": "classification", "metric": "Accuracy", "value": accuracy_score(y_true, y_pred), "threshold": threshold, "n_test": len(y_true)},
    ]


def run_ablation(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for window_cfg in config["forecast_windows"]:
        window = window_cfg["name"]
        df = panel[panel["forecast_window"] == window].copy()
        train = df[df["split"] == "train"].copy()
        validation = df[df["split"] == "validation"].copy()
        test = df[df["split"] == "test"].copy()
        for feature_set in config["round2_feature_sets"]:
            features = feature_set_columns(panel, feature_set)
            for model_name, estimator in regression_models(config).items():
                context = {
                    "protocol": "time_split",
                    "fold": "primary",
                    "forecast_window": window,
                    "feature_set": feature_set,
                    "model": model_name,
                    "target": "yield_t_ha",
                }
                y_pred = fit_predict_regression(train, test, features, estimator)
                rows.extend(regression_metric_rows(context, test["yield_t_ha"], y_pred))
            for model_name, estimator in classification_models(config).items():
                context = {
                    "protocol": "time_split",
                    "fold": "primary",
                    "forecast_window": window,
                    "feature_set": feature_set,
                    "model": model_name,
                    "target": "low_yield_risk",
                }
                y_proba, threshold = classify_with_threshold(train, validation, test, features, estimator)
                rows.extend(classification_metric_rows(context, test["low_yield_risk"], y_proba, threshold))
    return pd.DataFrame(rows)


def split_for_protocol(df: pd.DataFrame, protocol: str, fold: dict[str, Any] | None, heldout: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if protocol == "time_split":
        return (
            df[df["split"] == "train"].copy(),
            df[df["split"] == "validation"].copy(),
            df[df["split"] == "test"].copy(),
            "primary",
        )
    if protocol == "rolling_origin":
        assert fold is not None
        train = df[df["year_start"] <= int(fold["train_end"])].copy()
        validation = df[(df["year_start"] > int(fold["train_end"]) - 4) & (df["year_start"] <= int(fold["train_end"]))].copy()
        test = df[(df["year_start"] >= int(fold["test_start"])) & (df["year_start"] <= int(fold["test_end"]))].copy()
        return train, validation, test, fold["fold"]
    if protocol == "leave_one_region_out":
        assert heldout is not None
        base = df[df["region"] != heldout].copy()
        train = base[base["split"] == "train"].copy()
        validation = base[base["split"] == "validation"].copy()
        test = df[(df["region"] == heldout) & (df["split"] == "test")].copy()
        return train, validation, test, heldout
    if protocol == "leave_one_crop_out":
        assert heldout is not None
        base = df[df["crop"] != heldout].copy()
        train = base[base["split"] == "train"].copy()
        validation = base[base["split"] == "validation"].copy()
        test = df[(df["crop"] == heldout) & (df["split"] == "test")].copy()
        return train, validation, test, heldout
    raise ValueError(f"Unknown protocol: {protocol}")


def run_stress_validation(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    stress_feature_sets = ["weather_anomaly_soil_no_lag", "full_operational"]
    protocol_specs: list[tuple[str, dict[str, Any] | None, str | None]] = [("time_split", None, None)]
    protocol_specs.extend(("rolling_origin", fold, None) for fold in config["rolling_origin_folds"])
    protocol_specs.extend(("leave_one_region_out", None, region) for region in config["regions"])
    protocol_specs.extend(("leave_one_crop_out", None, crop) for crop in config["crops"])

    for protocol, fold, heldout in protocol_specs:
        for window_cfg in config["forecast_windows"]:
            window = window_cfg["name"]
            window_df = panel[panel["forecast_window"] == window].copy()
            for feature_set in stress_feature_sets:
                features = feature_set_columns(panel, feature_set)
                train, validation, test, fold_name = split_for_protocol(window_df, protocol, fold, heldout)
                if train.empty or validation.empty or test.empty:
                    continue
                for model_name, estimator in regression_models(config, stress=True).items():
                    context = {
                        "protocol": protocol,
                        "fold": fold_name,
                        "forecast_window": window,
                        "feature_set": feature_set,
                        "model": model_name,
                        "target": "yield_t_ha",
                    }
                    y_pred = fit_predict_regression(train, test, features, estimator)
                    rows.extend(regression_metric_rows(context, test["yield_t_ha"], y_pred))
                for model_name, estimator in classification_models(config, stress=True).items():
                    context = {
                        "protocol": protocol,
                        "fold": fold_name,
                        "forecast_window": window,
                        "feature_set": feature_set,
                        "model": model_name,
                        "target": "low_yield_risk",
                    }
                    y_proba, threshold = classify_with_threshold(train, validation, test, features, estimator)
                    rows.extend(classification_metric_rows(context, test["low_yield_risk"], y_proba, threshold))
    return pd.DataFrame(rows)


def train_pipeline(train: pd.DataFrame, features: list[str], estimator: Any) -> Pipeline:
    pipe = Pipeline([("preprocess", make_preprocessor(features)), ("model", clone(estimator))])
    pipe.fit(train[features], train["yield_t_ha"])
    return pipe


def group_permutation_importance(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["project"]["random_state"]))
    rows = []
    group_map = column_groups(panel)
    feature_sets = ["weather_anomaly_soil_no_lag", "full_operational"]
    for window in ROUND2_IMPORTANCE_WINDOWS:
        df = panel[panel["forecast_window"] == window].copy()
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == "test"].copy()
        for feature_set in feature_sets:
            features = feature_set_columns(panel, feature_set)
            pipe = train_pipeline(train, features, Ridge(alpha=1.0))
            baseline_pred = pipe.predict(test[features])
            baseline_rmse = rmse(test["yield_t_ha"], baseline_pred)
            for group_name in [
                "categorical_time",
                "rainfall",
                "heat_cold",
                "energy_dryness",
                "compound_stress",
                "weather_anomaly",
                "soil",
                "lag_yield",
            ]:
                group_features = [c for c in group_map[group_name] if c in features]
                if not group_features:
                    continue
                deltas = []
                for _ in range(5):
                    permuted = test[features].copy()
                    order = rng.permutation(len(permuted))
                    for col in group_features:
                        permuted[col] = permuted[col].to_numpy()[order]
                    pred = pipe.predict(permuted)
                    deltas.append(rmse(test["yield_t_ha"], pred) - baseline_rmse)
                rows.append(
                    {
                        "forecast_window": window,
                        "feature_set": feature_set,
                        "model": "Ridge",
                        "feature_group": group_name,
                        "n_features": len(group_features),
                        "baseline_rmse": baseline_rmse,
                        "rmse_delta_mean": float(np.mean(deltas)),
                        "rmse_delta_std": float(np.std(deltas)),
                    }
                )
    return pd.DataFrame(rows)


def best_metric_table(metrics: pd.DataFrame, top_per_window: int | None = 5) -> pd.DataFrame:
    reg = metrics[(metrics["task"] == "regression") & (metrics["metric"].isin(["MAE", "RMSE", "R2"]))]
    pivot = reg.pivot_table(
        index=["forecast_window", "feature_set", "model"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.sort_values(["forecast_window", "RMSE"])
    if top_per_window is None:
        return pivot
    return pivot.groupby("forecast_window").head(top_per_window)


def stress_best_summary_table(stress: pd.DataFrame, feature_sets: list[str] | None = None) -> pd.DataFrame:
    if feature_sets is None:
        feature_sets = ["weather_anomaly_soil_no_lag", "full_operational"]
    stress_reg = stress[(stress["task"] == "regression") & (stress["metric"].isin(["RMSE", "R2"]))]
    validation = stress_reg.pivot_table(
        index=["protocol", "fold", "forecast_window", "feature_set", "model"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    validation_summary = (
        validation.groupby(["protocol", "feature_set", "model"], as_index=False)
        .agg(mean_RMSE=("RMSE", "mean"), mean_R2=("R2", "mean"), folds=("fold", "nunique"))
    )
    return (
        validation_summary[validation_summary["feature_set"].isin(feature_sets)]
        .sort_values(["protocol", "feature_set", "mean_RMSE"])
        .groupby(["protocol", "feature_set"], as_index=False, sort=False)
        .head(1)
        .copy()
    )


def make_paper_tables(ablation: pd.DataFrame, stress: pd.DataFrame, importance: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    reg = ablation[(ablation["task"] == "regression") & (ablation["metric"].isin(["MAE", "RMSE", "R2"]))]
    reg_p = reg.pivot_table(
        index=["forecast_window", "feature_set", "model"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    lead_time = (
        reg_p[reg_p["feature_set"].isin(["weather_anomaly_soil_no_lag", "full_operational"])]
        .sort_values(["forecast_window", "feature_set", "RMSE"])
        .groupby(["forecast_window", "feature_set"])
        .head(1)
        .sort_values(["feature_set", "forecast_window"])
    )
    ablation_table = (
        reg_p[reg_p["model"].isin(["Ridge", "ElasticNet", "LightGBM", "CatBoost"])]
        .sort_values(["feature_set", "forecast_window", "RMSE"])
        .groupby(["feature_set", "forecast_window"])
        .head(1)
        .sort_values(["forecast_window", "RMSE"])
    )
    stress_reg = stress[(stress["task"] == "regression") & (stress["metric"].isin(["RMSE", "R2"]))]
    validation = stress_reg.pivot_table(
        index=["protocol", "fold", "forecast_window", "feature_set", "model"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    validation_summary = (
        validation.groupby(["protocol", "feature_set", "model"], as_index=False)
        .agg(mean_RMSE=("RMSE", "mean"), mean_R2=("R2", "mean"), folds=("fold", "nunique"))
        .sort_values(["protocol", "mean_RMSE"])
    )
    stress_best_summary = stress_best_summary_table(stress)
    importance_table = (
        importance.sort_values("rmse_delta_mean", ascending=False)
        .groupby(["forecast_window", "feature_set"])
        .head(5)
        .sort_values(["forecast_window", "feature_set", "rmse_delta_mean"], ascending=[True, True, False])
    )
    return {
        "round2_paper_table_lead_time": lead_time,
        "round2_paper_table_ablation": ablation_table,
        "round2_paper_table_validation": validation_summary,
        "round2_paper_table_stress_best_summary": stress_best_summary,
        "round2_feature_group_importance": importance_table,
    }


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_round2_figures(ablation: pd.DataFrame, stress: pd.DataFrame, importance: pd.DataFrame, config: dict[str, Any]) -> None:
    fig_dir = configured_path(config, "figures_dir")
    order = {w["name"]: w["order"] for w in config["forecast_windows"]}
    reg = ablation[(ablation["task"] == "regression") & (ablation["metric"].isin(["RMSE", "R2"]))]
    reg_p = reg.pivot_table(index=["forecast_window", "feature_set", "model"], columns="metric", values="value").reset_index()
    best = reg_p.sort_values(["feature_set", "forecast_window", "RMSE"]).groupby(["feature_set", "forecast_window"]).head(1)
    best["window_order"] = best["forecast_window"].map(order)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for feature_set, group in best.sort_values("window_order").groupby("feature_set"):
        ax.plot(group["forecast_window"], group["RMSE"], marker="o", label=FEATURE_LABELS.get(feature_set, feature_set))
    ax.set_title("Round 2 Ablation RMSE By Lead Time")
    ax.set_xlabel("Forecast window")
    ax.set_ylabel("Best RMSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, fig_dir / "fig12_ablation_rmse_by_window.png")

    full = best[best["feature_set"] == "full_operational"][["forecast_window", "R2"]].rename(columns={"R2": "full_R2"})
    delta = best.merge(full, on="forecast_window", how="left")
    delta["r2_delta_vs_full"] = delta["R2"] - delta["full_R2"]
    compact_delta_labels = {
        "identity_time_only": "Identity",
        "weather_anomaly_soil_no_lag": "NYH",
        "full_operational": "Operational",
        "weather_stage_only": "Weather",
        "weather_plus_anomaly": "Weather+Dev",
        "weather_plus_soil": "Weather+Soil",
        "lag_yield_only": "Yield hist.",
    }
    delta["feature_label"] = delta["feature_set"].map(compact_delta_labels)
    pivot = delta.pivot(index="forecast_window", columns="feature_label", values="r2_delta_vs_full").reindex(sorted(order, key=order.get))
    column_order = [
        "Identity",
        "NYH",
        "Operational",
        "Weather",
        "Weather+Dev",
        "Weather+Soil",
        "Yield hist.",
    ]
    pivot = pivot[[column for column in column_order if column in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12.8, 4.9))
    norm = TwoSlopeNorm(vmin=float(np.nanmin(pivot.values)), vcenter=0.0, vmax=max(0.001, float(np.nanmax(pivot.values))))
    im = ax.imshow(pivot.values, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=12)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10.5, color="#1f1f1f")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("R$^2$ delta", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("R$^2$ Delta Versus Full Operational Model", fontsize=15)
    ax.set_xlabel("Feature regime", fontsize=12)
    ax.set_ylabel("Forecast window", fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    save_fig(fig, fig_dir / "fig13_ablation_r2_delta_by_feature_set.png")

    stress_summary = stress_best_summary_table(stress)
    stress_summary["protocol_label"] = stress_summary["protocol"].map(PROTOCOL_LABELS)
    stress_summary["feature_label"] = stress_summary["feature_set"].map(FEATURE_LABELS)
    matrix = stress_summary.pivot(index="protocol_label", columns="feature_label", values="mean_RMSE")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(matrix.values, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Stress Validation Mean RMSE")
    fig.colorbar(im, ax=ax, label="RMSE")
    save_fig(fig, fig_dir / "fig14_stress_validation_heatmap.png")

    imp = importance.copy()
    imp = imp[imp["rmse_delta_mean"] > 0].sort_values("rmse_delta_mean", ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = (
        imp["forecast_window"]
        + " | "
        + imp["feature_set"].map(FEATURE_LABELS)
        + " | "
        + imp["feature_group"].replace({"weather_anomaly": "weather_dev"})
    )
    ax.barh(labels[::-1], imp["rmse_delta_mean"].iloc[::-1], color="#2f6f73")
    ax.set_xlabel("RMSE increase after permutation")
    ax.set_title("Feature Group Importance")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, fig_dir / "fig15_feature_group_importance.png")

    ps = best[best["feature_set"] == "weather_anomaly_soil_no_lag"].sort_values("window_order")
    op = best[best["feature_set"] == "full_operational"].sort_values("window_order")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(ps["forecast_window"], ps["RMSE"], marker="o", label="No-yield-history weather-soil")
    axes[0].plot(op["forecast_window"], op["RMSE"], marker="o", label="Operational with yield history")
    axes[0].set_title("RMSE")
    axes[1].plot(ps["forecast_window"], ps["R2"], marker="o", label="No-yield-history weather-soil")
    axes[1].plot(op["forecast_window"], op["R2"], marker="o", label="Operational with yield history")
    axes[1].set_title("R2")
    for ax in axes:
        ax.set_xlabel("Forecast window")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
    save_fig(fig, fig_dir / "fig16_paper_safe_vs_operational_model.png")


def write_round2_reports(
    ablation: pd.DataFrame,
    stress: pd.DataFrame,
    importance: pd.DataFrame,
    feature_validation: pd.DataFrame,
    paper_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> None:
    round2_dir = configured_path(config, "reports_dir").parent / ROUND2_DIR_NAME
    round2_dir.mkdir(parents=True, exist_ok=True)

    write_markdown(
        round2_dir / "01_ROUND2_PLAN_ABLATION_VALIDATION.md",
        """
# Round 2 Plan: Ablation, Stress Validation, And Paper-Ready Evidence

This round tests whether the improved AUS early-warning result is paper-proof. The key scientific question is whether performance is driven by stage-aware weather and soil signals, or mainly by historical lag-yield information.

Scope stays Australia-only, winter crops, May-Jun to May-Oct forecast windows, and state-level decision support.
""",
    )

    all_best_ablation = best_metric_table(ablation, top_per_window=None)
    best_ablation = all_best_ablation.head(12)
    stress_summary = paper_tables["round2_paper_table_validation"].head(12)
    top_importance = paper_tables["round2_feature_group_importance"].head(15)
    ps = all_best_ablation[all_best_ablation["feature_set"] == "weather_anomaly_soil_no_lag"].sort_values("RMSE").head(1)
    op = all_best_ablation[all_best_ablation["feature_set"] == "full_operational"].sort_values("RMSE").head(1)

    write_markdown(
        round2_dir / "02_ROUND2_IMPLEMENTATION_LOG.md",
        f"""
# Round 2 Implementation Log

## Implemented

- Added fixed ablation feature sets.
- Added time-split, rolling-origin, leave-one-region-out, and leave-one-crop-out stress validation.
- Added group permutation importance for paper-safe and operational feature sets.
- Generated paper tables and figures 12-16.

## Feature Set Validation

{markdown_table(feature_validation)}

## Output Tables

- `round2_ablation_metrics.csv`
- `round2_stress_validation_metrics.csv`
- `round2_feature_group_importance.csv`
- `round2_paper_table_lead_time.csv`
- `round2_paper_table_ablation.csv`
- `round2_paper_table_validation.csv`

## Output Figures

- `fig12_ablation_rmse_by_window.png`
- `fig13_ablation_r2_delta_by_feature_set.png`
- `fig14_stress_validation_heatmap.png`
- `fig15_feature_group_importance.png`
- `fig16_paper_safe_vs_operational_model.png`
""",
    )

    write_markdown(
        round2_dir / "03_ROUND2_RESULTS_ABLATION_VALIDATION.md",
        f"""
# Round 2 Results: Ablation And Stress Validation

## Best Ablation Results

{markdown_table(best_ablation)}

## Stress Validation Summary

{markdown_table(stress_summary)}

## Top Feature Group Importance

{markdown_table(top_importance)}

## Interpretation

The ablation table separates paper-safe weather/anomaly/soil evidence from operational forecasts that additionally use historical lag-yield features. If the operational model materially outperforms the paper-safe model, the paper should present both as different claims rather than mixing them.
""",
    )

    if not ps.empty and not op.empty:
        ps_row = ps.iloc[0]
        op_row = op.iloc[0]
        gap_text = (
            f"Operational best RMSE is {op_row['RMSE']:.3f} "
            f"({op_row['forecast_window']}, {op_row['model']}); "
            f"paper-safe best RMSE is {ps_row['RMSE']:.3f} "
            f"({ps_row['forecast_window']}, {ps_row['model']})."
        )
    else:
        gap_text = "Paper-safe and operational comparison could not be computed."

    write_markdown(
        round2_dir / "04_ROUND2_PAPER_READY_TABLES_AND_CLAIMS.md",
        f"""
# Round 2 Paper-Ready Tables And Claims

## Recommended Reporting Split

- **Paper-safe scientific model:** `weather_anomaly_soil_no_lag`.
- **Operational forecasting model:** `full_operational`.

{gap_text}

## Lead-Time Table

{markdown_table(paper_tables['round2_paper_table_lead_time'].head(20))}

## Ablation Table

{markdown_table(paper_tables['round2_paper_table_ablation'].head(20))}

## Validation Table

{markdown_table(paper_tables['round2_paper_table_validation'].head(20))}

## Safe Claims

- Stage-aware weather and soil features can be evaluated separately from historical yield memory.
- Operational performance improves when lag-yield features are allowed, but that should be labelled as operational forecasting rather than pure weather early warning.
- Stress validation identifies which crops and regions are harder to transfer across.

## Limitations To Write

- Leave-one-crop-out is a hard unseen-crop test and may underperform.
- Region-level soil features should be treated as background vulnerability indicators, not causal farm-level explanations.
- The framework remains state-level and should not be presented as farm-level prediction.
""",
    )


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    tables_dir = configured_path(config, "tables_dir")
    round2_dir = configured_path(config, "reports_dir").parent / ROUND2_DIR_NAME
    round2_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel_improved.csv")
    feature_validation = validate_feature_sets(panel, config["round2_feature_sets"])
    ablation = run_ablation(panel, config)
    stress = run_stress_validation(panel, config)
    importance = group_permutation_importance(panel, config)
    paper_tables = make_paper_tables(ablation, stress, importance, config)

    ablation.to_csv(tables_dir / "round2_ablation_metrics.csv", index=False)
    stress.to_csv(tables_dir / "round2_stress_validation_metrics.csv", index=False)
    importance.to_csv(tables_dir / "round2_feature_group_importance.csv", index=False)
    for name, table in paper_tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    feature_validation.to_csv(tables_dir / "round2_feature_set_validation.csv", index=False)

    plot_round2_figures(ablation, stress, importance, config)
    write_round2_reports(ablation, stress, importance, feature_validation, paper_tables, config)
    print(f"Wrote Round 2 outputs to {tables_dir} and {round2_dir}")


if __name__ == "__main__":
    main()
