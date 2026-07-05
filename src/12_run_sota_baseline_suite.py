from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from common import configured_path, ensure_output_dirs, load_config, markdown_table, write_markdown


WINDOW_ORDER = ["May-Jun", "May-Jul", "May-Aug", "May-Sep", "May-Oct"]
WEATHER_SEQUENCE_COLUMNS = ["rain_mm", "tmax_c", "tmin_c", "radiation_mj_m2", "vp_hpa", "evap_mm"]
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


@dataclass(frozen=True)
class ModelSpec:
    group: str
    model: str
    estimator: Any
    params: dict[str, Any]


class PYGAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lam: float = 0.6, max_iter: int = 120):
        self.lam = lam
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: pd.Series | np.ndarray):
        from functools import reduce

        from pygam import LinearGAM, s

        X_arr = np.asarray(X, dtype=float)
        terms = reduce(lambda acc, term: acc + term, (s(i) for i in range(X_arr.shape[1])))
        self.model_ = LinearGAM(terms, lam=self.lam, max_iter=self.max_iter).fit(X_arr, np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)), dtype=float)


def optional_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def importable_module(name: str) -> bool:
    if importlib.util.find_spec(name) is None:
        return False
    try:
        __import__(name)
    except Exception:
        return False
    return True


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def column_groups(panel: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(panel.columns)
    return {
        "categorical_time": [c for c in ["region", "crop", "year_start"] if c in cols],
        "weather_stage": [c for c in BASE_WEATHER_COLUMNS if c in cols],
        "weather_deviation": [c for c in cols if "anom_train_region_window" in c],
        "soil": [c for c in cols if c.startswith("soil_")],
        "lag_yield": [
            c
            for c in cols
            if c.startswith("yield_lag") or "rolling" in c or "expanding" in c
        ],
    }


def feature_columns(panel: pd.DataFrame, regime: str) -> list[str]:
    groups = column_groups(panel)
    if regime == "no_yield_history_weather_soil":
        features = (
            groups["categorical_time"]
            + groups["weather_stage"]
            + groups["weather_deviation"]
            + groups["soil"]
        )
    elif regime == "operational_with_yield_history":
        features = (
            groups["categorical_time"]
            + groups["weather_stage"]
            + groups["weather_deviation"]
            + groups["soil"]
            + groups["lag_yield"]
        )
    else:
        raise ValueError(f"Unknown feature regime: {regime}")
    blocked = TARGET_COLUMNS | LEAKAGE_COLUMNS | META_COLUMNS
    out = sorted(dict.fromkeys([c for c in features if c not in blocked and c in panel.columns]))
    if any(c in out for c in LEAKAGE_COLUMNS | TARGET_COLUMNS):
        raise RuntimeError(f"Leakage columns found in {regime}: {set(out) & (LEAKAGE_COLUMNS | TARGET_COLUMNS)}")
    return out


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in ["region", "crop"] if c in features]
    numeric = [c for c in features if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def tabular_model_specs(seed: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec("Classical ML", "Ridge", Ridge(alpha=1.0), {"alpha": 1.0}),
        ModelSpec("Classical ML", "Ridge", Ridge(alpha=3.0), {"alpha": 3.0}),
        ModelSpec("Classical ML", "ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20000, random_state=seed), {"alpha": 0.01, "l1_ratio": 0.2}),
        ModelSpec("Classical ML", "ElasticNet", ElasticNet(alpha=0.03, l1_ratio=0.4, max_iter=20000, random_state=seed), {"alpha": 0.03, "l1_ratio": 0.4}),
        ModelSpec("Classical ML", "RandomForest", RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=-1), {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt"}),
        ModelSpec("Classical ML", "RandomForest", RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=5, max_features=0.5, random_state=seed, n_jobs=-1), {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 5, "max_features": 0.5}),
        ModelSpec("Classical ML", "SVR-RBF", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"), {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}),
        ModelSpec("Classical ML", "SVR-RBF", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"), {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}),
        ModelSpec("Strong tabular ML", "HistGradientBoosting", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.05, random_state=seed), {"max_iter": 250, "learning_rate": 0.04, "max_leaf_nodes": 15}),
    ]
    if optional_module("xgboost"):
        from xgboost import XGBRegressor

        specs.extend(
            [
                ModelSpec("Strong tabular ML", "XGBoost", XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, objective="reg:squarederror", n_jobs=1), {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0}),
                ModelSpec("Strong tabular ML", "XGBoost", XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, random_state=seed, objective="reg:squarederror", n_jobs=1), {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0}),
            ]
        )
    if optional_module("lightgbm"):
        from lightgbm import LGBMRegressor

        specs.append(ModelSpec("Strong tabular ML", "LightGBM", LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=10, reg_lambda=0.5, random_state=seed, verbose=-1), {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 15}))
    if optional_module("catboost"):
        from catboost import CatBoostRegressor

        specs.append(ModelSpec("Strong tabular ML", "CatBoost", CatBoostRegressor(iterations=260, depth=4, learning_rate=0.04, random_seed=seed, verbose=False, loss_function="RMSE", allow_writing_files=False), {"iterations": 260, "depth": 4, "learning_rate": 0.04}))
    if optional_module("interpret"):
        from interpret.glassbox import ExplainableBoostingRegressor

        specs.append(ModelSpec("Interpretable ML", "EBM", ExplainableBoostingRegressor(random_state=seed), {"package": "interpret"}))
    elif optional_module("pygam"):
        specs.append(ModelSpec("Interpretable ML", "GAM", PYGAMRegressor(lam=0.6, max_iter=120), {"package": "pygam", "lam": 0.6, "max_iter": 120}))
    return specs


def evaluate_tabular_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    specs: list[ModelSpec],
) -> tuple[dict[str, Any], np.ndarray]:
    best: tuple[float, ModelSpec, Pipeline] | None = None
    for spec in specs:
        pipe = Pipeline([("preprocess", make_preprocessor(features)), ("model", clone(spec.estimator))])
        pipe.fit(train[features], train["yield_t_ha"])
        val_pred = pipe.predict(validation[features])
        score = rmse(validation["yield_t_ha"], val_pred)
        if best is None or score < best[0]:
            best = (score, spec, pipe)
    assert best is not None
    val_rmse, spec, pipe = best
    test_pred = pipe.predict(test[features])
    metrics = metric_dict(test["yield_t_ha"], test_pred)
    return {
        "model_group": spec.group,
        "model": spec.model,
        "selected_hyperparams": json.dumps(spec.params, sort_keys=True),
        "validation_RMSE": float(val_rmse),
        **metrics,
    }, np.asarray(test_pred, dtype=float)


def add_historical_baselines(panel: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    unique = panel[["region", "crop", "year_start", "split", "yield_t_ha"]].drop_duplicates(["region", "crop", "year_start"]).copy()
    train = unique[unique["split"] == "train"].copy()
    global_mean = float(train["yield_t_ha"].mean())
    crop_mean = train.groupby("crop")["yield_t_ha"].mean().to_dict()
    crop_region_mean = train.groupby(["crop", "region"])["yield_t_ha"].mean().to_dict()

    def fit_linear(group: pd.DataFrame) -> tuple[float, float] | None:
        if len(group) < 3 or group["year_start"].nunique() < 3:
            return None
        slope, intercept = np.polyfit(group["year_start"].to_numpy(dtype=float), group["yield_t_ha"].to_numpy(dtype=float), 1)
        return float(slope), float(intercept)

    crop_region_trends = {
        key: trend
        for key, group in train.groupby(["crop", "region"], sort=False)
        if (trend := fit_linear(group)) is not None
    }
    crop_trends = {
        crop: trend
        for crop, group in train.groupby("crop", sort=False)
        if (trend := fit_linear(group)) is not None
    }

    def trend_prediction(crop: str, region: str, year: int) -> float:
        trend = crop_region_trends.get((crop, region)) or crop_trends.get(crop)
        if trend is None:
            return crop_mean.get(crop, global_mean)
        slope, intercept = trend
        return float(slope * year + intercept)

    out = unique.sort_values(["region", "crop", "year_start"]).copy()
    out["hist_crop_region_mean"] = [
        crop_region_mean.get((row.crop, row.region), crop_mean.get(row.crop, global_mean))
        for row in out.itertuples(index=False)
    ]
    out["hist_crop_region_trend"] = [
        trend_prediction(str(row.crop), str(row.region), int(row.year_start))
        for row in out.itertuples(index=False)
    ]
    parts = []
    for _, group in out.groupby(["region", "crop"], sort=False):
        group = group.sort_values("year_start").copy()
        group["hist_previous_year"] = group["yield_t_ha"].shift(1)
        group["hist_rolling3"] = group["yield_t_ha"].shift(1).rolling(3, min_periods=1).mean()
        parts.append(group)
    out = pd.concat(parts, ignore_index=True)
    for col in ["hist_previous_year", "hist_rolling3"]:
        out[col] = out[col].fillna(out["hist_crop_region_mean"])

    rows = []
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    labels = {
        "hist_crop_region_mean": ("Historical", "Crop-region train mean", "train-only mean"),
        "hist_crop_region_trend": ("Historical", "Crop-region train trend", "train-only crop-region/crop/global trend hierarchy"),
        "hist_previous_year": ("Historical", "Previous-year yield", "past-only chronological"),
        "hist_rolling3": ("Historical", "3-year rolling mean", "past-only chronological"),
    }
    for window in WINDOW_ORDER:
        window_test = (
            panel[(panel["forecast_window"] == window) & (panel["split"] == "test")]
            [["region", "crop", "year_start", "yield_t_ha"]]
            .sort_values(["region", "crop", "year_start"])
        )
        merged = window_test.merge(out[["region", "crop", "year_start", *labels.keys()]], on=["region", "crop", "year_start"], how="left", validate="one_to_one")
        for col, (group, model, params) in labels.items():
            y_pred = merged[col].to_numpy(dtype=float)
            metrics = metric_dict(merged["yield_t_ha"], y_pred)
            rows.append(
                {
                    "model_group": group,
                    "model": model,
                    "feature_regime": "past_only_yield_history",
                    "forecast_window": window,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "validation_RMSE": np.nan,
                    "n_train": int((out["split"] == "train").sum()),
                    "n_val": int((out["split"] == "validation").sum()),
                    "n_test": int(len(merged)),
                    "selected_hyperparams": json.dumps({"rule": params}),
                    "seed": seed,
                    "runtime_seconds": 0.0,
                    "status": "ok",
                }
            )
            preds[(model, "past_only_yield_history", window)] = y_pred
    return pd.DataFrame(rows), preds


def run_tabular_suite(panel: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    specs = tabular_model_specs(seed)
    rows = []
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    for window in WINDOW_ORDER:
        df = panel[panel["forecast_window"] == window].copy()
        train = df[df["split"] == "train"].sort_values(["region", "crop", "year_start"]).copy()
        validation = df[df["split"] == "validation"].sort_values(["region", "crop", "year_start"]).copy()
        test = df[df["split"] == "test"].sort_values(["region", "crop", "year_start"]).copy()
        for regime in ["no_yield_history_weather_soil", "operational_with_yield_history"]:
            features = feature_columns(panel, regime)
            by_model: dict[str, list[ModelSpec]] = {}
            for spec in specs:
                by_model.setdefault(spec.model, []).append(spec)
            for model_name, model_specs in by_model.items():
                start = time.perf_counter()
                try:
                    result, pred = evaluate_tabular_model(train, validation, test, features, model_specs)
                    status = "ok"
                except Exception as exc:
                    result = {
                        "model_group": model_specs[0].group,
                        "model": model_name,
                        "selected_hyperparams": json.dumps({"error": str(exc)[:300]}),
                        "validation_RMSE": np.nan,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                    }
                    pred = np.full(len(test), np.nan)
                    status = "failed"
                elapsed = time.perf_counter() - start
                row = {
                    **result,
                    "feature_regime": regime,
                    "forecast_window": window,
                    "n_train": int(len(train)),
                    "n_val": int(len(validation)),
                    "n_test": int(len(test)),
                    "seed": seed,
                    "runtime_seconds": float(elapsed),
                    "status": status,
                }
                rows.append(row)
                if status == "ok":
                    preds[(model_name, regime, window)] = pred
    return pd.DataFrame(rows), preds


def make_sequence_arrays(config: dict[str, Any], panel: pd.DataFrame, window: str, regime: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weather = pd.read_csv(configured_path(config, "daily_weather"), parse_dates=["date"])
    window_cfg = next(w for w in config["forecast_windows"] if w["name"] == window)
    weather["year_start"] = weather["date"].dt.year
    weather["month"] = weather["date"].dt.month
    weather = weather[(weather["month"] >= int(window_cfg["start_month"])) & (weather["month"] <= int(window_cfg["end_month"]))].copy()
    weather = weather.sort_values(["region", "year_start", "date"])
    seq_map: dict[tuple[str, int], np.ndarray] = {}
    for (region, year), group in weather.groupby(["region", "year_start"], sort=False):
        seq_map[(str(region), int(year))] = group[WEATHER_SEQUENCE_COLUMNS].to_numpy(dtype=np.float32)
    length = max(len(v) for v in seq_map.values())
    static_cols = feature_columns(panel, "operational_with_yield_history" if regime == "daily_weather_sequence_operational" else "no_yield_history_weather_soil")
    static_cols = [c for c in static_cols if c not in BASE_WEATHER_COLUMNS and "anom_train_region_window" not in c]
    df = panel[panel["forecast_window"] == window].copy().sort_values(["split", "region", "crop", "year_start"])
    x_seq = np.zeros((len(df), length, len(WEATHER_SEQUENCE_COLUMNS)), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        arr = seq_map[(str(row.region), int(row.year_start))]
        x_seq[i, : len(arr), :] = arr

    static_pre = make_preprocessor(static_cols)
    train_mask = df["split"] == "train"
    static_pre.fit(df.loc[train_mask, static_cols])
    x_static = static_pre.transform(df[static_cols]).astype(np.float32)
    y = df["yield_t_ha"].to_numpy(dtype=np.float32)

    seq_mean = x_seq[train_mask.to_numpy()].reshape(-1, x_seq.shape[-1]).mean(axis=0)
    seq_std = x_seq[train_mask.to_numpy()].reshape(-1, x_seq.shape[-1]).std(axis=0)
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)
    x_seq = (x_seq - seq_mean) / seq_std
    return (
        x_seq[df["split"].eq("train").to_numpy()],
        x_static[df["split"].eq("train").to_numpy()],
        y[df["split"].eq("train").to_numpy()],
        x_seq[df["split"].eq("validation").to_numpy()],
        x_static[df["split"].eq("validation").to_numpy()],
        y[df["split"].eq("validation").to_numpy()],
        x_seq[df["split"].eq("test").to_numpy()],
        x_static[df["split"].eq("test").to_numpy()],
        y[df["split"].eq("test").to_numpy()],
    )


def run_sequence_suite(config: dict[str, Any], panel: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    rows = []
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    try:
        import torch
        from torch import nn
    except Exception as exc:
        for window in WINDOW_ORDER:
            for regime in ["daily_weather_sequence_no_yield_history", "daily_weather_sequence_operational"]:
                rows.append(
                    {
                        "model_group": "Sequence",
                        "model": "DailyWeather-GRU",
                        "feature_regime": regime,
                        "forecast_window": window,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "train_RMSE": np.nan,
                        "validation_RMSE": np.nan,
                        "n_train": 0,
                        "n_val": 0,
                        "n_test": 0,
                        "selected_hyperparams": json.dumps({"error": f"torch import failed: {str(exc)[:220]}"}),
                        "seed": seed,
                        "runtime_seconds": 0.0,
                        "status": "failed",
                    }
                )
        return pd.DataFrame(rows), preds

    class DailyWeatherGRU(nn.Module):
        def __init__(self, n_weather: int, n_static: int, hidden_size: int, dropout: float):
            super().__init__()
            self.gru = nn.GRU(input_size=n_weather, hidden_size=hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_size + n_static, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x_seq, x_static):
            _, hidden = self.gru(x_seq)
            x = torch.cat([hidden[-1], x_static], dim=1)
            return self.head(self.dropout(x)).squeeze(1)

    def fit_candidate(
        xtr_s: np.ndarray,
        xtr_static: np.ndarray,
        ytr: np.ndarray,
        xv_s: np.ndarray,
        xv_static: np.ndarray,
        yv: np.ndarray,
        hidden_size: int,
        dropout: float,
        lr: float,
    ) -> tuple[float, float, Any, dict[str, Any]]:
        torch.manual_seed(seed)
        model = DailyWeatherGRU(xtr_s.shape[-1], xtr_static.shape[1], hidden_size, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        tensors = {
            "xtr_s": torch.tensor(xtr_s, dtype=torch.float32),
            "xtr_static": torch.tensor(xtr_static, dtype=torch.float32),
            "ytr": torch.tensor(ytr, dtype=torch.float32),
            "xv_s": torch.tensor(xv_s, dtype=torch.float32),
            "xv_static": torch.tensor(xv_static, dtype=torch.float32),
        }
        best_state = None
        best_val = float("inf")
        best_epoch = 0
        patience = 16
        stale = 0
        max_epochs = 160
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(tensors["xtr_s"], tensors["xtr_static"])
            loss = loss_fn(pred, tensors["ytr"])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(tensors["xv_s"], tensors["xv_static"]).detach().cpu().numpy()
            val_score = rmse(yv, val_pred)
            if val_score + 1e-5 < best_val:
                best_val = val_score
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred = model(tensors["xtr_s"], tensors["xtr_static"]).detach().cpu().numpy()
        params = {"backend": "torch", "hidden_size": hidden_size, "dropout": dropout, "learning_rate": lr, "max_epochs": max_epochs, "patience": patience, "best_epoch": best_epoch}
        return best_val, rmse(ytr, train_pred), model, params

    for window in WINDOW_ORDER:
        for regime in ["daily_weather_sequence_no_yield_history", "daily_weather_sequence_operational"]:
            start = time.perf_counter()
            try:
                xtr_s, xtr_static, ytr, xv_s, xv_static, yv, xt_s, xt_static, yt = make_sequence_arrays(config, panel, window, regime)
                candidates = [
                    fit_candidate(xtr_s, xtr_static, ytr, xv_s, xv_static, yv, hidden_size=16, dropout=0.15, lr=0.001),
                    fit_candidate(xtr_s, xtr_static, ytr, xv_s, xv_static, yv, hidden_size=32, dropout=0.20, lr=0.001),
                ]
                val_score, train_score, model, params = min(candidates, key=lambda item: item[0])
                with torch.no_grad():
                    test_pred = model(torch.tensor(xt_s, dtype=torch.float32), torch.tensor(xt_static, dtype=torch.float32)).detach().cpu().numpy()
                metrics = metric_dict(yt, test_pred)
                row = {
                    "model_group": "Sequence",
                    "model": "DailyWeather-GRU",
                    "feature_regime": regime,
                    "forecast_window": window,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "train_RMSE": train_score,
                    "validation_RMSE": val_score,
                    "n_train": int(len(ytr)),
                    "n_val": int(len(yv)),
                    "n_test": int(len(yt)),
                    "selected_hyperparams": json.dumps(params, sort_keys=True),
                    "seed": seed,
                    "runtime_seconds": float(time.perf_counter() - start),
                    "status": "ok",
                }
                rows.append(row)
                preds[("DailyWeather-GRU", regime, window)] = test_pred
            except Exception as exc:
                rows.append(
                    {
                        "model_group": "Sequence",
                        "model": "DailyWeather-GRU",
                        "feature_regime": regime,
                        "forecast_window": window,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "train_RMSE": np.nan,
                        "validation_RMSE": np.nan,
                        "n_train": 0,
                        "n_val": 0,
                        "n_test": 0,
                        "selected_hyperparams": json.dumps({"error": str(exc)[:300]}),
                        "seed": seed,
                        "runtime_seconds": float(time.perf_counter() - start),
                        "status": "failed",
                    }
                )
    return pd.DataFrame(rows), preds


def bootstrap_ci(panel: pd.DataFrame, comparison: pd.DataFrame, preds: dict[tuple[str, str, str], np.ndarray], seed: int, n_boot: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    key_rows = comparison[(comparison["status"] == "ok") & (comparison["forecast_window"].isin(["May-Jun", "May-Oct"]))].copy()
    key_rows = key_rows.sort_values(["forecast_window", "feature_regime", "RMSE"]).groupby(["forecast_window", "feature_regime"]).head(2)
    for row in key_rows.itertuples(index=False):
        key = (row.model, row.feature_regime, row.forecast_window)
        if key not in preds:
            continue
        test = panel[(panel["forecast_window"] == row.forecast_window) & (panel["split"] == "test")].sort_values(["region", "crop", "year_start"]).copy()
        y = test["yield_t_ha"].to_numpy()
        pred = preds[key]
        values = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            values.append(rmse(y[idx], pred[idx]))
        lo, hi = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "model": row.model,
                "feature_regime": row.feature_regime,
                "forecast_window": row.forecast_window,
                "metric": "RMSE",
                "estimate": row.RMSE,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_bootstrap": n_boot,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def plot_baseline_comparison(comparison: pd.DataFrame, figures_dir: Path) -> None:
    ok = comparison[comparison["status"] == "ok"].copy()
    regimes = ["past_only_yield_history", "no_yield_history_weather_soil", "operational_with_yield_history", "daily_weather_sequence_no_yield_history"]
    rows = []
    for window in ["May-Jun", "May-Oct"]:
        for regime in regimes:
            subset = ok[(ok["forecast_window"] == window) & (ok["feature_regime"] == regime)]
            if subset.empty:
                continue
            rows.append(subset.sort_values("RMSE").iloc[0])
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["forecast_window"] + "\n" + plot_df["feature_regime"].str.replace("_", " ") + "\n" + plot_df["model"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].barh(plot_df["label"], plot_df["RMSE"], color="#33658a")
    axes[0].set_xlabel("RMSE (t/ha)")
    axes[0].set_title("Best RMSE by comparator")
    axes[1].barh(plot_df["label"], plot_df["R2"], color="#6a994e")
    axes[1].set_xlabel("R2")
    axes[1].set_title("Best R2 by comparator")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig20_baseline_comparison_rmse_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def latex_escape(value: Any) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_")


def write_tabular_latex(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main_baseline_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "past_only_yield_history": "Past-only yield history",
        "no_yield_history_weather_soil": "No-yield-history weather-soil",
        "operational_with_yield_history": "Operational with yield history",
        "daily_weather_sequence_no_yield_history": "Daily sequence no-yield",
        "daily_weather_sequence_operational": "Daily sequence + yield history",
    }
    wanted = [
        ("Historical", "past_only_yield_history"),
        ("Classical ML", "no_yield_history_weather_soil"),
        ("Strong tabular ML", "no_yield_history_weather_soil"),
        ("Interpretable ML", "no_yield_history_weather_soil"),
        ("Sequence", "daily_weather_sequence_no_yield_history"),
        ("Sequence", "daily_weather_sequence_operational"),
        ("Strong tabular ML", "operational_with_yield_history"),
        ("Classical ML", "operational_with_yield_history"),
    ]
    ok = comparison[comparison["status"] == "ok"].copy()
    rows = []
    for group, regime in wanted:
        subset = ok[
            (ok["model_group"] == group)
            & (ok["feature_regime"] == regime)
            & (ok["forecast_window"] == "May-Oct")
        ].copy()
        if subset.empty:
            continue
        best = subset.sort_values("RMSE").iloc[0]
        rows.append(
            {
                "Group": group,
                "Regime": labels.get(regime, regime),
                "Model": best["model"],
                "Uses daily sequence?": "Yes" if group == "Sequence" else "No",
                "Uses yield history?": "Yes"
                if regime in {"past_only_yield_history", "operational_with_yield_history", "daily_weather_sequence_operational"}
                else "No",
                "May-Oct RMSE": float(best["RMSE"]),
                "May-Oct R2": float(best["R2"]),
            }
        )
    return pd.DataFrame(rows)


def write_sota_latex_table(comparison: pd.DataFrame, path: Path) -> None:
    main_rows = main_baseline_rows(comparison)
    rows = [
        [
            row["Group"],
            row["Regime"],
            row["Model"],
            row["Uses daily sequence?"],
            row["Uses yield history?"],
            fmt(row["May-Oct RMSE"]),
            fmt(row["May-Oct R2"]),
        ]
        for _, row in main_rows.iterrows()
    ]
    write_tabular_latex(path, ["Group", "Regime", "Model", "Daily seq.", "Yield history", "May-Oct RMSE", "May-Oct R2"], rows)


def write_full_appendix_latex(comparison: pd.DataFrame, path: Path) -> None:
    ok = comparison[comparison["status"] == "ok"].copy()
    ok["window_order"] = ok["forecast_window"].map({name: i for i, name in enumerate(["May-Jun", "May-Jul", "May-Aug", "May-Sep", "May-Oct"])})
    ok = ok.sort_values(["model_group", "feature_regime", "window_order", "RMSE"])
    rows = [
        [
            row.model_group,
            row.feature_regime,
            row.forecast_window,
            row.model,
            fmt(row.MAE),
            fmt(row.RMSE),
            fmt(row.R2),
        ]
        for row in ok.itertuples(index=False)
    ]
    write_tabular_latex(path, ["Group", "Regime", "Window", "Model", "MAE", "RMSE", "R2"], rows)


def package_versions() -> dict[str, str | None]:
    packages = ["python", "python_executable", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "interpret", "pygam", "torch"]
    versions: dict[str, str | None] = {}
    for package in packages:
        if package == "python":
            import sys

            versions[package] = sys.version.split()[0]
            continue
        if package == "python_executable":
            import sys

            versions[package] = sys.executable
            continue
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def mirror_checklist_artifacts(project_root: Path, tables_dir: Path, figures_dir: Path, reports_dir: Path) -> None:
    mirrors = [
        (tables_dir / "sota_baseline_comparison.csv", project_root / "results" / "sota_baseline_comparison.csv"),
        (tables_dir / "sota_baseline_comparison.csv", project_root / "results" / "baselines_all_windows.csv"),
        (tables_dir / "sota_baseline_comparison_latex.tex", project_root / "results" / "sota_baseline_comparison_latex.tex"),
        (tables_dir / "baselines_may_oct_main.csv", project_root / "results" / "baselines_may_oct_main.csv"),
        (tables_dir / "baselines_may_oct_main.tex", project_root / "results" / "baselines_may_oct_main.tex"),
        (tables_dir / "baselines_full_appendix.tex", project_root / "results" / "baselines_full_appendix.tex"),
        (tables_dir / "full_window_baselines.csv", project_root / "results" / "full_window_baselines.csv"),
        (tables_dir / "sequence_baselines.csv", project_root / "results" / "sequence_baselines.csv"),
        (tables_dir / "bootstrap_metric_ci.csv", project_root / "results" / "bootstrap_metric_ci.csv"),
        (figures_dir / "fig20_baseline_comparison_rmse_r2.png", project_root / "figures" / "baseline_comparison_rmse_r2.png"),
        (figures_dir / "fig20_baseline_comparison_rmse_r2.png", project_root / "figures" / "baseline_rmse_by_window.png"),
        (reports_dir / "baseline_suite_config.json", project_root / "logs" / "baseline_suite_config.json"),
        (reports_dir / "sota_baseline_suite_log.md", project_root / "logs" / "baseline_suite_run_log.md"),
    ]
    for src, dst in mirrors:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_outputs(config: dict[str, Any], comparison: pd.DataFrame, sequence: pd.DataFrame, ci: pd.DataFrame, availability: dict[str, bool]) -> None:
    tables_dir = configured_path(config, "tables_dir")
    reports_dir = configured_path(config, "reports_dir")
    figures_dir = configured_path(config, "figures_dir")
    project_root = tables_dir.parents[1]
    comparison.to_csv(tables_dir / "sota_baseline_comparison.csv", index=False)
    comparison.to_csv(tables_dir / "full_window_baselines.csv", index=False)
    sequence.to_csv(tables_dir / "sequence_baselines.csv", index=False)
    ci.to_csv(tables_dir / "bootstrap_metric_ci.csv", index=False)
    main_rows = main_baseline_rows(comparison)
    main_rows.to_csv(tables_dir / "baselines_may_oct_main.csv", index=False)
    write_sota_latex_table(comparison, tables_dir / "sota_baseline_comparison_latex.tex")
    write_sota_latex_table(comparison, tables_dir / "baselines_may_oct_main.tex")
    write_full_appendix_latex(comparison, tables_dir / "baselines_full_appendix.tex")
    plot_baseline_comparison(comparison, figures_dir)
    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel_improved.csv")
    split_ids = (
        panel[["region", "crop", "year_start", "split"]]
        .drop_duplicates(["region", "crop", "year_start"])
        .sort_values(["split", "region", "crop", "year_start"])
    )
    split_ids.to_csv(reports_dir / "baseline_split_ids.csv", index=False)
    config_log = {
        "seed": int(config["project"]["random_state"]),
        "splits": config["splits"],
        "split_counts": split_ids["split"].value_counts().to_dict(),
        "split_ids": split_ids.to_dict("records"),
        "forecast_windows": [w["name"] for w in config["forecast_windows"]],
        "availability": availability,
        "package_versions": package_versions(),
        "feature_regimes": {
            "no_yield_history_weather_soil": "weather, train-derived weather deviations, soil, crop, region, year; no lagged yield",
            "operational_with_yield_history": "no-yield-history features plus lagged/rolling/expanding past yield",
        },
    }
    (reports_dir / "baseline_suite_config.json").write_text(json.dumps(config_log, indent=2), encoding="utf-8")
    preview = comparison[comparison["status"] == "ok"].sort_values(["forecast_window", "feature_regime", "RMSE"]).groupby(["forecast_window", "feature_regime"]).head(1)
    write_markdown(
        reports_dir / "sota_baseline_suite_log.md",
        f"""
# SOTA-Style Baseline Suite Log

## Availability

{markdown_table(pd.DataFrame([{"package": k, "available": v} for k, v in availability.items()]))}

## Best Comparator Per Window And Regime

{markdown_table(preview[["forecast_window", "feature_regime", "model_group", "model", "RMSE", "R2", "status"]].round(3))}

## Leakage Checks

- `production_kt`, `area_000ha`, target columns, and test-derived quantities are excluded from feature matrices.
- Tabular preprocessing is fit inside train-only pipelines.
- Hyperparameters are selected by validation RMSE, then test metrics are reported once.
- Sequence models use raw daily weather only up to the forecast-window cutoff and static features fit/scaled from train only.
- GRU sequence models use PyTorch; sequence comparator rows are completed GRU runs.
""",
    )
    mirror_checklist_artifacts(project_root, tables_dir, figures_dir, reports_dir)


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    seed = int(config["project"]["random_state"])
    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel_improved.csv")
    unique = panel[["region", "crop", "year_start"]].drop_duplicates()
    if len(panel) != 4830 or len(unique) != 966:
        raise RuntimeError(f"Unexpected panel shape: {len(panel)} rows and {len(unique)} unique observations")

    availability = {name: importable_module(name) for name in ["xgboost", "lightgbm", "catboost", "interpret", "pygam", "torch"]}
    historical, pred_map = add_historical_baselines(panel, seed)
    tabular, tabular_preds = run_tabular_suite(panel, seed)
    sequence, sequence_preds = run_sequence_suite(config, panel, seed)
    pred_map.update(tabular_preds)
    pred_map.update(sequence_preds)
    comparison = pd.concat([historical, tabular, sequence], ignore_index=True, sort=False)
    ci = bootstrap_ci(panel, comparison, pred_map, seed)
    write_outputs(config, comparison, sequence, ci, availability)
    print(f"Wrote {len(comparison)} baseline rows and {len(ci)} CI rows.")


if __name__ == "__main__":
    main()
