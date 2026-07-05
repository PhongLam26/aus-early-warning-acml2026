from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_panel_improved.csv"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "improvement_round3_paper_safe_residual_2026_06_24"

FORECAST_WINDOWS = ["May-Jun", "May-Jul", "May-Aug", "May-Sep", "May-Oct"]
TRAIN_END = 2012
VAL_START = 2013
VAL_END = 2016
TEST_START = 2017
TEST_END = 2021

IDENTITY_FEATURES = ["crop", "region", "year_start"]
LEAKAGE_FEATURES = {"production_kt", "area_000ha"}
LAG_PATTERNS = ("yield_lag", "rolling", "expanding")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def metric_rows(
    forecast_window: str,
    experiment: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float | None,
    n_train: int,
    n_val: int,
    n_test: int,
) -> list[dict[str, object]]:
    alpha_value = -1.0 if alpha is None else alpha
    scores = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    return [
        {
            "forecast_window": forecast_window,
            "experiment": experiment,
            "model": model,
            "metric": metric,
            "value": value,
            "alpha": alpha_value,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        }
        for metric, value in scores.items()
    ]


def select_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    weather_cols = [
        c
        for c in df.columns
        if c
        in {
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
            "n_weather_days",
            "lat",
            "lon",
            "window_order",
        }
    ]
    anomaly_cols = [c for c in df.columns if "_anom_train_region_window" in c]
    soil_cols = [c for c in df.columns if c.startswith("soil_")]
    paper_safe_cols = weather_cols + anomaly_cols + soil_cols
    for col in paper_safe_cols:
        if col in LEAKAGE_FEATURES or any(pattern in col for pattern in LAG_PATTERNS):
            raise ValueError(f"Paper-safe Round 3 feature failed leakage check: {col}")
    direct_cols = IDENTITY_FEATURES + paper_safe_cols
    return weather_cols, anomaly_cols, soil_cols, direct_cols


def make_linear_pipeline(features: list[str], categorical: Iterable[str]) -> Pipeline:
    categorical = list(categorical)
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", Ridge(alpha=3.0))])


def make_model_pipeline(features: list[str], categorical: Iterable[str], model_name: str) -> Pipeline:
    categorical = list(categorical)
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )
    if model_name == "Ridge":
        model = Ridge(alpha=3.0)
    elif model_name == "ElasticNet":
        model = ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=20000, random_state=42)
    elif model_name == "LightGBM" and LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unsupported model for this environment: {model_name}")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def choose_alpha(y_val: np.ndarray, base_val: np.ndarray, residual_val: np.ndarray) -> float:
    candidates = np.linspace(0.0, 1.25, 26)
    losses = [(alpha, rmse(y_val, base_val + alpha * residual_val)) for alpha in candidates]
    return min(losses, key=lambda item: item[1])[0]


def pivot_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=["forecast_window", "experiment", "model", "alpha", "n_train", "n_val", "n_test"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    return pivot.sort_values(["forecast_window", "RMSE"])


def make_figures(comparison: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    best = (
        comparison.sort_values(["forecast_window", "experiment", "RMSE"])
        .groupby(["forecast_window", "experiment"], as_index=False)
        .head(1)
    )
    order = {name: i for i, name in enumerate(FORECAST_WINDOWS)}
    best["window_order"] = best["forecast_window"].map(order)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for experiment, group in best.sort_values("window_order").groupby("experiment"):
        ax.plot(group["forecast_window"], group["RMSE"], marker="o", label=experiment)
    ax.set_title("Round 3 Paper-Safe Residual Test: RMSE")
    ax.set_xlabel("Forecast window")
    ax.set_ylabel("Test RMSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig17_round3_residual_vs_direct_rmse.png", dpi=220)
    plt.close(fig)

    identity = best[best["experiment"] == "identity_time_only"][["forecast_window", "RMSE", "R2"]].rename(
        columns={"RMSE": "identity_RMSE", "R2": "identity_R2"}
    )
    delta = best.merge(identity, on="forecast_window", how="left")
    delta = delta[delta["experiment"] != "identity_time_only"].copy()
    delta["R2_delta_vs_identity"] = delta["R2"] - delta["identity_R2"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    delta_pivot = delta.pivot(index="forecast_window", columns="experiment", values="R2_delta_vs_identity")
    delta_pivot = delta_pivot.reindex(FORECAST_WINDOWS)
    delta_pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Incremental R2 Versus Identity/Time Baseline")
    ax.set_xlabel("Forecast window")
    ax.set_ylabel("R2 delta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig18_round3_incremental_weather_signal.png", dpi=220)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    columns = list(shown.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "/")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(fmt(row[col]) for col in columns) + " |"
        for _, row in shown.iterrows()
    ]
    return "\n".join([header, separator] + body)


def write_reports(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_counts: dict[str, int],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best = (
        comparison.sort_values(["forecast_window", "experiment", "RMSE"])
        .groupby(["forecast_window", "experiment"], as_index=False)
        .head(1)
    )
    identity = best[best["experiment"] == "identity_time_only"][["forecast_window", "RMSE", "R2"]].rename(
        columns={"RMSE": "identity_RMSE", "R2": "identity_R2"}
    )
    result = best.merge(identity, on="forecast_window", how="left")
    result["RMSE_delta_vs_identity"] = result["RMSE"] - result["identity_RMSE"]
    result["R2_delta_vs_identity"] = result["R2"] - result["identity_R2"]

    residual_best = result[result["experiment"] == "residual_paper_safe"].sort_values("RMSE").head(1)
    direct_best = result[result["experiment"] == "direct_paper_safe"].sort_values("RMSE").head(1)
    identity_best = result[result["experiment"] == "identity_time_only"].sort_values("RMSE").head(1)

    if not residual_best.empty and not direct_best.empty:
        rb = residual_best.iloc[0]
        db = direct_best.iloc[0]
        ib = identity_best.iloc[0]
        if rb["RMSE"] < db["RMSE"] and rb["RMSE"] < ib["RMSE"]:
            decision = (
                "Round 3 improves the paper-safe story: residualized weather/anomaly/soil features "
                "beat both direct paper-safe modeling and the identity/time baseline."
            )
        elif rb["RMSE"] < db["RMSE"]:
            decision = (
                "Round 3 improves the direct paper-safe model but still does not beat the identity/time baseline. "
                "Use it as a diagnostic check, not as the main result."
            )
        else:
            decision = (
                "Round 3 does not improve the paper-safe model. Keep Round 2 as the main result and report "
                "that much of the state-level predictability is carried by identity/time and lag-yield history."
            )
        headline = (
            f"Best residual paper-safe: {rb['forecast_window']} {rb['model']} RMSE {rb['RMSE']:.3f}, "
            f"R2 {rb['R2']:.3f}. Best direct paper-safe: {db['forecast_window']} {db['model']} "
            f"RMSE {db['RMSE']:.3f}, R2 {db['R2']:.3f}. Best identity/time: {ib['forecast_window']} "
            f"{ib['model']} RMSE {ib['RMSE']:.3f}, R2 {ib['R2']:.3f}."
        )
    else:
        decision = "Round 3 decision could not be computed."
        headline = "Round 3 headline could not be computed."

    (REPORT_DIR / "02_ROUND3_IMPLEMENTATION_LOG.md").write_text(
        f"""# Round 3 Implementation Log

## Implemented

- Froze the accepted Round 2 version before experimentation.
- Added a paper-safe residual experiment that fits `crop + region + year_start` first.
- Added direct paper-safe and residual paper-safe comparisons.
- Tuned residual shrinkage `alpha` on validation years only.
- Wrote new Round 3 tables and figures without overwriting Round 2 outputs.

## Feature Counts

| group | n_features |
| --- | ---: |
| identity_time_only | {feature_counts['identity']} |
| direct_paper_safe | {feature_counts['direct']} |
| residual_paper_safe | {feature_counts['residual']} |

## Leakage Checks

- No `production_kt`.
- No `area_000ha`.
- No lag-yield, rolling-yield, or expanding-yield features in paper-safe features.
- Test years 2017-2021 are not used to tune residual shrinkage.
""",
        encoding="utf-8",
    )

    (REPORT_DIR / "03_ROUND3_RESULTS_AND_DECISION.md").write_text(
        f"""# Round 3 Results And Decision

## Headline

{headline}

## Decision

{decision}

## Best Per Window And Experiment

{markdown_table(result.sort_values(['forecast_window', 'experiment']))}

## Best Overall Rows

{markdown_table(comparison.sort_values('RMSE'))}

## Interpretation For Paper

This test is intentionally conservative. It asks whether paper-safe weather/anomaly/soil information explains the residual left after a crop/region/year baseline. If the residual model does not beat the identity/time baseline, the manuscript should avoid claiming that weather alone drives most predictive skill. It can still report weather/anomaly/soil as monitoring evidence, while presenting lag-yield history as an operational forecasting enhancement.
""",
        encoding="utf-8",
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PANEL_PATH)

    weather_cols, anomaly_cols, soil_cols, direct_cols = select_feature_groups(df)
    residual_cols = weather_cols + anomaly_cols + soil_cols
    feature_counts = {
        "identity": len(IDENTITY_FEATURES),
        "direct": len(direct_cols),
        "residual": len(residual_cols),
    }

    rows: list[dict[str, object]] = []
    model_names = ["Ridge", "ElasticNet"]
    if LGBMRegressor is not None:
        model_names.append("LightGBM")

    for window in FORECAST_WINDOWS:
        panel = df[df["forecast_window"] == window].copy()
        train = panel["year_start"].between(panel["year_start"].min(), TRAIN_END)
        val = panel["year_start"].between(VAL_START, VAL_END)
        test = panel["year_start"].between(TEST_START, TEST_END)
        y_train = panel.loc[train, "yield_t_ha"].to_numpy()
        y_val = panel.loc[val, "yield_t_ha"].to_numpy()
        y_test = panel.loc[test, "yield_t_ha"].to_numpy()
        n_train, n_val, n_test = int(train.sum()), int(val.sum()), int(test.sum())

        identity_model = make_model_pipeline(IDENTITY_FEATURES, ["crop", "region"], "Ridge")
        identity_model.fit(panel.loc[train, IDENTITY_FEATURES], y_train)
        base_val = identity_model.predict(panel.loc[val, IDENTITY_FEATURES])
        base_test = identity_model.predict(panel.loc[test, IDENTITY_FEATURES])
        rows.extend(
            metric_rows(
                window,
                "identity_time_only",
                "Ridge",
                y_test,
                base_test,
                alpha=None,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
            )
        )

        for model_name in model_names:
            direct_model = make_model_pipeline(direct_cols, ["crop", "region"], model_name)
            direct_model.fit(panel.loc[train, direct_cols], y_train)
            direct_pred = direct_model.predict(panel.loc[test, direct_cols])
            rows.extend(
                metric_rows(
                    window,
                    "direct_paper_safe",
                    model_name,
                    y_test,
                    direct_pred,
                    alpha=None,
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_test,
                )
            )

            residual_model = make_model_pipeline(residual_cols, [], model_name)
            residual_train = y_train - identity_model.predict(panel.loc[train, IDENTITY_FEATURES])
            residual_model.fit(panel.loc[train, residual_cols], residual_train)
            residual_val = residual_model.predict(panel.loc[val, residual_cols])
            residual_test = residual_model.predict(panel.loc[test, residual_cols])
            alpha = choose_alpha(y_val, base_val, residual_val)
            residual_pred = base_test + alpha * residual_test
            rows.extend(
                metric_rows(
                    window,
                    "residual_paper_safe",
                    model_name,
                    y_test,
                    residual_pred,
                    alpha=float(alpha),
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_test,
                )
            )

    metrics = pd.DataFrame(rows)
    comparison = pivot_metrics(metrics)
    metrics.to_csv(TABLE_DIR / "round3_paper_safe_residual_metrics.csv", index=False)
    comparison.to_csv(TABLE_DIR / "round3_paper_safe_residual_comparison.csv", index=False)
    make_figures(comparison)
    write_reports(metrics, comparison, feature_counts)
    print(f"Wrote Round 3 residual experiment outputs to {REPORT_DIR}")


if __name__ == "__main__":
    main()
