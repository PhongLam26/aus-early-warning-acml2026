from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from common import configured_path, ensure_output_dirs, load_config, markdown_table, write_markdown


WINDOW_ORDER = ["May-Jun", "May-Jul", "May-Aug", "May-Sep", "May-Oct"]
FEATURE_LABELS = {
    "weather_anomaly_soil_no_lag": "No-yield-history weather-soil",
    "full_operational": "Operational with yield history",
}
REVISION_DIR_NAME = "paper_revision_support_2026_06_24"


def rmse(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_linear_trend(group: pd.DataFrame) -> tuple[float, float] | None:
    if group["year_start"].nunique() < 3:
        return None
    slope, intercept = np.polyfit(group["year_start"].astype(float), group["yield_t_ha"].astype(float), deg=1)
    return float(slope), float(intercept)


def predict_from_trend(row: Any, crop_region_trends: dict[tuple[str, str], tuple[float, float]], crop_trends: dict[str, tuple[float, float]], overall_mean: float) -> float:
    key = (str(row.crop), str(row.region))
    crop = str(row.crop)
    year = float(row.year_start)
    if key in crop_region_trends:
        slope, intercept = crop_region_trends[key]
        return float(slope * year + intercept)
    if crop in crop_trends:
        slope, intercept = crop_trends[crop]
        return float(slope * year + intercept)
    return overall_mean


def unique_yield_panel(panel: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "region",
        "crop",
        "year_start",
        "split",
        "yield_t_ha",
        "expected_yield_t_ha",
        "yield_shortfall",
        "low_yield_risk",
        "low_yield_shortfall_threshold",
    ]
    return panel[keep].drop_duplicates(["region", "crop", "year_start"]).sort_values(["region", "crop", "year_start"]).reset_index(drop=True)


def add_naive_predictions(unique_obs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    train_end = int(config["splits"]["train_end"])
    train = unique_obs[unique_obs["year_start"] <= train_end].copy()
    global_mean = float(train["yield_t_ha"].mean())
    crop_mean = train.groupby("crop")["yield_t_ha"].mean().to_dict()
    crop_region_mean = train.groupby(["crop", "region"])["yield_t_ha"].mean().to_dict()

    crop_trends = {
        str(crop): trend
        for crop, group in train.groupby("crop")
        if (trend := fit_linear_trend(group)) is not None
    }
    crop_region_trends = {
        (str(crop), str(region)): trend
        for (crop, region), group in train.groupby(["crop", "region"])
        if (trend := fit_linear_trend(group)) is not None
    }

    out = unique_obs.copy()
    fallback = [
        crop_region_mean.get((row.crop, row.region), crop_mean.get(row.crop, global_mean))
        for row in out.itertuples(index=False)
    ]
    out["baseline_crop_region_train_mean"] = fallback
    out["baseline_crop_region_train_trend"] = [
        predict_from_trend(row, crop_region_trends, crop_trends, global_mean)
        for row in out.itertuples(index=False)
    ]

    parts = []
    for _, group in out.groupby(["region", "crop"], sort=False):
        group = group.sort_values("year_start").copy()
        group["baseline_previous_year_yield"] = group["yield_t_ha"].shift(1)
        group["baseline_rolling3_past_mean"] = group["yield_t_ha"].shift(1).rolling(window=3, min_periods=1).mean()
        parts.append(group)
    out = pd.concat(parts, ignore_index=True)
    for col in ["baseline_previous_year_yield", "baseline_rolling3_past_mean"]:
        out[col] = out[col].fillna(out["baseline_crop_region_train_mean"])
    return out


def make_naive_baselines(panel: pd.DataFrame, config: dict[str, Any], tables_dir: Path) -> pd.DataFrame:
    unique_obs = add_naive_predictions(unique_yield_panel(panel), config)
    expanded = panel[["region", "crop", "year_start", "forecast_window", "split", "yield_t_ha"]].merge(
        unique_obs[
            [
                "region",
                "crop",
                "year_start",
                "baseline_crop_region_train_mean",
                "baseline_previous_year_yield",
                "baseline_rolling3_past_mean",
                "baseline_crop_region_train_trend",
            ]
        ],
        on=["region", "crop", "year_start"],
        how="left",
        validate="many_to_one",
    )
    model_labels = {
        "baseline_crop_region_train_mean": "Crop-region train mean",
        "baseline_previous_year_yield": "Previous-year yield",
        "baseline_rolling3_past_mean": "3-year rolling past mean",
        "baseline_crop_region_train_trend": "Crop-region train trend",
    }
    rows = []
    for window in WINDOW_ORDER:
        subset = expanded[(expanded["split"] == "test") & (expanded["forecast_window"] == window)].copy()
        for col, label in model_labels.items():
            rows.append(
                {
                    "forecast_window": window,
                    "baseline": label,
                    "MAE": mean_absolute_error(subset["yield_t_ha"], subset[col]),
                    "RMSE": rmse(subset["yield_t_ha"], subset[col]),
                    "R2": r2_score(subset["yield_t_ha"], subset[col]),
                    "n_test": int(len(subset)),
                    "information_rule": "train-only" if "train" in col else "past-only chronological",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "paper_revision_naive_baselines.csv", index=False)
    return result


def make_fixed_model_lead_time(tables_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(tables_dir / "round2_ablation_metrics.csv")
    subset = metrics[
        (metrics["protocol"] == "time_split")
        & (metrics["fold"] == "primary")
        & (metrics["target"] == "yield_t_ha")
        & (metrics["task"] == "regression")
        & (metrics["feature_set"].isin(["weather_anomaly_soil_no_lag", "full_operational"]))
        & (metrics["model"].isin(["Ridge", "LightGBM"]))
        & (metrics["metric"].isin(["MAE", "RMSE", "R2"]))
    ].copy()
    pivot = (
        subset.pivot_table(
            index=["forecast_window", "feature_set", "model", "n_test"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["window_order"] = pivot["forecast_window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    pivot["feature_label"] = pivot["feature_set"].map(FEATURE_LABELS)
    pivot = pivot.sort_values(["feature_set", "model", "window_order"]).drop(columns=["window_order"])
    pivot.to_csv(tables_dir / "paper_revision_fixed_model_lead_time.csv", index=False)
    return pivot


def make_threshold_sensitivity(panel: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_obs = unique_yield_panel(panel)
    train = unique_obs[unique_obs["split"] == "train"].copy()
    detail_rows = []
    summary_rows = []
    for percentile in [70, 80, 90]:
        thresholds = train.groupby("crop")["yield_shortfall"].quantile(percentile / 100.0).to_dict()
        temp = unique_obs.copy()
        temp["threshold"] = temp["crop"].map(thresholds)
        temp["risk_at_percentile"] = (temp["yield_shortfall"] >= temp["threshold"]).astype(int)
        for crop, threshold in thresholds.items():
            crop_df = temp[temp["crop"] == crop]
            detail_rows.append(
                {
                    "percentile": percentile,
                    "crop": crop,
                    "shortfall_threshold": threshold,
                    "train_rate": crop_df.loc[crop_df["split"] == "train", "risk_at_percentile"].mean(),
                    "validation_rate": crop_df.loc[crop_df["split"] == "validation", "risk_at_percentile"].mean(),
                    "test_rate": crop_df.loc[crop_df["split"] == "test", "risk_at_percentile"].mean(),
                }
            )
        split_rates = temp.groupby("split")["risk_at_percentile"].mean().to_dict()
        split_counts = temp.groupby("split")["risk_at_percentile"].size().to_dict()
        summary_rows.append(
            {
                "percentile": percentile,
                "train_rate": split_rates.get("train", np.nan),
                "validation_rate": split_rates.get("validation", np.nan),
                "test_rate": split_rates.get("test", np.nan),
                "n_train": int(split_counts.get("train", 0)),
                "n_validation": int(split_counts.get("validation", 0)),
                "n_test": int(split_counts.get("test", 0)),
            }
        )
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(tables_dir / "paper_revision_threshold_sensitivity.csv", index=False)
    summary.to_csv(tables_dir / "paper_revision_threshold_sensitivity_summary.csv", index=False)
    return detail, summary


def selected_may_oct_classifier(tables_dir: Path) -> pd.Series:
    metrics = pd.read_csv(tables_dir / "classification_metrics_tuned_thresholds.csv")
    may_oct = metrics[metrics["forecast_window"] == "May-Oct"].copy()
    return may_oct.sort_values(["test_f1", "test_recall", "test_pr_auc"], ascending=False).iloc[0]


def make_confusion_and_watch_list(panel: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected_may_oct_classifier(tables_dir)
    predictions = pd.read_csv(tables_dir / "improved_test_predictions.csv")
    cls = predictions[
        (predictions["prediction_type"] == "classification")
        & (predictions["forecast_window"] == selected["forecast_window"])
        & (predictions["model"] == selected["model"])
    ].copy()
    threshold = float(selected["threshold"])
    cls["predicted_alert"] = (cls["y_proba"] >= threshold).astype(int)
    y_true = cls["y_true"].astype(int)
    y_pred = cls["predicted_alert"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    confusion = pd.DataFrame(
        [
            {
                "forecast_window": selected["forecast_window"],
                "model": selected["model"],
                "strategy": selected["strategy"],
                "threshold": threshold,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "n_test": int(len(cls)),
            }
        ]
    )

    panel_cols = [
        "row_id",
        "region",
        "crop",
        "year_start",
        "forecast_window",
        "yield_t_ha",
        "expected_yield_t_ha",
        "yield_shortfall",
        "low_yield_risk",
    ]
    watch = cls.merge(panel[panel_cols], on=["row_id", "forecast_window"], how="left", validate="one_to_one")
    watch = watch.drop(columns=[c for c in ["y_low", "y_high"] if c in watch.columns])
    interval = predictions[
        (predictions["prediction_type"] == "interval")
        & (predictions["forecast_window"] == selected["forecast_window"])
        & (predictions["model"].astype(str).str.startswith("ConformalBestPoint"))
    ][["row_id", "forecast_window", "y_low", "y_high"]].copy()
    interval["interval_width"] = interval["y_high"] - interval["y_low"]
    watch = watch.merge(interval, on=["row_id", "forecast_window"], how="left", validate="one_to_one")
    watch = watch.sort_values(["y_proba", "yield_shortfall"], ascending=False).head(10).copy()
    watch.insert(0, "rank", range(1, len(watch) + 1))
    watch["suggested_use"] = "Analyst review watch-list"
    watch = watch[
        [
            "rank",
            "region",
            "crop",
            "year_start",
            "forecast_window",
            "y_proba",
            "predicted_alert",
            "low_yield_risk",
            "yield_t_ha",
            "expected_yield_t_ha",
            "yield_shortfall",
            "y_low",
            "y_high",
            "interval_width",
            "suggested_use",
        ]
    ]
    confusion.to_csv(tables_dir / "paper_revision_confusion_matrix.csv", index=False)
    watch.to_csv(tables_dir / "paper_revision_watch_list_top10.csv", index=False)
    return confusion, watch


def make_uncertainty_summary(tables_dir: Path) -> pd.DataFrame:
    intervals = pd.read_csv(tables_dir / "improved_interval_summary.csv")
    conformal = intervals[intervals["model"].astype(str).str.startswith("ConformalBestPoint")].copy()
    class_metrics = pd.read_csv(tables_dir / "classification_metrics_tuned_thresholds.csv")
    best_class = (
        class_metrics.sort_values(["forecast_window", "test_f1", "test_recall", "test_pr_auc"], ascending=[True, False, False, False])
        .groupby("forecast_window", as_index=False)
        .head(1)
        [["forecast_window", "model", "strategy", "threshold", "test_brier", "test_roc_auc", "test_pr_auc"]]
        .rename(columns={"model": "classification_model"})
    )
    summary = conformal.merge(best_class, on="forecast_window", how="left")
    summary["nominal_coverage"] = 0.80
    summary["window_order"] = summary["forecast_window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    summary = summary.sort_values("window_order").drop(columns=["window_order"])
    summary.to_csv(tables_dir / "paper_revision_uncertainty_summary.csv", index=False)
    return summary


def make_split_importance_figure(tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    importance = pd.read_csv(tables_dir / "round2_feature_group_importance.csv")
    keep = importance[importance["feature_set"].isin(["full_operational", "weather_anomaly_soil_no_lag"])].copy()
    summary = (
        keep.groupby(["feature_set", "feature_group"], as_index=False)
        .agg(mean_rmse_delta=("rmse_delta_mean", "mean"), max_rmse_delta=("rmse_delta_mean", "max"), n_windows=("forecast_window", "nunique"))
        .sort_values(["feature_set", "mean_rmse_delta"], ascending=[True, False])
    )
    group_labels = {"weather_anomaly": "weather_dev"}
    summary["feature_group"] = summary["feature_group"].replace(group_labels)
    summary["feature_label"] = summary["feature_set"].map(FEATURE_LABELS)
    csv_summary = summary.copy()
    csv_summary["feature_set"] = csv_summary["feature_label"]
    csv_summary.to_csv(tables_dir / "paper_revision_feature_importance_split.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False)
    colors = {"full_operational": "#22577a", "weather_anomaly_soil_no_lag": "#b56576"}
    for ax, feature_set in zip(axes, ["full_operational", "weather_anomaly_soil_no_lag"]):
        plot_df = summary[summary["feature_set"] == feature_set].sort_values("mean_rmse_delta", ascending=True).tail(7)
        ax.barh(plot_df["feature_group"], plot_df["mean_rmse_delta"], color=colors[feature_set])
        ax.set_title(FEATURE_LABELS[feature_set])
        ax.set_xlabel("Mean RMSE increase")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Feature Group Importance By Modeling Regime")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig19_feature_importance_split.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def write_revision_report(report_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    revision_dir = report_dir.parent / REVISION_DIR_NAME
    revision_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Revision Support Log",
        "",
        "Generated support artifacts for the ACML Australia early-warning manuscript revision.",
        "",
        "## Output Summary",
        "",
    ]
    for name, df in outputs.items():
        lines.append(f"- `{name}`: {len(df)} rows")
    lines.extend(
        [
            "",
            "## Key Checks",
            "",
            "- Naive historical baselines use train-only means/trends or past-only chronological yield values.",
            "- Confusion matrix and watch-list use May-Oct validation-selected classification threshold only.",
            "- Uncertainty summary prioritizes conformal intervals and keeps coverage/width as diagnostics.",
            "- Feature-importance figure separates operational lag-history importance from no-yield-history weather-soil importance.",
            "",
            "## Preview: May-Oct Watch List",
            "",
            markdown_table(outputs["watch_list"].head(10).round(3)),
        ]
    )
    write_markdown(revision_dir / "01_PAPER_REVISION_SUPPORT_LOG.md", "\n".join(lines))


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    tables_dir = configured_path(config, "tables_dir")
    figures_dir = configured_path(config, "figures_dir")
    report_dir = configured_path(config, "reports_dir")
    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel_improved.csv")

    unique_obs = panel[["region", "crop", "year_start"]].drop_duplicates()
    if len(panel) != 4830 or len(unique_obs) != 966:
        raise RuntimeError(f"Unexpected panel shape: {len(panel)} rows and {len(unique_obs)} unique observations")

    naive = make_naive_baselines(panel, config, tables_dir)
    fixed = make_fixed_model_lead_time(tables_dir)
    threshold_detail, threshold_summary = make_threshold_sensitivity(panel, tables_dir)
    confusion, watch = make_confusion_and_watch_list(panel, tables_dir)
    uncertainty = make_uncertainty_summary(tables_dir)
    importance = make_split_importance_figure(tables_dir, figures_dir)
    outputs = {
        "naive_baselines": naive,
        "fixed_model_lead_time": fixed,
        "threshold_sensitivity": threshold_detail,
        "threshold_sensitivity_summary": threshold_summary,
        "confusion_matrix": confusion,
        "watch_list": watch,
        "uncertainty_summary": uncertainty,
        "feature_importance_split": importance,
    }
    write_revision_report(report_dir, outputs)
    print("Wrote paper revision support artifacts.")


if __name__ == "__main__":
    main()
