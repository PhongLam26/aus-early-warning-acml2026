from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import configured_path, ensure_output_dirs, load_config, write_markdown


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def window_order(config: dict) -> dict[str, int]:
    return {w["name"]: int(w["order"]) for w in config["forecast_windows"]}


def figure_framework(config: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    x = np.arange(len(months))
    ax.plot(x, np.zeros_like(x), color="#2f6f73", linewidth=3)
    ax.scatter(x, np.zeros_like(x), s=160, color="#f2c14e", edgecolor="#1f2933", zorder=3)
    for i, month in enumerate(months):
        ax.text(i, 0.15, month, ha="center", va="bottom", fontsize=11, fontweight="bold")
    for window in config["forecast_windows"]:
        end = int(window["end_month"]) - 5
        ax.annotate(
            window["name"],
            xy=(end, 0),
            xytext=(end, -0.38 - 0.06 * end),
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "#334e68"},
            fontsize=9,
        )
    ax.text(
        2.5,
        0.55,
        "Partial-season daily weather indicators feed probabilistic yield-shortfall risk models",
        ha="center",
        fontsize=11,
    )
    ax.set_ylim(-1.0, 1.0)
    ax.set_axis_off()
    save(fig, configured_path(config, "figures_dir") / "fig01_framework_timeline.png")


def figure_skill(metrics: pd.DataFrame, config: dict) -> None:
    order = window_order(config)
    rows = metrics[
        (metrics["target"] == "yield_t_ha")
        & (metrics["task"] == "regression")
        & (metrics["model"] == "HistGradientBoosting")
        & (metrics["metric"].isin(["RMSE", "R2"]))
    ].copy()
    rows["window_order"] = rows["forecast_window"].map(order)
    pivot = rows.pivot_table(index=["forecast_window", "window_order"], columns="metric", values="value").reset_index()
    pivot = pivot.sort_values("window_order")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    ax1.plot(pivot["forecast_window"], pivot["RMSE"], marker="o", color="#ba4a00", label="RMSE")
    ax1.set_ylabel("RMSE", color="#ba4a00")
    ax1.tick_params(axis="y", labelcolor="#ba4a00")
    ax1.set_xlabel("Forecast window")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(pivot["forecast_window"], pivot["R2"], marker="s", color="#22577a", label="R2")
    ax2.set_ylabel("R2", color="#22577a")
    ax2.tick_params(axis="y", labelcolor="#22577a")
    ax1.set_title("Forecast Skill By Lead Time")
    save(fig, configured_path(config, "figures_dir") / "fig02_forecast_skill_by_lead_time.png")


def figure_calibration(predictions: pd.DataFrame, config: dict) -> None:
    subset = predictions[
        (predictions["prediction_type"] == "classification")
        & (predictions["model"].isin(["HistGradientBoostingClassifier", "LogisticRegression", "HistoricalRiskRate"]))
    ].copy()
    if "HistGradientBoostingClassifier" in set(subset["model"]):
        subset = subset[subset["model"] == "HistGradientBoostingClassifier"]
    elif "LogisticRegression" in set(subset["model"]):
        subset = subset[subset["model"] == "LogisticRegression"]
    subset = subset.dropna(subset=["y_proba"])
    subset["bin"] = pd.cut(subset["y_proba"], bins=np.linspace(0, 1, 6), include_lowest=True)
    cal = subset.groupby("bin", observed=True).agg(predicted=("y_proba", "mean"), observed=("y_true", "mean"), n=("y_true", "size")).reset_index()

    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7b8794", label="Perfect calibration")
    ax.scatter(cal["predicted"], cal["observed"], s=70 + cal["n"] * 2, color="#2f6f73", edgecolor="#102a43")
    ax.plot(cal["predicted"], cal["observed"], color="#2f6f73", alpha=0.65)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted low-yield probability")
    ax.set_ylabel("Observed low-yield frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    save(fig, configured_path(config, "figures_dir") / "fig03_calibration_curve.png")


def figure_interval(predictions: pd.DataFrame, panel: pd.DataFrame, config: dict) -> None:
    subset = predictions[
        (predictions["prediction_type"] == "interval")
        & (predictions["model"] == "ConformalHGB")
        & (predictions["forecast_window"] == "May-Aug")
    ].copy()
    if subset.empty:
        subset = predictions[
            (predictions["prediction_type"] == "interval") & (predictions["model"] == "ConformalHGB")
        ].copy()
    meta = panel[["row_id", "crop", "region", "year_start"]]
    subset = subset.merge(meta, on="row_id", how="left").sort_values("y_true").head(45)
    labels = [f"{r.crop[:2]}-{r.region.split()[0][:2]}-{int(r.year_start)}" for r in subset.itertuples()]
    yerr = np.vstack([subset["y_pred"] - subset["y_low"], subset["y_high"] - subset["y_pred"]])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.errorbar(np.arange(len(subset)), subset["y_pred"], yerr=yerr, fmt="o", color="#22577a", ecolor="#8ecae6", capsize=2, label="Conformal interval")
    ax.scatter(np.arange(len(subset)), subset["y_true"], color="#ba4a00", s=18, label="Observed yield")
    ax.set_xticks(np.arange(len(subset))[::3])
    ax.set_xticklabels(labels[::3], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Yield t/ha")
    ax.set_title("Prediction Intervals For Test Crop-State-Years")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, configured_path(config, "figures_dir") / "fig04_prediction_interval_plot.png")


def figure_vulnerability(panel: pd.DataFrame, config: dict) -> None:
    base = panel.drop_duplicates(["region", "crop", "year_start"])
    pivot = (
        base.groupby(["crop", "region"])["low_yield_risk"]
        .mean()
        .unstack("region")
        .reindex(index=config["crops"], columns=config["regions"])
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=max(0.4, float(np.nanmax(pivot.values))))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Observed Crop-State Low-Yield Vulnerability")
    fig.colorbar(im, ax=ax, label="Low-yield rate")
    save(fig, configured_path(config, "figures_dir") / "fig05_crop_state_vulnerability_matrix.png")


def figure_response_curves(panel: pd.DataFrame, config: dict) -> None:
    subset = panel[panel["forecast_window"] == "May-Oct"].copy()
    features = ["rain_sum", "tmax_max", "max_consecutive_dry_days"]
    titles = ["May-Oct rainfall", "Maximum Tmax", "Longest dry spell"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, feature, title in zip(axes, features, titles):
        temp = subset[[feature, "low_yield_risk"]].dropna().copy()
        temp["bin"] = pd.qcut(temp[feature], q=8, duplicates="drop")
        curve = temp.groupby("bin", observed=True).agg(x=(feature, "mean"), risk=("low_yield_risk", "mean")).reset_index()
        ax.plot(curve["x"], curve["risk"], marker="o", color="#2f6f73")
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel("Observed low-yield rate")
        ax.grid(alpha=0.25)
    save(fig, configured_path(config, "figures_dir") / "fig06_response_curves.png")


def figure_dashboard(predictions: pd.DataFrame, panel: pd.DataFrame, config: dict) -> None:
    subset = predictions[
        (predictions["prediction_type"] == "classification")
        & (predictions["model"].isin(["HistGradientBoostingClassifier", "LogisticRegression"]))
        & (predictions["forecast_window"] == "May-Aug")
    ].copy()
    if "HistGradientBoostingClassifier" in set(subset["model"]):
        subset = subset[subset["model"] == "HistGradientBoostingClassifier"]
    meta = panel[["row_id", "crop", "region", "year_start"]]
    subset = subset.merge(meta, on="row_id", how="left")
    latest = subset[subset["year_start"] == subset["year_start"].max()].copy()
    ranked = latest.sort_values("y_proba", ascending=False).head(12)
    labels = [f"{r.crop} | {r.region}" for r in ranked.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(labels[::-1], ranked["y_proba"].iloc[::-1], color="#e76f51")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted low-yield probability")
    ax.set_title(f"Decision-Support Risk Snapshot: May-Aug {int(latest['year_start'].max())}")
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.02,
        -0.16,
        "State-level monitoring only. Use with agronomic expertise; not a farm-level decision engine.",
        transform=ax.transAxes,
        fontsize=9,
        color="#334e68",
    )
    save(fig, configured_path(config, "figures_dir") / "fig07_dashboard_mockup.png")


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)

    metrics = pd.read_csv(configured_path(config, "tables_dir") / "model_metrics_by_window.csv")
    predictions = pd.read_csv(configured_path(config, "tables_dir") / "test_predictions.csv")
    panel = pd.read_csv(configured_path(config, "processed_dir") / "model_ready_panel.csv")

    figure_framework(config)
    figure_skill(metrics, config)
    figure_calibration(predictions, config)
    figure_interval(predictions, panel, config)
    figure_vulnerability(panel, config)
    figure_response_curves(panel, config)
    figure_dashboard(predictions, panel, config)

    write_markdown(
        configured_path(config, "reports_dir") / "figures_report.md",
        """
# Figures Report

Generated figures:

- `fig01_framework_timeline.png`
- `fig02_forecast_skill_by_lead_time.png`
- `fig03_calibration_curve.png`
- `fig04_prediction_interval_plot.png`
- `fig05_crop_state_vulnerability_matrix.png`
- `fig06_response_curves.png`
- `fig07_dashboard_mockup.png`

These figures are designed for the Australia early-warning paper direction rather than the post-hoc anomaly explanation style of the U.S. paper.
""",
    )
    print(f"Wrote figures to {configured_path(config, 'figures_dir')}")


if __name__ == "__main__":
    main()
