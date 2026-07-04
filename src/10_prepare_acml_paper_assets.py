from __future__ import annotations

import csv
import re
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_TEMPLATE = PROJECT_ROOT / "ACML_camera_ready.zip"
PAPER_DIR = PROJECT_ROOT / "paper_acml"
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"
OUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SAFE_DIR = PROJECT_ROOT / "outputs" / "safe_versions" / "round2_safe_2026_06_24"
REFERENCE_DOCX = PROJECT_ROOT / "Tong_hop_ref_bai_Uc_Early_Warning.docx"


WINDOW_ORDER = ["May-Jun", "May-Jul", "May-Aug", "May-Sep", "May-Oct"]
FEATURE_LABELS = {
    "identity_time_only": "Identity+Time",
    "weather_stage_only": "Weather",
    "weather_plus_anomaly": "Weather+Dev",
    "weather_plus_soil": "Weather+Soil",
    "weather_anomaly_soil_no_lag": "No-yield-history weather-soil",
    "lag_yield_only": "Yield History",
    "full_operational": "Operational with yield history",
}
BASELINE_REGIME_LABELS = {
    "past_only_yield_history": "Past-only yield history",
    "no_yield_history_weather_soil": "No-yield-history weather-soil",
    "operational_with_yield_history": "Operational with yield history",
    "daily_weather_sequence_no_yield_history": "Daily sequence no-yield",
    "daily_weather_sequence_operational": "Daily sequence + yield history",
}
PROTOCOL_LABELS = {
    "time_split": "Held-out test years, all-window mean",
    "rolling_origin": "Rolling-origin",
    "leave_one_region_out": "Leave-one-region-out",
    "leave_one_crop_out": "Leave-one-crop-out",
}


def ensure_dirs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_tabular(path: Path, colspec: str, headers: list[str], rows: list[list[object]]) -> None:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_longtable(path: Path, colspec: str, headers: list[str], rows: list[list[object]]) -> None:
    header = " & ".join(latex_escape(h) for h in headers) + r" \\"
    lines = [
        r"\tiny",
        f"\\begin{{longtable}}{{{colspec}}}",
        r"\caption{Compact internal baseline-suite summary by forecast window and feature regime. Full metrics are retained in the accompanying CSV artifacts.}\label{tab:baselines-full-appendix}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_template_files() -> None:
    with zipfile.ZipFile(ZIP_TEMPLATE) as zf:
        for inner, out_name in {
            "ACML_camera_ready/jmlr.cls": "jmlr.cls",
            "ACML_camera_ready/acml26_submission_template.pdf": "acml26_submission_template.pdf",
        }.items():
            (PAPER_DIR / out_name).write_bytes(zf.read(inner))


def copy_figures() -> dict[str, str]:
    figure_map = {
        "fig01_australia_study_regions.pdf": "fig01_australia_study_regions.pdf",
        "fig01_australia_study_regions.png": "fig01_australia_study_regions.png",
        "fig16_paper_safe_vs_operational_model.png": "fig16_paper_safe_vs_operational_model.png",
        "fig13_ablation_r2_delta_by_feature_set.png": "fig13_ablation_r2_delta_by_feature_set.png",
        "fig14_stress_validation_heatmap.png": "fig14_stress_validation_heatmap.png",
        "fig19_feature_importance_split.png": "fig19_feature_importance_split.png",
        "fig20_baseline_comparison_rmse_r2.png": "fig20_baseline_comparison_rmse_r2.png",
    }
    for stale_png in FIG_DIR.glob("*.png"):
        stale_png.unlink()
    for stale_pdf in FIG_DIR.glob("*.pdf"):
        stale_pdf.unlink()
    copied: dict[str, str] = {}
    for source_name, target_name in figure_map.items():
        src = OUT_FIG_DIR / source_name
        dst = FIG_DIR / target_name
        if src.exists():
            shutil.copy2(src, dst)
            copied[source_name] = target_name
    return copied


def make_data_summary_table() -> None:
    panel = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "model_ready_panel_improved.csv")
    obs = panel[["region", "crop", "year_start"]].drop_duplicates()
    split_counts = obs.merge(
        panel[["region", "crop", "year_start", "split"]].drop_duplicates(),
        on=["region", "crop", "year_start"],
        how="left",
    )["split"].value_counts()
    rows = [
        ["Analysis unit", "State/region, crop, harvest year"],
        ["Yield observations", f"{len(obs):,}"],
        ["Window-expanded panel rows", f"{len(panel):,}"],
        ["Years", f"{panel['year_start'].min()}--{panel['year_start'].max()}"],
        ["Crops", ", ".join(sorted(panel["crop"].unique()))],
        ["Regions", f"{panel['region'].nunique()} Australian states/territories"],
        ["Forecast windows", ", ".join(WINDOW_ORDER)],
        [
            "Split",
            f"train {int(split_counts.get('train', 0))}, validation {int(split_counts.get('validation', 0))}, test {int(split_counts.get('test', 0))}",
        ],
    ]
    write_tabular(TABLE_DIR / "table_data_summary.tex", "ll", ["Item", "Value"], rows)


def make_lead_time_table() -> None:
    df = best_baseline_lead_time()
    compact_regime_labels = {
        "no_yield_history_weather_soil": "NYH weather-soil",
        "operational_with_yield_history": "Operational",
    }
    rows = [
        [
            row.forecast_window,
            compact_regime_labels.get(row.feature_regime, row.feature_regime),
            row.model,
            fmt(row.MAE),
            fmt(row.RMSE),
            fmt(row.R2),
        ]
        for row in df.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_lead_time.tex",
        "@{}lllrrr@{}",
        ["Window", "Feature set", "Model", "MAE", "RMSE", "R2"],
        rows,
    )


def best_baseline_lead_time() -> pd.DataFrame:
    df = pd.read_csv(OUT_TABLE_DIR / "sota_baseline_comparison.csv")
    keep = df[
        (df["status"] == "ok")
        & (df["feature_regime"].isin(["no_yield_history_weather_soil", "operational_with_yield_history"]))
    ].copy()
    keep["window_order"] = keep["forecast_window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    keep["feature_order"] = keep["feature_regime"].map({"no_yield_history_weather_soil": 0, "operational_with_yield_history": 1})
    return (
        keep.sort_values(["feature_regime", "forecast_window", "RMSE"])
        .groupby(["feature_regime", "forecast_window"], as_index=False)
        .head(1)
        .sort_values(["feature_order", "window_order"])
    )


def make_baseline_lead_time_figure() -> None:
    df = best_baseline_lead_time()
    df["label"] = df["feature_regime"].map(BASELINE_REGIME_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for label, group in df.groupby("label"):
        group = group.sort_values("window_order")
        axes[0].plot(group["forecast_window"], group["RMSE"], marker="o", label=label)
        axes[1].plot(group["forecast_window"], group["R2"], marker="o", label=label)
    axes[0].set_title("RMSE")
    axes[1].set_title("R2")
    for ax in axes:
        ax.set_xlabel("Forecast window")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_DIR / "fig16_paper_safe_vs_operational_model.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_study_region_map_figure() -> None:
    """Draw a compact state-level coverage schematic for the benchmark panel."""
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    regions = {
        "WA": {
            "poly": [(113.0, -35.5), (113.5, -25.0), (121.0, -13.8), (129.0, -13.8), (129.0, -35.5)],
            "label_xy": (121.0, -25.8),
            "text": "WA\n165 obs\n5 crops\ncomplete",
            "color": "#4c78a8",
        },
        "SA": {
            "poly": [(129.0, -35.5), (129.0, -26.0), (138.0, -26.0), (141.0, -31.5), (141.0, -38.5), (133.5, -38.0)],
            "label_xy": (135.0, -32.6),
            "text": "SA\n165 obs\n5 crops\ncomplete",
            "color": "#59a14f",
        },
        "QLD": {
            "poly": [(138.0, -10.5), (145.0, -10.5), (153.2, -23.0), (153.2, -29.0), (141.0, -29.0), (138.0, -26.0)],
            "label_xy": (146.4, -22.2),
            "text": "QLD\n153 obs\n5 crops\nmissing 12",
            "color": "#f28e2b",
        },
        "NSW": {
            "poly": [(141.0, -29.0), (153.2, -29.0), (151.4, -34.2), (146.0, -36.6), (141.0, -35.0)],
            "label_xy": (147.1, -32.8),
            "text": "NSW\n165 obs\n5 crops\ncomplete",
            "color": "#e15759",
        },
        "VIC": {
            "poly": [(141.0, -35.0), (146.0, -36.6), (150.0, -37.2), (149.2, -39.3), (142.0, -38.8)],
            "label_xy": (151.2, -37.9),
            "text": "VIC\n165 obs\n5 crops\ncomplete",
            "color": "#b07aa1",
            "external": True,
            "anchor_xy": (146.2, -37.8),
        },
        "TAS": {
            "poly": [(144.6, -40.2), (148.3, -40.4), (148.0, -43.6), (145.0, -43.8), (143.7, -42.2)],
            "label_xy": (151.2, -42.0),
            "text": "TAS\n153 obs\n5 crops\nmissing 12",
            "color": "#76b7b2",
            "external": True,
            "anchor_xy": (146.4, -42.1),
        },
    }
    unused = {
        "NT": [(129.0, -13.8), (138.0, -10.5), (138.0, -26.0), (129.0, -26.0)],
    }
    fig, ax = plt.subplots(figsize=(7.3, 4.45))
    ax.set_facecolor("#f7f7f5")
    for coords in unused.values():
        ax.add_patch(Polygon(coords, closed=True, facecolor="#dedede", edgecolor="white", linewidth=1.1, zorder=1))
    for spec in regions.values():
        ax.add_patch(Polygon(spec["poly"], closed=True, facecolor=spec["color"], edgecolor="white", linewidth=1.25, alpha=0.92, zorder=2))
    for spec in regions.values():
        if spec.get("external"):
            continue
        ax.text(
            spec["label_xy"][0],
            spec["label_xy"][1],
            spec["text"],
            ha="center",
            va="center",
            fontsize=7.6,
            color="white",
            fontweight="bold",
            linespacing=1.08,
            zorder=3,
        )
    for spec in regions.values():
        if not spec.get("external"):
            continue
        ax.annotate(
            spec["text"],
            xy=spec["anchor_xy"],
            xytext=spec["label_xy"],
            ha="left",
            va="center",
            fontsize=7.0,
            color="#222222",
            fontweight="bold",
            linespacing=1.05,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": spec["color"], "linewidth": 1.0},
            arrowprops={"arrowstyle": "-", "color": spec["color"], "linewidth": 1.0},
            zorder=4,
        )
    ax.text(133.4, -19.8, "not modeled", ha="center", va="center", fontsize=7, color="#666666", zorder=3)
    ax.text(112.7, -8.2, "State-level study regions", ha="left", va="bottom", fontsize=11, fontweight="bold", color="#222222")
    ax.text(
        112.7,
        -9.9,
        "966 observed region-crop-year entries, five winter crops, harvest years 1989-2021",
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#333333",
    )
    ax.set_xlim(111.5, 156.0)
    ax.set_ylim(-44.8, -7.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT_FIG_DIR / "fig01_australia_study_regions.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / "fig01_australia_study_regions.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_ablation_table() -> None:
    rows = [
        ["Identity/time only", "No", "No", "No", "Crop-region persistence benchmark"],
        ["Weather", "Yes", "No", "No", "Stage-weather signal only"],
        ["Weather+Dev", "Yes", "No", "No", "Adds train-derived anomaly features"],
        ["Weather+Soil", "Yes", "Yes", "No", "Adds static regional vulnerability"],
        ["No-yield-history weather-soil", "Yes", "Yes", "No", "Main information-isolation regime"],
        ["Yield history", "No", "No", "Yes", "Persistent productivity memory"],
        ["Operational with yield history", "Yes", "Yes", "Yes", "Main deployment-oriented regime"],
    ]
    write_tabular(
        TABLE_DIR / "table_ablation_compact.tex",
        "lllll",
        ["Feature set", "Weather", "Soil", "Yield hist.", "Interpretation"],
        rows,
    )


def make_validation_table() -> None:
    summary_path = OUT_TABLE_DIR / "round2_paper_table_stress_best_summary.csv"
    df = pd.read_csv(summary_path if summary_path.exists() else OUT_TABLE_DIR / "round2_paper_table_validation.csv")
    keep = df["feature_set"].isin(["weather_anomaly_soil_no_lag", "full_operational"])
    df = df[keep].copy()
    df["protocol_label"] = df["protocol"].map(PROTOCOL_LABELS)
    df["feature_label"] = df["feature_set"].map(FEATURE_LABELS)
    df = df.sort_values(["protocol", "feature_set", "mean_RMSE"])
    df = df.groupby(["protocol", "feature_set"], as_index=False, sort=False).head(1)
    rows = [
        [
            row.protocol_label,
            row.feature_label,
            row.model,
            fmt(row.mean_RMSE),
            fmt(row.mean_R2),
            int(row.folds),
        ]
        for row in df.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_validation.tex",
        "lllrrr",
        ["Protocol", "Feature set", "Model", "Mean RMSE", "Mean R2", "Folds"],
        rows,
    )


def make_classification_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "classification_metrics_tuned_thresholds.csv")
    best = (
        df.sort_values(["forecast_window", "test_f1"], ascending=[True, False])
        .groupby("forecast_window", as_index=False)
        .head(1)
    )
    best["window_order"] = best["forecast_window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    best = best.sort_values("window_order")
    rows = [
        [
            row.forecast_window,
            row.model.replace("Classifier", ""),
            str(row.strategy).replace("precision_ge_0_3", "precision ge 0.3").replace("recall_ge_0_5", "recall ge 0.5").replace("best_f1", "best F1"),
            fmt(row.threshold, 2),
            fmt(row.test_precision, 2),
            fmt(row.test_recall, 2),
            fmt(row.test_f1, 2),
            fmt(row.test_roc_auc, 2),
            fmt(row.test_pr_auc, 2),
            fmt(row.test_brier),
            fmt(row.test_positive_rate, 2),
        ]
        for row in best.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_classification_summary.tex",
        "lllrrrrrrrr",
        ["Window", "Model", "Strategy", "Thr.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "Pred.+"],
        rows,
    )
    table_path = TABLE_DIR / "table_classification_summary.tex"
    table_text = table_path.read_text(encoding="utf-8").replace("precision ge 0.3", r"precision $\geq$ 0.3")
    table_path.write_text(table_text, encoding="utf-8")


def make_round3_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "round3_paper_safe_residual_comparison.csv")
    best = (
        df.sort_values(["experiment", "RMSE"])
        .groupby("experiment", as_index=False)
        .head(1)
        .sort_values("RMSE")
    )
    labels = {
        "direct_paper_safe": "Direct weather-soil",
        "residual_paper_safe": "Residual weather-soil",
        "identity_time_only": "Identity+time baseline",
    }
    rows = [
        [
            labels.get(row.experiment, row.experiment),
            row.forecast_window,
            row.model,
            fmt(row.alpha, 2) if row.alpha >= 0 else "--",
            fmt(row.MAE),
            fmt(row.RMSE),
            fmt(row.R2),
        ]
        for row in best.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_round3_residual.tex",
        "lllrrrr",
        ["Experiment", "Window", "Model", "Alpha", "MAE", "RMSE", "R2"],
        rows,
    )


def make_feature_group_table() -> None:
    rows = [
        ["Rainfall/dryness", "rain_sum, rain_days, dry_days, dry-spell length", "Water supply and dry-spell stress"],
        ["Heat/cold", "tmax_mean, heat_days_30, heat degree days, frost days", "Heat exposure and frost/cold risk"],
        ["Energy/evaporation", "radiation_sum, evap_sum, vapor pressure", "Growth energy and evaporative demand"],
        ["Compound stress", "hot-dry days, high-evaporation dry days", "Combined weather stress"],
        ["Soil background", "AWC, clay/sand, SOC, pH, bulk density, depth", "Regional vulnerability conditioning"],
    ]
    write_tabular(
        TABLE_DIR / "table_feature_groups.tex",
        "lll",
        ["Feature group", "Examples", "Interpretation"],
        rows,
    )


def make_sota_baseline_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "sota_baseline_comparison.csv")
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("sota_baseline_comparison.csv has no successful rows")
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
    rows = []
    for model_group, regime in wanted:
        subset = ok[
            (ok["model_group"] == model_group)
            & (ok["feature_regime"] == regime)
            & (ok["forecast_window"] == "May-Oct")
        ].copy()
        if subset.empty:
            continue
        best = subset.sort_values("RMSE").iloc[0]
        rows.append(
            [
                model_group,
                BASELINE_REGIME_LABELS.get(regime, regime),
                best["model"],
                "Yes" if model_group == "Sequence" else "No",
                "Yes"
                if regime in {"past_only_yield_history", "operational_with_yield_history", "daily_weather_sequence_operational"}
                else "No",
                fmt(best["RMSE"]),
                fmt(best["R2"]),
            ]
        )
    write_tabular(
        TABLE_DIR / "table_naive_baselines.tex",
        "lllllrr",
        ["Group", "Regime", "Model", "Daily seq.", "Yield history", "May-Oct RMSE", "May-Oct R2"],
        rows,
    )
    appendix = ok.copy()
    appendix["window_order"] = appendix["forecast_window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    appendix = (
        appendix.sort_values(["feature_regime", "forecast_window", "RMSE"])
        .groupby(["feature_regime", "forecast_window"], as_index=False)
        .head(1)
        .sort_values(["feature_regime", "window_order"])
    )
    appendix_regime_labels = {
        "past_only_yield_history": "Past-only yield",
        "no_yield_history_weather_soil": "NYH weather-soil",
        "operational_with_yield_history": "Operational",
        "daily_weather_sequence_no_yield_history": "GRU no-yield",
        "daily_weather_sequence_operational": "GRU operational",
    }
    appendix_model_labels = {
        "HistGradientBoosting": "HistGB",
        "DailyWeather-GRU": "Weather-GRU",
        "Crop-region train mean": "CR mean",
        "Crop-region train trend": "CR trend",
        "Previous-year yield": "Prev-year",
        "3-year rolling mean": "Rolling3",
        "RandomForest": "RF",
        "ElasticNet": "ENet",
    }
    appendix_group_labels = {
        "Strong tabular ML": "Tabular ML",
        "Interpretable ML": "Additive ML",
    }
    appendix_rows = [
        [
            appendix_group_labels.get(row.model_group, row.model_group),
            appendix_regime_labels.get(row.feature_regime, row.feature_regime),
            row.forecast_window,
            appendix_model_labels.get(row.model, row.model),
            fmt(row.MAE, 2),
            fmt(row.RMSE, 2),
            fmt(row.R2, 2),
        ]
        for row in appendix.itertuples(index=False)
    ]
    write_longtable(
        TABLE_DIR / "table_baselines_full_appendix.tex",
        r"@{}p{0.12\textwidth}p{0.20\textwidth}p{0.08\textwidth}p{0.16\textwidth}rrr@{}",
        ["Group", "Regime", "Window", "Model", "MAE", "RMSE", "R2"],
        appendix_rows,
    )


def make_naive_baseline_table() -> None:
    make_sota_baseline_table()


def make_fixed_model_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "sota_baseline_comparison.csv")
    rows = []
    regimes = ["no_yield_history_weather_soil", "operational_with_yield_history"]
    for window in WINDOW_ORDER:
        row = [window]
        for model in ["Ridge", "LightGBM"]:
            for regime in regimes:
                subset = df[
                    (df["status"] == "ok")
                    & (df["forecast_window"] == window)
                    & (df["model"] == model)
                    & (df["feature_regime"] == regime)
                ]
                row.append(fmt(subset.iloc[0]["RMSE"], 2) if not subset.empty else "--")
        rows.append(row)
    write_tabular(
        TABLE_DIR / "table_fixed_model_lead_time.tex",
        "lrrrr",
        ["Window", "Ridge NYH", "Ridge Op.", "LGBM NYH", "LGBM Op."],
        rows,
    )


def make_threshold_sensitivity_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "paper_revision_threshold_sensitivity_summary.csv")
    rows = [
        [
            int(row.percentile),
            fmt(100 * row.train_rate, 1),
            fmt(100 * row.validation_rate, 1),
            fmt(100 * row.test_rate, 1),
        ]
        for row in df.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_threshold_sensitivity.tex",
        "rrrr",
        ["Percentile", "Train rate (%)", "Validation rate (%)", "Test rate (%)"],
        rows,
    )


def make_confusion_matrix_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "paper_revision_confusion_matrix.csv")
    rows = [
        [
            row.forecast_window,
            row.model,
            fmt(row.threshold, 2),
            int(row.TN),
            int(row.FP),
            int(row.FN),
            int(row.TP),
            fmt(row.precision),
            fmt(row.recall),
            fmt(row.F1),
        ]
        for row in df.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_confusion_matrix.tex",
        "lllrrrrrrr",
        ["Window", "Model", "Thr.", "TN", "FP", "FN", "TP", "Prec.", "Recall", "F1"],
        rows,
    )


def make_watch_list_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "paper_revision_watch_list_top10.csv")
    region_labels = {
        "New South Wales": "NSW",
        "Queensland": "QLD",
        "South Australia": "SA",
        "Tasmania": "TAS",
        "Victoria": "VIC",
        "Western Australia": "WA",
    }
    rows = [
        [
            int(row.rank),
            region_labels.get(row.region, row.region),
            row.crop,
            int(row.year_start),
            fmt(row.y_proba, 3),
            int(row.predicted_alert),
            int(row.low_yield_risk),
            fmt(row.yield_shortfall),
        ]
        for row in df.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_watch_list_top10.tex",
        "rllrrrrr",
        ["Rank", "Region", "Crop", "Year", "Rank score", "Alert", "Observed", "Shortfall"],
        rows,
    )


def make_missing_counts_table() -> None:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "yield" / "yield_panel.csv")
    years = range(int(raw["year_start"].min()), int(raw["year_start"].max()) + 1)
    rows = []
    for (region, crop), group in raw.groupby(["region", "crop"], sort=True):
        observed = set(group["year_start"].astype(int))
        rows.append([region, crop, len(observed), len(set(years) - observed)])
    total_expected = raw[["region", "crop"]].drop_duplicates().shape[0] * len(list(years))
    rows.append(["Total", "All crops", len(raw), total_expected - len(raw)])
    write_tabular(
        TABLE_DIR / "table_missing_counts.tex",
        "llrr",
        ["Region", "Crop", "Observed years", "Missing years"],
        rows,
    )


def make_uncertainty_table() -> None:
    df = pd.read_csv(OUT_TABLE_DIR / "paper_revision_uncertainty_summary.csv")
    rows = [
        [
            row.forecast_window,
            row.model.replace("ConformalBestPoint_", "Conformal "),
            fmt(row.nominal_coverage),
            fmt(row.coverage),
            fmt(row.width),
            fmt(row.qhat),
            fmt(row.test_brier),
        ]
        for row in df.itertuples(index=False)
    ]
    write_tabular(
        TABLE_DIR / "table_uncertainty_summary.tex",
        "llrrrrr",
        ["Window", "Interval model", "Nominal", "Coverage", "Width", "qhat", "Brier"],
        rows,
    )


def extract_bibtex_entries_from_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("python-docx is required to extract the ACML reference list") from exc

    document = docx.Document(str(path))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    entries: list[str] = []
    index = 0
    while True:
        start = text.find("@", index)
        if start == -1:
            break
        type_end = text.find("{", start)
        if type_end == -1 or not text[start + 1 : type_end].strip().replace("_", "").isalpha():
            index = start + 1
            continue

        depth = 0
        end = type_end
        for pos in range(type_end, len(text)):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    break
        block = text[start:end].strip()
        if block and block not in entries:
            entries.append(block)
        index = max(end, start + 1)

    if len(entries) < 25:
        raise RuntimeError(f"Expected the reference docx to contain about 30 BibTeX entries, found {len(entries)}")
    return "\n\n".join(entries)


def normalize_bibtex_accents(bib: str) -> str:
    replacements = {
        r"Fran{\c{c}}ois": r"Fran{\c c}ois",
        r"Herv{'e}": r"Herv{\'e}",
        r'M{"u}ller': r'M{\"u}ller',
    }
    for old, new in replacements.items():
        bib = bib.replace(old, new)
    return bib


def write_bibliography() -> None:
    bib = normalize_bibtex_accents(extract_bibtex_entries_from_docx(REFERENCE_DOCX))
    manual_entries = {
        "abares2026australianCropReport": r"""@misc{abares2026australianCropReport,
  title = {Australian Crop Report: State Crop Data Workbook},
  author = {{Australian Bureau of Agricultural and Resource Economics and Sciences}},
  year = {2026},
  note = {State-level crop statistics workbook \path{03_AustCropRrt20260303_StateCropData_v1.0.0.xlsx}, accessed 4 July 2026}
}""",
        "jeffrey2001silo": r"""@article{jeffrey2001silo,
  title = {Using spatial interpolation to construct a comprehensive archive of Australian climate data},
  author = {Jeffrey, Stephen J. and Carter, John O. and Moodie, Keith B. and Beswick, Alan R.},
  journal = {Environmental Modelling and Software},
  volume = {16},
  number = {4},
  pages = {309--330},
  year = {2001}
}""",
        "grundy2015slga": r"""@article{grundy2015slga,
  title = {Soil and Landscape Grid of Australia},
  author = {Grundy, Michael J. and Rossel, Raphael A. Viscarra and Searle, Ross D. and Wilson, Peter L. and Chen, Chen and Gregory, Luke J.},
  journal = {Soil Research},
  volume = {53},
  number = {8},
  pages = {835--844},
  year = {2015}
}""",
    }
    for key, entry in manual_entries.items():
        pattern = re.compile(r"@\w+\{" + re.escape(key) + r",.*?\n\}", re.DOTALL)
        if pattern.search(bib):
            bib = pattern.sub(lambda _match: entry, bib)
        else:
            bib = bib.rstrip() + "\n\n" + entry
    (PAPER_DIR / "acml26.bib").write_text(bib.strip() + "\n", encoding="utf-8")


def _write_main_tex_legacy() -> None:
    tex = r"""
\documentclass[wcp]{jmlr}

\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{lineno}

\pagenumbering{gobble}
\hypersetup{
  pageanchor=false,
  hidelinks,
  colorlinks=false,
  linkcolor=black,
  citecolor=black,
  urlcolor=black,
  filecolor=black,
  pdfborder={0 0 0}
}
\bibpunct{[}{]}{,}{n}{,}{,}
\setcitestyle{numbers,square,comma,sort&compress}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\newcommand{\cs}[1]{\texttt{\char`\\#1}}
\makeatletter
\let\Ginclude@graphics\@org@Ginclude@graphics
\setlength{\@fptop}{0pt}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother

\jmlryear{2026}
\jmlrworkshop{ACML 2026}
\jmlrvolume{}
\jmlrpages{}
\makeatletter
\renewcommand*{\@titlefoot}{}
\gdef\@editor{}
\let\cleardoublepage\clearpage
\def\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{}%
  \def\@evenhead{}%
  \def\@oddfoot{}%
  \def\@evenfoot{}%
}
\makeatother

\title[Stage-Aware Crop Yield Warning]{Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall Using Daily Weather and Soil Data}

\author{}

\begin{document}

\maketitle

\begin{abstract}
Early warning of crop yield shortfall is useful only when risk can be updated before the season is effectively complete and when forecast skill is not driven by post-harvest leakage. We study Australian winter crops at state level using 966 region-crop-year yield observations from 1989--2021, daily SILO weather, and Soil and Landscape Grid attributes. Daily weather is converted into May-Jun, May-Jul, May-Aug, May-Sep, and May-Oct feature windows. We separate a no-yield-history weather-soil setting from an operational yield-history setting to distinguish within-season environmental signal from persistent crop-region yield memory. On held-out 2017--2021 seasons, the best no-yield-history May-Oct internal-suite model reaches RMSE 0.821 t/ha and R2 0.579, while the best operational May-Oct model reaches RMSE 0.660 t/ha and R2 0.728. Ablation and internal baseline-suite comparisons show that weather and soil provide measurable but secondary monitoring information; persistent crop-region structure and lagged yield history account for much of the strongest operational skill. Risk classification and uncertainty diagnostics support the framework as a state-level monitoring tool for known crop-region histories, while clearly limiting causal, farm-level, and unseen-region claims.
\end{abstract}

\begin{keywords}
crop yield forecasting; early warning; climate risk; Australian agriculture; weather features; soil data; uncertainty
\end{keywords}

\section{Introduction}

Australian dryland grain systems face recurring climate risks, including rainfall deficits, heat events, frost, and periods of high evaporative demand. These risks are particularly important for winter crops, where the difference between early vegetative conditions and late-season grain-filling conditions can change the decision value of a forecast. Prior work has shown that Australian wheat yields have been affected by adverse climate trends and by drought-heat exposure \citep{hochman2017climateTrendsAustralia,feng2019droughtHeatAustralia}. For decision support, however, the operational question is not only whether yield can be estimated after the season, but whether risk can be updated at meaningful within-season stages.

This paper studies state-level early warning for Australian winter crops. The task is to predict yield and low-yield shortfall risk for each region, crop, harvest year, and forecast window. Unlike a post-hoc anomaly attribution or multi-method XAI study, the central question is operational timing: how early can yield-shortfall risk be monitored? A May-Jun estimate supports early monitoring, while a May-Oct estimate approaches a full-season pre-harvest assessment. This framing is aligned with recent early-warning work that emphasizes anticipatory crop-failure monitoring rather than post-season diagnosis alone \citep{anderson2024preseasonWarning}.

The empirical design is intentionally conservative. We remove production and area variables from the main predictors, compute expected-yield and weather-deviation baselines from training years only, and keep the 2017--2021 test period outside all scaling, thresholding, calibration, and residual estimation steps. This avoids turning post-harvest information into apparent forecast skill.

The main contribution is a stage-aware benchmark framework, not a new forecasting algorithm. First, a no-yield-history weather-soil model tests whether daily weather, train-derived weather deviations, and soil background provide measurable evidence of within-season monitoring signal. Second, an operational model adds lagged yield history to quantify how much forecast quality improves when historical production memory is allowed. This distinction matters because a strong operational forecast may partly reflect persistent crop-region productivity rather than purely within-season weather signal.

We make three contributions. First, we formulate Australian winter-crop forecasting as a lead-time-aware shortfall-monitoring benchmark, where predictions are updated from May-Jun to May-Oct rather than evaluated only after the season is nearly complete. Second, we separate a no-yield-history weather-soil setting from an operational yield-history setting, allowing weather-derived early-warning evidence to be distinguished from persistent crop-region productivity memory. Third, we evaluate practical usefulness through temporal holdout testing, internal baseline-suite comparison, feature-regime ablation, stress validation, threshold-tuned risk screening, and uncertainty diagnostics, framing the outputs as state-level decision support rather than causal or farm-level prediction. The machine-learning contribution is therefore the leakage-safe temporal evaluation protocol and benchmark design: staged information sets, explicit feature-regime separation, and stress tests for Australian winter-crop monitoring.

We organize the study around four research questions. RQ1 asks how much forecasting skill is available at successive monitoring windows from May-Jun to May-Oct. RQ2 asks how much of this skill comes from stage-aware weather and soil features compared with lagged yield history. RQ3 asks whether low-yield-risk predictions can support screening without becoming automatic decision triggers. RQ4 asks how robust the framework is under rolling-origin, leave-one-crop-out, and leave-one-region-out stress tests.

\section{Related Work}

\subsection{Crop Yield Forecasting and Early Warning}

Machine learning has become a common approach for regional crop-yield forecasting, with weather, soil, remote sensing, and management covariates appearing across many systems \citep{vanKlompenburg2020systematicReview,paudel2021largeScaleYield}. At the same time, several studies warn that yield datasets are often small relative to the dimensionality of environmental predictors, making regularized and tree-based tabular models attractive for grain forecasting \citep{meroni2021smallData}. Recent work also pushes yield modeling toward multi-modal benchmarks and operational decision-support interfaces \citep{lin2024cropnet,declercq2024indiaRice}. Our setting is smaller and more targeted: state-level Australian winter crops with a fixed May-Oct monitoring calendar.

Early-warning studies differ from generic yield prediction because lead time is part of the objective. A model that works only after most weather information is observed may have limited preparedness value. Preseason and early-season crop-failure forecasts show the importance of evaluating when skill emerges, not only how accurate the final estimate is \citep{anderson2024preseasonWarning}. We adopt this view by reporting skill separately for each forecast window.

\subsection{Stage-Aware Weather and Extreme-Event Features}

Weather-yield relationships are stage dependent. Developmental-stage studies show that climatic means and weather extremes can have different effects depending on when they occur during crop growth \citep{schierhorn2021developmentalStages}. Broader climate-risk evidence also emphasizes compound hot-dry extremes, crop-specific vulnerability, and heterogeneous regional responses \citep{heino2023hotDryExtremes,sjulgard2023swedenAnomalies}. We therefore aggregate daily weather into stage-aware rainfall, heat, cold, evaporative-demand, radiation, and compound-stress features rather than relying only on full-season averages. Recent extreme-weather yield studies further motivate sparse or regularized modeling of drought, heat, flooding, and compound hazards \citep{heilemann2024lassoExtremes,kabtih2025hotPluvial}.

\subsection{Interpretability, Uncertainty, and Decision Support}

For agricultural risk monitoring, the output is used to prioritize attention rather than to prove a mechanistic causal pathway. Interpretability and uncertainty are therefore practical requirements, especially when models inform public or business decisions \citep{rudin2022blackboxBrief,paudel2023interpretability}. We use transparent ablation, group permutation importance, and simple lead-time comparisons rather than making the paper a post-hoc XAI study. Explainable boosting and related glassbox tools provide one route for transparent tabular modeling \citep{nori2019interpretml}, while conformal and conformalized quantile methods motivate distribution-aware intervals with explicit coverage diagnostics \citep{romano2019cqr,angelopoulos2023conformal}.

\section{Data and Prediction Targets}

\subsection{Study Panel and Data Sources}

The analysis unit is a region-crop-year observation. The yield panel is derived from the ABARES Australian Crop Report state-crop data workbook \path{03_AustCropRrt20260303_StateCropData_v1.0.0.xlsx}, which reports crop area, production, and yield for major Australian crops at state or territory level; we cite it as the yield data source and used the workbook accessed on 4 July 2026 \citep{abares2026australianCropReport}. We use the raw ABARES workbook and a derived \texttt{yield\_panel.csv} after filtering to five winter crops and six state-level regions. Of the 990 possible state-crop-year combinations for 1989--2021, 966 are observed; the missing 24 source entries are summarized in the supplementary material. Table~\ref{tab:data-summary} summarizes the modeled panel. Each of the 966 yield observations is expanded over five forecast windows, giving 4,830 window rows. This expansion does not create new yield outcomes: models and metrics are evaluated within forecast windows, and all train/validation/test decisions are made by calendar year so the same region-crop-year outcome cannot be split across evaluation periods.

\begin{table}[htbp]
\centering
\caption{Data summary for the Australian winter crop early-warning panel.}
\label{tab:data-summary}
\input{tables/table_data_summary.tex}
\end{table}

Daily weather is taken from SILO/LongPaddock Australian climate records, which are based on spatial interpolation of observed weather data \citep{jeffrey2001silo}. The raw project tables retain daily state-level regional series with latitude and longitude metadata; the current benchmark aggregates these regional daily series by forecast cutoff as a state-level exposure proxy and does not claim crop-area-weighted gridded exposure. We use daily rainfall, maximum and minimum temperature, radiation, evaporation, and vapor pressure to build cutoff-specific summaries. Soil covariates are static regional background attributes from ASRIS/Soil and Landscape Grid of Australia \citep{grundy2015slga}. They include available water capacity, bulk density, clay, sand, silt, soil organic carbon, total nitrogen, total phosphorus, pH, cation exchange capacity, depth of soil, and depth of regolith, summarized as topsoil/subsoil or regional aggregates where available. Because these soil summaries are region-level and static, they are treated as background vulnerability covariates, not as farm-level measurements or causal soil interventions.

\subsection{Targets}

The regression target is yield in tonnes per hectare, denoted $y_{r,c,t}$. We also define an expected-yield baseline from training years and the shortfall
\[
  s_{r,c,t} = \hat{y}^{\mathrm{exp}}_{r,c,t} - y_{r,c,t}.
\]
The expected-yield baseline is fitted using training years only. In the main specification, expected yield is estimated from a crop-region linear trend in calendar year; crop-level linear trends are used when a crop-region series is too short, and the overall training mean is the final tier in this training-only hierarchy. This baseline excludes current-season weather, production, area, lagged yield, and all validation/test outcomes. Validation and test shortfalls are then computed by applying the fixed training-period baseline to later years. The binary low-yield-risk label is one when shortfall exceeds the crop-specific 80th percentile of the training distribution. We use this threshold as a broad watch-list definition rather than as a severe crop-failure definition. This target follows the general idea that yield anomalies or shortfalls require a reference trend before climate impacts can be assessed \citep{meng2024detrending}, but here the shortfall is used for early-warning classification rather than post-hoc causal attribution. The classification target is used for secondary risk-alert analyses; the main quantitative results remain regression and ablation metrics.

\section{Stage-Aware Features and Models}

\subsection{Weather Windows}

For each forecast window $w$, daily weather is aggregated only from May through the cutoff month of $w$. The design intentionally mimics an operational update: May-Jun uses only two months of weather, while May-Oct uses the longest pre-harvest window. No weather feature uses data after the cutoff month. This makes the lead-time comparison interpretable as a sequence of information sets rather than a single full-season model.

The weather feature groups are rainfall accumulation and dry spells, heat and cold exposure, radiation and evaporative demand, and compound hot-dry stress. Rainfall summaries include totals, rainy days, dry days, short-window maxima, heavy-rain days, and maximum consecutive dry days. Heat/cold features include mean and extreme maximum and minimum temperatures, heat-day thresholds, heat degree days, frost days, and cold days. Energy and dryness features include radiation, evaporation, and vapor pressure summaries. Compound features flag hot-dry days and high-evaporation dry days. These choices are motivated by stage-aware and extreme-weather yield studies \citep{schierhorn2021developmentalStages,heino2023hotDryExtremes,heilemann2024lassoExtremes}.

\begin{table}[htbp]
\centering
\caption{Feature groups used for stage-aware early-warning models.}
\label{tab:feature-groups}
\resizebox{\textwidth}{!}{\input{tables/table_feature_groups.tex}}
\end{table}

We additionally construct weather-deviation features by subtracting train-period region-window baselines. This produces rainfall, temperature, dry-spell, heat-day, evapotranspiration, and radiation deviations that are comparable across regions and forecast windows. The weather-deviation baselines are computed using training years only, so they do not leak test-period information.

\subsection{No-Yield-History and Operational Feature Sets}

We report two primary feature regimes. The no-yield-history weather-soil regime uses stage weather features, train-derived weather-deviation features, soil background features, crop identity, region identity, and year. It excludes lagged yield, rolling yield, production, and area; it is not free of crop, state-level region, or time structure. The operational with yield history regime adds past-yield features from prior years for the same crop-region series. This makes it useful for deployment in known crop-region histories, but it is not interpreted as pure weather evidence.

The model suite combines regularized linear baselines with gradient-boosted tabular models. Ridge and ElasticNet provide stable small-data baselines, while LightGBM and CatBoost provide nonlinear tabular benchmarks \citep{ke2017lightgbm,prokhorenkova2018catboost}. The expanded internal baseline suite also includes Random Forest, SVR-RBF, XGBoost, HistGradientBoosting, GAM, and a daily-weather GRU sequence comparator; implementation details are reported in the supplementary material. We avoid presenting a black-box-only result: every performance table is paired with feature-set ablation, lead-time comparison, or group permutation importance. This is consistent with the broader argument that high-stakes decision support should prefer transparent or at least carefully audited modeling workflows \citep{rudin2022blackboxBrief}.

Classification models use validation-tuned thresholds for low-yield-risk decisions. Instead of fixing a probability threshold at 0.5, we choose thresholds on validation years for F1, recall constraints, or precision constraints, then report test metrics once. Probabilistic interval diagnostics use quantile-style predictions and conformal calibration ideas as supplementary checks \citep{romano2019cqr,angelopoulos2023conformal}. These outputs are not framed as definitive event decisions; they are screening layers for state-level monitoring.

\section{Evaluation Protocol}

The main split trains on 1989--2012, tunes on 2013--2016, and tests on 2017--2021. The test period is not used for scalers, feature thresholds, residual calibration, conformal residuals, expected-yield baselines, weather-deviation baselines, or low-yield-risk thresholds. This temporal split reflects the intended forecasting direction: models learn from earlier seasons and are evaluated on later seasons.

Regression metrics are MAE, RMSE, and R2. Classification metrics are precision, recall, F1, ROC-AUC, PR-AUC, and Brier score. For probabilistic diagnostics, we summarize interval coverage and width. The conformal intervals target nominal 80\% marginal coverage using validation-period calibration residuals only; test coverage is reported without retuning. Deviations from nominal coverage are interpreted as evidence of calibration difficulty under temporal shift and small calibration samples, not as a target corrected using test outcomes. The primary model-selection comparisons are always made within the same forecast window and split.

Robustness checks include rolling-origin folds, leave-one-crop-out validation, and leave-one-region-out validation. Rolling-origin validation tests sensitivity to a single time split. Leave-one-crop-out and leave-one-region-out are intentionally difficult transfer tests. We do not treat them as the main deployment setting, because the intended use case is monitoring crop-region combinations with historical observations. Instead, they identify where the framework is not yet robust enough for unseen-crop or unseen-state generalization.

\section{Results}

\subsection{Lead-Time Skill}

Table~\ref{tab:lead-time} compares the best no-yield-history weather-soil and operational models at each lead time within the internal baseline suite. For each window and feature regime, the displayed model is selected using validation years only, with RMSE as the selection metric. The operational model is already informative at May-Jun, with RMSE 0.719 t/ha and R2 0.677, and remains strongest at May-Oct with RMSE 0.660 t/ha and R2 0.728. This early operational result may be driven substantially by lagged yield and crop-region memory, so it should not be read as early-season weather-only skill. The no-yield-history suite is weaker but still positive, reaching RMSE 0.821 t/ha and R2 0.579 at May-Oct with an SVR-RBF comparator. Its trajectory is not strictly monotonic: later windows add useful weather information but also correlated features and noise in a small state-level panel. We therefore interpret the curve as a monitoring trajectory rather than assuming that every additional month must improve skill. The gap between regimes motivates the central interpretation of the paper: weather and soil provide measurable but secondary monitoring signal, while historical yield memory is a major driver of operational accuracy. A fixed-model lead-time check is included in the separate supplementary material.

\begin{table}[htbp]
\centering
\caption{Validation-selected test performance by forecast window for the no-yield-history weather-soil model and the operational with yield history model.}
\label{tab:lead-time}
\input{tables/table_lead_time.tex}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig01_framework_timeline.png}
\caption{Stage-aware early-warning workflow. Each forecast window uses weather only up to its cutoff month, combines it with static soil and crop-region context, and updates forecasts, risk scores, intervals, and analyst watch lists.}
\label{fig:framework}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig16_paper_safe_vs_operational_model.png}
\caption{No-yield-history weather-soil model versus operational with yield history model across forecast windows.}
\label{fig:paper-safe-operational}
\end{figure}

\subsection{Ablation: Weather Signal Versus Yield History}

Table~\ref{tab:ablation} summarizes the ablation regimes used to isolate information sources. The identity/time baseline is not trivial because crop and state-level region encode persistent productivity differences. Weather-only and weather-plus-soil feature sets add stage-specific environmental information, while the full operational model combines this information with past-yield history. The no-yield-history weather-soil models retain positive test skill, especially in the May-Oct window, but they do not consistently outperform identity/time or simple historical baselines. This limits the interpretation of the weather-soil setting: it provides evidence that within-season environmental features contain monitoring signal, not that weather and soil alone dominate persistent crop-region structure. Weather-deviation features are not uniformly beneficial once raw stage summaries and soil are included, which suggests that train-derived deviations should be treated as candidate monitoring features rather than guaranteed improvements. The lag-yield-only feature set is also strong, indicating that persistent crop-region yield history carries important predictive information.

\begin{table}[htbp]
\centering
\caption{Feature-regime definitions used for ablation and information-source analysis.}
\label{tab:ablation}
\resizebox{\textwidth}{!}{\input{tables/table_ablation_compact.tex}}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig13_ablation_r2_delta_by_feature_set.png}
\caption{R2 differences relative to the full operational feature set.}
\label{fig:ablation-delta}
\end{figure}

Table~\ref{tab:naive-baselines} adds internal comparator models run under the same temporal split and information constraints. These are not cross-paper SOTA claims, because external crop-yield papers differ in crop, spatial scale, features, and splits. Instead, the table asks whether chronological baselines, classical ML, strong tabular models, interpretable additive models, daily-weather GRU sequence models, and operational yield-history models behave differently on this benchmark. Simple historical baselines remain strong, and the operational models are the main deployment candidates for known crop-region histories. The no-yield-history weather-soil model remains useful as an information-isolation benchmark rather than as a claim of a novel algorithm.

\begin{table}[htbp]
\centering
\caption{Internal baseline-suite comparison under the same leakage-safe split. Each row reports the best validation-selected May-Oct model within the listed group and regime; the comparison is internal to this benchmark and is not a claim of outperforming external crop-yield systems.}
\label{tab:naive-baselines}
\resizebox{\textwidth}{!}{\input{tables/table_naive_baselines.tex}}
\end{table}

\subsection{Low-Yield-Risk Classification}

Risk classification is treated as a decision-support layer rather than the primary performance claim. With thresholds chosen on validation years only, the best May-Oct classifier is LogisticRegression with threshold 0.27, precision 0.469, recall 0.556, F1 0.508, ROC-AUC 0.797, PR-AUC 0.390, and Brier score 0.135 on the test period. The May-Oct test prevalence is 18.1\%, so the PR-AUC is above the random-ranking baseline but still far from a reliable automatic-alert system. The tuned threshold improves recall relative to a naive 0.5 cutoff, which is desirable for early warning where missing a shortfall may be costly. However, the modest precision means alerts should be interpreted as screening signals rather than deterministic event calls. In practice, the classifier would be most useful as a ranked watch list for analyst review, not as an automatic policy trigger.

\begin{table}[htbp]
\centering
\caption{May-Oct low-yield-risk confusion matrix using a validation-selected threshold.}
\label{tab:confusion}
\input{tables/table_confusion_matrix.tex}
\end{table}

\begin{table}[htbp]
\centering
\caption{Held-out May-Oct ranked watch-list example. Rank scores are uncalibrated classifier scores used for ordering analyst-review lists and should not be interpreted as fully calibrated event probabilities. Observed indicates the held-out low-yield-risk label.}
\label{tab:watch-list}
\resizebox{\textwidth}{!}{\input{tables/table_watch_list_top10.tex}}
\end{table}

\subsection{Stress Validation}

Table~\ref{tab:validation} summarizes robustness checks using the best model within each validation protocol and feature regime. Values are averaged across the folds and forecast-window rows belonging to each protocol, so the held-out-test row is a stress-validation summary rather than a duplicate of the single May-Oct result in Table~\ref{tab:lead-time}. Rolling-origin performance remains strong, especially for the operational model, suggesting that the headline time-split result is not purely an artifact of one train-test boundary. Leave-one-crop-out is substantially harder for the no-yield-history model, which is expected because crop identity changes the yield scale and sensitivity to weather. Leave-one-region-out produces weak or negative R2. The region-transfer result is the most important limitation: the current framework is better supported for known state-level histories than for unseen-state transfer. This weakness is also consistent with the feature-importance results, where crop-region history and categorical/time structure carry substantial signal.

\begin{table}[htbp]
\centering
\caption{Compact stress validation summary. Rows report mean performance across the folds and forecast windows belonging to each protocol for the best model within a feature regime. The held-out-test row is therefore an all-window stress summary, not the May-Oct-only result in Table~\ref{tab:lead-time}; leave-one-region-out is a deliberately hard unseen-region test.}
\label{tab:validation}
\resizebox{\textwidth}{!}{\input{tables/table_validation.tex}}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.72\textwidth]{figures/fig14_stress_validation_heatmap.png}
\caption{Stress validation heatmap by validation protocol and feature regime. Values use the same best-model, all-window summary as Table~\ref{tab:validation}, not single-window May-Oct scores.}
\label{fig:stress}
\end{figure}
\FloatBarrier

\subsection{Feature Group Importance}

Group permutation importance confirms the ablation results. Lag-yield features dominate the operational model, especially in early windows where less current-season weather has accumulated. In the no-yield-history setting, soil background, categorical/time structure, and heat/cold features are among the most important groups. Rainfall and dry-spell features contribute in some windows, but their importance is less stable than the combined history signal. We interpret these patterns as evidence for monitoring and vulnerability conditioning rather than as causal attribution: the model can identify useful statistical signals, but the design is not an intervention study.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.94\textwidth]{figures/fig19_feature_importance_split.png}
\caption{Feature group importance measured as mean RMSE increase after group permutation. Operational importance should not be interpreted as within-season weather importance because lag-yield features are allowed in that regime.}
\label{fig:importance}
\end{figure}
\FloatBarrier

\section{Discussion}

The results suggest a two-tier interpretation. The no-yield-history weather-soil model establishes that stage-aware weather and soil information contain measurable but secondary early-warning signal. The operational model, however, is the more accurate forecasting tool because it also uses historical yield memory. This distinction is practically important: agencies or supply-chain users with stable regional histories may prefer the operational model, while scientific analyses of within-season climate signal should emphasize the no-yield-history model.

Taken together, the results answer the four research questions as follows. RQ1: useful operational skill is available from May-Jun and improves toward May-Oct, while no-yield-history skill is positive but noisier. RQ2: lagged yield history and crop-region persistence explain much of the strongest accuracy, whereas weather-soil features provide a weaker but interpretable monitoring signal. RQ3: risk classification is useful as a ranked screening layer, not as an automatic trigger. RQ4: rolling-origin checks are encouraging, but leave-one-region-out transfer remains weak, limiting deployment to known crop-region histories.

The May-Jun operational result indicates that useful signal exists before the full growing season is observed. May-Oct provides the strongest final pre-harvest performance. Thus, the framework is best seen as a monitoring trajectory rather than a single fixed-date predictor. A decision maker could inspect May-Jun results as an early watch list, update the same crop-region panel as the season progresses, and use May-Sep or May-Oct results for more confident supply-risk assessment.

The strongest practical use cases are state-level monitoring, drought preparedness, procurement planning, and supply-risk assessment. The model does not decide policy or allocate payments; instead, it helps prioritize which crop-region-year combinations deserve attention. This is also why calibration, threshold tuning, and interval coverage are included as supplementary diagnostics. Even when a point forecast is accurate on average, risk communication requires knowing how often alerts are missed, how often intervals cover outcomes, and how much uncertainty remains.

A residual-check analysis further clarifies the claim. When a crop-region-year baseline is fitted first, a residual-only weather-soil model does not beat the direct no-yield-history weather-soil model. This negative result supports the conservative narrative: weather and soil signals are useful, but much state-level predictability is tied to persistent crop-region structure and historical yield memory.

Soil attributes should be interpreted as regional conditioning variables. Their incremental accuracy gains are small and not uniform across feature regimes, but they help represent background vulnerability differences that would otherwise be absorbed entirely by crop-region identifiers.

\section{Limitations}

The analysis is state-level, not farm-level. It does not support field-scale prescription, insurance payout decisions, or causal claims about heat, drought, or soil mechanisms. The soil features are regional background covariates and should not be interpreted as independently manipulated causal factors. Leave-one-region-out validation is weak, so transfer to unseen states remains an open problem. Crop-specific phenological calendars are also left for future work; this study uses a common May-Oct winter-crop proxy for all crops. Finally, the yield panel is small by modern machine-learning standards, so future work should test whether remote sensing, crop calendars, and finer spatial labels can improve transfer without introducing leakage.

\section{Conclusion}

We presented a stage-aware early-warning benchmark framework for Australian winter crop yield shortfall. The best operational May-Oct model reached RMSE 0.660 t/ha and R2 0.728, while the best no-yield-history weather-soil May-Oct internal-suite model reached RMSE 0.821 t/ha and R2 0.579. The gap between these regimes shows that weather and soil provide measurable but secondary monitoring signal, while historical yield memory is central to the strongest operational forecasts. This supports cautious state-level yield-risk monitoring while clearly bounding farm-level, causal, and unseen-region claims. The framework should be recalibrated before use in unseen regions or states.

\begingroup
\small
\setlength{\bibsep}{0pt}
\bibliography{acml26}
\endgroup

\end{document}
"""
    (PAPER_DIR / "main.tex").write_text(textwrap.dedent(tex).strip() + "\n", encoding="utf-8")


def write_main_tex() -> None:
    tex = r"""
\documentclass[wcp]{jmlr}

\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{placeins}
\usepackage{lineno}

\pagenumbering{gobble}
\hypersetup{
  pageanchor=false,
  hidelinks,
  colorlinks=false,
  linkcolor=black,
  citecolor=black,
  urlcolor=black,
  filecolor=black,
  pdfborder={0 0 0}
}
\bibpunct{[}{]}{,}{n}{,}{,}
\setcitestyle{numbers,square,comma,sort&compress}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\newcommand{\cs}[1]{\texttt{\char`\\#1}}
\makeatletter
\let\Ginclude@graphics\@org@Ginclude@graphics
\setlength{\@fptop}{0pt}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother

\jmlryear{2026}
\jmlrworkshop{ACML 2026}
\jmlrvolume{}
\jmlrpages{}
\makeatletter
\renewcommand*{\@titlefoot}{}
\gdef\@editor{}
\let\cleardoublepage\clearpage
\def\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{}%
  \def\@evenhead{}%
  \def\@oddfoot{}%
  \def\@evenfoot{}%
}
\makeatother

\title[Stage-Aware Crop Yield Warning]{Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall Using Daily Weather and Soil Data}

\author{}

\begin{document}

\maketitle

\begin{abstract}
Early warning of crop yield shortfall is useful only when risk can be updated before the season is effectively complete and when forecast skill is not driven by post-harvest leakage. We study Australian winter crops at state level using 966 region-crop-year yield observations from 1989--2021, daily SILO weather, and Soil and Landscape Grid attributes. Daily weather is converted into May-Jun, May-Jul, May-Aug, May-Sep, and May-Oct feature windows. We separate a no-yield-history weather-soil setting from an operational yield-history setting to distinguish within-season environmental signal from persistent crop-region yield memory. On held-out 2017--2021 seasons, the best no-yield-history May-Oct internal-suite model reaches RMSE 0.821 t/ha and R2 0.579, while the best operational May-Oct model reaches RMSE 0.660 t/ha and R2 0.728. Ablation and internal baseline-suite comparisons show that weather and soil provide measurable but secondary monitoring information; persistent crop-region structure and lagged yield history account for much of the strongest operational skill. Risk classification and uncertainty diagnostics support the framework as a state-level monitoring tool for known crop-region histories, while clearly limiting causal, farm-level, and unseen-region claims.
\end{abstract}

\begin{keywords}
crop yield forecasting; early warning; climate risk; Australian agriculture; weather features; soil data; uncertainty
\end{keywords}

\section{Introduction}

Australian dryland grain systems face recurring climate risks, including rainfall deficits, heat events, frost, and periods of high evaporative demand. These risks are particularly important for winter crops, where the difference between early vegetative conditions and late-season grain-filling conditions can change the decision value of a forecast. Prior work has shown that Australian wheat yields have been affected by adverse climate trends and by drought-heat exposure \citep{hochman2017climateTrendsAustralia,feng2019droughtHeatAustralia}. For decision support, however, the operational question is not only whether yield can be estimated after the season, but whether risk can be updated at meaningful within-season stages.

This paper studies state-level early warning for Australian winter crops. The task is to predict yield and low-yield shortfall risk for each region, crop, harvest year, and forecast window. Unlike a post-hoc anomaly attribution or multi-method XAI study, the central question is operational timing: how early can yield-shortfall risk be monitored? A May-Jun estimate supports early monitoring, while a May-Oct estimate approaches a full-season pre-harvest assessment. This framing is aligned with early-warning work that emphasizes anticipatory crop-failure monitoring rather than post-season diagnosis alone \citep{anderson2024preseasonWarning}.

The empirical design is intentionally conservative. We remove production and area variables from the main predictors, compute expected-yield and weather-deviation baselines from training years only, and keep the 2017--2021 test period outside all scaling, thresholding, calibration, and residual estimation steps. This avoids turning post-harvest information into apparent forecast skill.

The main contribution is a stage-aware benchmark framework, not a new forecasting algorithm. First, a no-yield-history weather-soil model tests whether daily weather, train-derived weather deviations, and soil background provide measurable evidence of within-season monitoring signal. Second, an operational model adds lagged yield history to quantify how much forecast quality improves when historical production memory is allowed. This distinction matters because a strong operational forecast may partly reflect persistent crop-region productivity rather than purely within-season weather signal.

We make three contributions. First, we formulate Australian winter-crop forecasting as a lead-time-aware shortfall-monitoring benchmark, where predictions are updated from May-Jun to May-Oct rather than evaluated only after the season is nearly complete. Second, we separate a no-yield-history weather-soil setting from an operational yield-history setting, allowing weather-derived early-warning evidence to be distinguished from persistent crop-region productivity memory. Third, we evaluate practical usefulness through temporal holdout testing, internal baseline-suite comparison, feature-regime ablation, stress validation, threshold-tuned risk screening, and uncertainty diagnostics, framing the outputs as state-level decision support rather than causal or farm-level prediction. The machine-learning contribution is therefore the leakage-safe temporal evaluation protocol and benchmark design: staged information sets, explicit feature-regime separation, and stress tests for Australian winter-crop monitoring.

\section{Related Work}

\subsection{Crop Yield Forecasting and Early Warning}

Machine learning has become a common approach for regional crop-yield forecasting, with weather, soil, remote sensing, and management covariates appearing across many systems \citep{vanKlompenburg2020systematicReview,paudel2021largeScaleYield}. At the same time, several studies warn that yield datasets are often small relative to the dimensionality of environmental predictors, making regularized and tree-based tabular models attractive for grain forecasting \citep{meroni2021smallData}. Recent work also pushes yield modeling toward multi-modal benchmarks and operational decision-support interfaces \citep{lin2024cropnet,declercq2024indiaRice}. Our setting is smaller and more targeted: state-level Australian winter crops with a fixed May-Oct monitoring calendar.

Early-warning studies differ from generic yield prediction because lead time is part of the objective. A model that works only after most weather information is observed may have limited preparedness value. Preseason and early-season crop-failure forecasts show the importance of evaluating when skill emerges, not only how accurate the final estimate is \citep{anderson2024preseasonWarning}. We adopt this view by reporting skill separately for each forecast window.

\subsection{Stage-Aware Weather and Extreme-Event Features}

Weather-yield relationships are stage dependent. Developmental-stage studies show that climatic means and weather extremes can have different effects depending on when they occur during crop growth \citep{schierhorn2021developmentalStages}. Broader climate-risk evidence also emphasizes compound hot-dry extremes, crop-specific vulnerability, and heterogeneous regional responses \citep{heino2023hotDryExtremes,sjulgard2023swedenAnomalies}. We therefore aggregate daily weather into stage-aware rainfall, heat, cold, evaporative-demand, radiation, and compound-stress features rather than relying only on full-season averages. Recent extreme-weather yield studies further motivate sparse or regularized modeling of drought, heat, flooding, and compound hazards \citep{heilemann2024lassoExtremes,kabtih2025hotPluvial}.

\subsection{Interpretability, Uncertainty, and Decision Support}

For agricultural risk monitoring, the output is used to prioritize attention rather than to prove a mechanistic causal pathway. Interpretability and uncertainty are therefore practical requirements, especially when models inform public or business decisions \citep{rudin2022blackboxBrief,paudel2023interpretability}. We use transparent ablation, group permutation importance, and simple lead-time comparisons rather than making the paper a post-hoc XAI study. Explainable boosting and related glassbox tools provide one route for transparent tabular modeling \citep{nori2019interpretml}, while conformal and conformalized quantile methods motivate distribution-aware intervals with explicit coverage diagnostics \citep{romano2019cqr,angelopoulos2023conformal}.

\section{Methodology}

\subsection{Study Setting and Data}

The analysis unit is a region-crop-year observation. The yield panel is derived from the ABARES Australian Crop Report state crop data workbook \path{03_AustCropRrt20260303_StateCropData_v1.0.0.xlsx}, accessed on 4 July 2026 \citep{abares2026australianCropReport}. After filtering to five winter crops and six state-level regions, the panel contains 966 observed entries for harvest years 1989--2021 (Figure~\ref{fig:study-regions}; Table~\ref{tab:data-summary}). Each yield observation is expanded over five forecast windows, giving 4,830 window rows. This expansion does not create new yield outcomes: models and metrics are evaluated within forecast windows, and all train/validation/test decisions are made by calendar year so the same region-crop-year outcome cannot be split across evaluation periods.

\begin{figure}[t]
\centering
\includegraphics[width=0.86\textwidth]{figures/fig01_australia_study_regions.pdf}
\caption{Study regions and panel coverage for the Australian winter-crop benchmark. The panel contains 966 observed region-crop-year entries from six state-level regions, five winter crops, and harvest years 1989--2021. QLD and TAS have fewer observed entries because some crop-region-year source records are missing; the benchmark uses state-level regional weather exposure proxies and does not claim crop-area-weighted gridded exposure.}
\label{fig:study-regions}
\end{figure}

\begin{table}[t]
\centering
\caption{Data summary for the Australian winter crop early-warning panel.}
\label{tab:data-summary}
\input{tables/table_data_summary.tex}
\end{table}

Daily weather is taken from SILO/LongPaddock Australian climate records, which are based on spatial interpolation of observed weather data \citep{jeffrey2001silo}. The raw project tables retain daily state-level regional series with latitude and longitude metadata; the current benchmark aggregates these regional daily series by forecast cutoff as a state-level exposure proxy and does not claim crop-area-weighted gridded exposure. We use daily rainfall, maximum and minimum temperature, radiation, evaporation, and vapor pressure to build cutoff-specific summaries. Soil covariates are static regional background attributes from ASRIS/Soil and Landscape Grid of Australia \citep{grundy2015slga}. They include available water capacity, bulk density, clay, sand, silt, soil organic carbon, total nitrogen, total phosphorus, pH, cation exchange capacity, depth of soil, and depth of regolith, summarized as topsoil/subsoil or regional aggregates where available. Because these soil summaries are region-level and static, they are treated as background vulnerability covariates, not as farm-level measurements or causal soil interventions.

\subsection{Prediction Targets}

The regression target is yield in tonnes per hectare, denoted $y_{r,c,t}$. We also define an expected-yield baseline from training years and the shortfall
\[
  s_{r,c,t} = \hat{y}^{\mathrm{exp}}_{r,c,t} - y_{r,c,t}.
\]
The expected-yield baseline is fitted using training years only. In the main specification, expected yield is estimated from a crop-region linear trend in calendar year; crop-level linear trends are used when a crop-region series is too short, and the overall training mean is the final tier in this training-only hierarchy. This baseline excludes current-season weather, production, area, lagged yield, and all validation/test outcomes. Validation and test shortfalls are then computed by applying the fixed training-period baseline to later years. The binary low-yield-risk label is one when shortfall exceeds the crop-specific 80th percentile of the training distribution. We use this threshold as a broad watch-list definition rather than as a severe crop-failure definition. This target follows the general idea that yield anomalies or shortfalls require a reference trend before climate impacts can be assessed \citep{meng2024detrending}, but here the shortfall is used for early-warning classification rather than post-hoc causal attribution. The classification target is used for secondary risk-alert analyses; the main quantitative results remain regression and ablation metrics.

\subsection{Feature Construction and Model Regimes}

For each forecast window $w$, daily weather is aggregated only from May through the cutoff month of $w$. The design intentionally mimics an operational update: May-Jun uses only two months of weather, while May-Oct uses the longest pre-harvest window. No weather feature uses data after the cutoff month. This makes the lead-time comparison interpretable as a sequence of information sets rather than a single full-season model.

The weather feature groups are rainfall accumulation and dry spells, heat and cold exposure, radiation and evaporative demand, and compound hot-dry stress. These choices are motivated by stage-aware and extreme-weather yield studies \citep{schierhorn2021developmentalStages,heino2023hotDryExtremes,heilemann2024lassoExtremes}. We additionally construct weather-deviation features by subtracting train-period region-window baselines. This produces rainfall, temperature, dry-spell, heat-day, evapotranspiration, and radiation deviations that are comparable across regions and forecast windows. The weather-deviation baselines are computed using training years only, so they do not leak test-period information. Feature groups and ablation regimes are listed in the supplementary material.

We report two primary feature regimes. The no-yield-history weather-soil regime uses stage weather features, train-derived weather-deviation features, soil background features, crop identity, region identity, and year. It excludes lagged yield, rolling yield, production, and area; it is not free of crop, state-level region, or time structure. The operational with yield history regime adds past-yield features from prior years for the same crop-region series. This makes it useful for deployment in known crop-region histories, but it is not interpreted as pure weather evidence.

The model suite combines regularized linear baselines with gradient-boosted tabular models. Ridge and ElasticNet provide stable small-data baselines, while LightGBM and CatBoost provide nonlinear tabular benchmarks \citep{ke2017lightgbm,prokhorenkova2018catboost}. The expanded internal baseline suite also includes Random Forest, SVR-RBF, XGBoost, HistGradientBoosting, GAM, and a daily-weather GRU sequence comparator; implementation details are reported in the supplementary material. We avoid presenting a black-box-only result: every performance table is paired with feature-set ablation, lead-time comparison, or group permutation importance. This is consistent with the broader argument that high-stakes decision support should prefer transparent or at least carefully audited modeling workflows \citep{rudin2022blackboxBrief}.

Classification models use validation-tuned thresholds for low-yield-risk decisions. Instead of fixing a probability threshold at 0.5, we choose thresholds on validation years for F1, recall constraints, or precision constraints, then report test metrics once. Probabilistic interval diagnostics use quantile-style predictions and conformal calibration ideas as supplementary checks \citep{romano2019cqr,angelopoulos2023conformal}. These outputs are not framed as definitive event decisions; they are screening layers for state-level monitoring.

\subsection{Experimental Setup}

The main split trains on 1989--2012, tunes on 2013--2016, and tests on 2017--2021. The test period is not used for scalers, feature thresholds, residual calibration, conformal residuals, expected-yield baselines, weather-deviation baselines, or low-yield-risk thresholds. This temporal split reflects the intended forecasting direction: models learn from earlier seasons and are evaluated on later seasons.

Regression metrics are MAE, RMSE, and R2. Classification metrics are precision, recall, F1, ROC-AUC, PR-AUC, and Brier score. For probabilistic diagnostics, we summarize interval coverage and width. The conformal intervals target nominal 80\% marginal coverage using validation-period calibration residuals only; test coverage is reported without retuning. Deviations from nominal coverage are interpreted as evidence of calibration difficulty under temporal shift and small calibration samples, not as a target corrected using test outcomes. The primary model-selection comparisons are always made within the same forecast window and split.

Robustness checks include rolling-origin folds, leave-one-crop-out validation, and leave-one-region-out validation. Rolling-origin validation tests sensitivity to a single time split. Leave-one-crop-out and leave-one-region-out are intentionally difficult transfer tests. We do not treat them as the main deployment setting, because the intended use case is monitoring crop-region combinations with historical observations. Instead, they identify where the framework is not yet robust enough for unseen-crop or unseen-state generalization.

\section{Results}

\subsection{Lead-Time Skill}

Forecast accuracy improves as later-season information becomes available, and the operational setting consistently outperforms the no-yield-history setting across all forecast windows. This pattern indicates that lagged yield history contributes substantial predictive value beyond within-season weather and soil information. By May-Oct, the best operational model reaches RMSE 0.660 t/ha and R2 0.728, while the best no-yield-history model reaches RMSE 0.821 t/ha and R2 0.579 (Table~\ref{tab:lead-time}).

The operational model is already informative at May-Jun, with RMSE 0.719 t/ha and R2 0.677. This early result should not be read as weather-only skill, because operational predictors include lagged yield and crop-region memory. The no-yield-history trajectory is weaker and not strictly monotonic: later windows add useful weather information but also correlated features and noise in a small state-level panel. We therefore interpret the estimates as a monitoring trajectory rather than assuming that each added month must improve accuracy. A fixed-model lead-time check is included in the supplementary material.

\begin{table}[t]
\centering
\caption{Validation-selected test performance by forecast window for the no-yield-history weather-soil model and the operational with yield history model.}
\label{tab:lead-time}
\input{tables/table_lead_time.tex}
\end{table}
\FloatBarrier

\subsection{Ablation and Internal Comparisons}

The ablation results confirm that weather and soil provide a measurable but secondary monitoring signal, whereas persistent crop-region structure and lagged yield history explain much of the strongest predictive performance. This conclusion is consistent across the feature-regime comparison and the internal baseline suite: no-yield-history weather-soil models retain positive test skill, but they do not consistently dominate identity/time or simple historical baselines.

Table~\ref{tab:naive-baselines} adds internal comparator models run under the same temporal split and information constraints. These are not cross-paper SOTA claims, because external crop-yield papers differ in crop, spatial scale, features, and splits. Instead, the table asks whether chronological baselines, classical ML, strong tabular models, interpretable additive models, daily-weather GRU sequence models, and operational yield-history models behave differently on this benchmark. Simple historical baselines remain strong, and operational models are the main deployment candidates for known crop-region histories. The no-yield-history weather-soil model remains useful as an information-isolation benchmark rather than as a claim of a novel algorithm. Detailed feature-regime definitions and ablation heatmaps are in the supplementary material.

\begin{table}[t]
\centering
\caption{Internal baseline-suite comparison under the same leakage-safe split. Each row reports the best validation-selected May-Oct model within the listed group and regime; the comparison is internal to this benchmark and is not a claim of outperforming external crop-yield systems.}
\label{tab:naive-baselines}
\resizebox{\textwidth}{!}{\input{tables/table_naive_baselines.tex}}
\end{table}
\FloatBarrier

\subsection{Low-Yield-Risk Classification}

Risk classification is treated as a decision-support layer rather than the primary performance claim. With thresholds chosen on validation years only, the best May-Oct classifier is LogisticRegression with threshold 0.27, precision 0.469, recall 0.556, F1 0.508, ROC-AUC 0.797, PR-AUC 0.390, and Brier score 0.135 on the test period. The May-Oct test prevalence is 18.1\%, so the PR-AUC is above the random-ranking baseline but still far from a reliable automatic-alert system. The tuned threshold improves recall relative to a naive 0.5 cutoff, which is desirable for early warning where missing a shortfall may be costly. However, the modest precision means alerts should be interpreted as screening signals rather than deterministic event calls. In practice, the classifier would be most useful as a ranked watch list for analyst review, not as an automatic policy trigger. Full threshold sensitivity, confusion-matrix, and ranked watch-list details are reported in the supplementary material.

\subsection{Stress Validation}

Robustness declines substantially under harder transfer settings, especially when the model is evaluated on unseen regions. In contrast, rolling-origin validation remains comparatively strong, suggesting that the main temporal holdout result is not driven only by a single train-test boundary. These stress tests support the intended use case of monitoring known crop-region histories while highlighting weak transfer to unseen state-level regions (Table~\ref{tab:validation}).

Leave-one-crop-out is substantially harder for the no-yield-history model, which is expected because crop identity changes the yield scale and sensitivity to weather. Leave-one-region-out produces weak or negative R2. The region-transfer result is the most important limitation: the current framework is better supported for known state-level histories than for unseen-state transfer. A heatmap view of the same stress-validation summary is included in the supplementary material.

\begin{table}[t]
\centering
\caption{Compact stress validation summary. Rows report mean performance across the folds and forecast windows belonging to each protocol for the best model within a feature regime. The held-out-test row is therefore an all-window stress summary, not the May-Oct-only result in Table~\ref{tab:lead-time}; leave-one-region-out is a deliberately hard unseen-region test.}
\label{tab:validation}
\resizebox{\textwidth}{!}{\input{tables/table_validation.tex}}
\end{table}
\FloatBarrier

\subsection{Feature Group Importance}

Group permutation importance confirms the ablation results. Lag-yield features dominate the operational model, especially in early windows where less current-season weather has accumulated. In the no-yield-history setting, soil background, categorical/time structure, and heat/cold features are among the most important groups. Rainfall and dry-spell features contribute in some windows, but their importance is less stable than the combined history signal. We interpret these patterns as evidence for monitoring and vulnerability conditioning rather than as causal attribution: the model can identify useful statistical signals, but the design is not an intervention study. The full feature-importance figure is kept in the supplementary material to avoid duplicating the main ablation evidence.

\section{Discussion}

The results suggest a two-tier interpretation. The no-yield-history weather-soil model establishes that stage-aware weather and soil information contain measurable but secondary early-warning signal. The operational model, however, is the more accurate forecasting tool because it also uses historical yield memory. This distinction is practically important: agencies or supply-chain users with stable regional histories may prefer the operational model, while scientific analyses of within-season climate signal should emphasize the no-yield-history model.

The May-Jun operational result indicates that useful signal exists before the full growing season is observed. May-Oct provides the strongest final pre-harvest performance. Thus, the framework is best seen as a monitoring trajectory rather than a single fixed-date predictor. A decision maker could inspect May-Jun results as an early watch list, update the same crop-region panel as the season progresses, and use May-Sep or May-Oct results for more confident supply-risk assessment.

The strongest practical use cases are state-level monitoring, drought preparedness, procurement planning, and supply-risk assessment. The model does not decide policy or allocate payments; instead, it helps prioritize which crop-region-year combinations deserve attention. This is also why calibration, threshold tuning, and interval coverage are included as supplementary diagnostics. Even when a point forecast is accurate on average, risk communication requires knowing how often alerts are missed, how often intervals cover outcomes, and how much uncertainty remains.

A residual-check analysis further clarifies the claim. When a crop-region-year baseline is fitted first, a residual-only weather-soil model does not beat the direct no-yield-history weather-soil model. This negative result supports the conservative narrative: weather and soil signals are useful, but much state-level predictability is tied to persistent crop-region structure and historical yield memory.

Soil attributes should be interpreted as regional conditioning variables. Their incremental accuracy gains are small and not uniform across feature regimes, but they help represent background vulnerability differences that would otherwise be absorbed entirely by crop-region identifiers.

The limitations define the proper deployment boundary. The analysis is state-level, not farm-level. It does not support field-scale prescription, insurance payout decisions, or causal claims about heat, drought, or soil mechanisms. The soil features are regional background covariates and should not be interpreted as independently manipulated causal factors. Leave-one-region-out validation is weak, so transfer to unseen states remains an open problem. The common May-Oct winter-crop calendar is a practical simplification that does not capture crop-specific phenology. The yield panel is also small by modern machine-learning standards. Future work should test whether remote sensing, crop-specific calendars, finer spatial labels, improved transfer learning, and better uncertainty calibration can improve monitoring without introducing leakage.

\section{Conclusion}

We presented a stage-aware early-warning benchmark framework for Australian winter crop yield shortfall. The best operational May-Oct model reached RMSE 0.660 t/ha and R2 0.728, while the best no-yield-history weather-soil May-Oct internal-suite model reached RMSE 0.821 t/ha and R2 0.579. The gap between these regimes shows that weather and soil provide measurable but secondary monitoring signal, while historical yield memory is central to the strongest operational forecasts. This supports cautious state-level yield-risk monitoring while clearly bounding farm-level, causal, and unseen-region claims. The framework should be recalibrated before use in unseen regions or states.

\begingroup
\small
\setlength{\bibsep}{0pt}
\bibliography{acml26}
\endgroup

\end{document}
"""
    (PAPER_DIR / "main.tex").write_text(textwrap.dedent(tex).strip() + "\n", encoding="utf-8")


def _write_supplementary_tex_legacy() -> None:
    tex = r"""
\documentclass[wcp]{jmlr}

\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}

\hypersetup{hidelinks,colorlinks=false,pdfborder={0 0 0}}
\renewcommand{\thetable}{S\arabic{table}}
\jmlryear{2026}
\jmlrworkshop{ACML 2026}
\jmlrvolume{}
\jmlrpages{}
\makeatletter
\renewcommand*{\@titlefoot}{}
\gdef\@editor{}
\let\cleardoublepage\clearpage
\def\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{}%
  \def\@evenhead{}%
  \def\@oddfoot{}%
  \def\@evenfoot{}%
}
\makeatother
\title[Supplementary]{Supplementary Material: Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall}
\author{}

\begin{document}
\maketitle

\section{Fixed-Model and Target Checks}

\begin{table}[H]
\centering
\caption{Fixed-model lead-time comparison. NYH denotes the no-yield-history weather-soil regime; Op. denotes the operational regime. Values are test RMSE.}
\input{tables/table_fixed_model_lead_time.tex}
\end{table}

\begin{table}[H]
\centering
\caption{Missing source yield entries by state-crop series. The total confirms 966 observed entries out of 990 possible state-crop-year combinations in 1989--2021 for the observed crop-region pairs.}
\input{tables/table_missing_counts.tex}
\end{table}

\input{tables/table_baselines_full_appendix.tex}

Full baseline-suite metrics remain available in the accompanying CSV artifacts; the PDF keeps the compact summary to preserve readability.

\section{Data and Implementation Details}

The raw yield source is the ABARES Australian Crop Report state-crop-data workbook \path{03_AustCropRrt20260303_StateCropData_v1.0.0.xlsx}, accessed on 4 July 2026; the modeled panel is the derived \texttt{yield\_panel.csv}. Fields used are crop area, production, and yield in tonnes per hectare, filtered to the five winter crops and six state-level regions in the main paper. The harvest year is represented by \texttt{year\_start}. Production and area are retained for audit but excluded from model features.

SILO/LongPaddock weather inputs are daily rainfall, maximum and minimum temperature, radiation, evaporation, and vapor pressure. They are aggregated from May through each forecast cutoff. The benchmark uses state-level regional daily series as a monitoring exposure proxy; latitude and longitude metadata are retained, but no crop-area-weighted gridded exposure is claimed. Weather-deviation features use training years only: for feature value $x_{r,w,t}$, the deviation is $x_{r,w,t}-\bar{x}_{r,w}^{train}$ for the same region and forecast window. Soil/ASRIS attributes are static regional background covariates from the Soil and Landscape Grid of Australia, including available water capacity, texture, bulk density, soil organic carbon, nutrients, pH, cation exchange capacity, and depth variables. These features condition regional vulnerability and may partly proxy region identity.

The no-yield-history regime has 73 predictors before one-hot expansion: crop, state-level region, year, 30 stage-weather summaries, 18 train-derived weather-deviation features, and 22 soil background attributes after excluding coordinate metadata. The operational regime adds one-year lag, three-year rolling past mean, and expanding past mean, for 76 predictors before one-hot expansion. Scaling, imputation, encoders, baselines, thresholds, and conformal residuals are fitted without test years; one-hot encoders handle unknown categories in stress tests. Configured weather thresholds are rain day 1 mm, heavy rain 10/25 mm, heat 25/30/35 C, frost 0 C, cold 5 C, and high evaporation 5 mm.

\section{Baseline-Suite Implementation}

The internal baseline suite is run once with random seed 42. This single-seed design is a limitation; model selection is by validation RMSE within each forecast window and feature regime, and test metrics are reported once. Tabular preprocessing uses median imputation and standard scaling for numeric predictors, one-hot encoding for crop and region, and unknown-category handling in held-out stress tests. The suite uses scikit-learn 1.9.0, XGBoost 3.3.0, LightGBM 4.6.0, CatBoost 1.2.10, pyGAM 0.12.0, and PyTorch 2.12.1+cpu in the recorded run.

Candidate models include Ridge with alpha 1.0 or 3.0; ElasticNet with alpha 0.01/l1 ratio 0.2 or alpha 0.03/l1 ratio 0.4; RandomForest with 400 trees and either unrestricted depth/min leaf 3 or depth 8/min leaf 5; SVR-RBF with C 1 or 10 and epsilon 0.1; HistGradientBoosting; XGBoost; LightGBM; CatBoost; and a pyGAM additive model over transformed features. The GRU sequence comparator uses daily weather up to the cutoff window, static covariates concatenated after the sequence encoder, and for the operational regime adds lag/rolling/expanding yield-history features to the static block. Two GRU configurations are validation-selected: hidden size 16 or 32, dropout 0.15 or 0.20, Adam learning rate 0.001, weight decay 1e-4, up to 160 epochs with patience 16.

\begin{table}[H]
\centering
\caption{Sensitivity of the low-yield-risk label to crop-specific training shortfall percentiles. The 80th percentile is the main setting.}
\input{tables/table_threshold_sensitivity.tex}
\end{table}

\begin{table}[H]
\centering
\caption{Best threshold-tuned classification metrics by forecast window. Pred.+ is the fraction of test examples flagged positive by the selected threshold.}
\resizebox{\textwidth}{!}{\input{tables/table_classification_summary.tex}}
\end{table}

\section{Uncertainty Diagnostics}

Coverage below the nominal 80\% target, especially in later windows, is interpreted as temporal calibration difficulty under small validation samples. These intervals are diagnostics for analyst review rather than reliable calibrated intervals for automatic decisions.

\begin{table}[H]
\centering
\caption{Compact conformal uncertainty diagnostics by forecast window. Coverage and interval width are evaluated on held-out test years after validation-period calibration.}
\resizebox{\textwidth}{!}{\input{tables/table_uncertainty_summary.tex}}
\end{table}

\end{document}
"""
    (PAPER_DIR / "supplementary.tex").write_text(textwrap.dedent(tex).strip() + "\n", encoding="utf-8")


def write_supplementary_tex() -> None:
    tex = r"""
\documentclass[wcp]{jmlr}

\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{placeins}

\hypersetup{hidelinks,colorlinks=false,pdfborder={0 0 0}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\jmlryear{2026}
\jmlrworkshop{ACML 2026}
\jmlrvolume{}
\jmlrpages{}
\makeatletter
\renewcommand*{\@titlefoot}{}
\gdef\@editor{}
\let\cleardoublepage\clearpage
\def\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{}%
  \def\@evenhead{}%
  \def\@oddfoot{}%
  \def\@evenfoot{}%
}
\makeatother
\title[Supplementary]{Supplementary Material: Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall}
\author{}

\begin{document}
\maketitle

\section{Fixed-Model and Feature-Regime Details}

\begin{table}[H]
\centering
\caption{Fixed-model lead-time comparison. NYH denotes the no-yield-history weather-soil regime; Op. denotes the operational regime. Values are test RMSE.}
\input{tables/table_fixed_model_lead_time.tex}
\end{table}

\begin{table}[H]
\centering
\caption{Feature groups used for stage-aware early-warning models.}
\resizebox{\textwidth}{!}{\input{tables/table_feature_groups.tex}}
\end{table}

\begin{table}[H]
\centering
\caption{Feature-regime definitions used for ablation and information-source analysis.}
\resizebox{\textwidth}{!}{\input{tables/table_ablation_compact.tex}}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig13_ablation_r2_delta_by_feature_set.png}
\caption{Ablation heatmap showing R2 differences relative to the full operational feature set. Short labels are used to keep the dense comparison readable.}
\end{figure}
\FloatBarrier

\section{Data Coverage and Baseline Suite}

\begin{table}[H]
\centering
\caption{Missing source yield entries by state-crop series. The total confirms 966 observed entries out of 990 possible state-crop-year combinations in 1989--2021 for the observed crop-region pairs.}
\input{tables/table_missing_counts.tex}
\end{table}

\input{tables/table_baselines_full_appendix.tex}

Full baseline-suite metrics remain available in the accompanying CSV artifacts; the PDF keeps the compact summary to preserve readability.

\section{Data and Implementation Details}

The raw yield source is the ABARES Australian Crop Report state-crop-data workbook \path{03_AustCropRrt20260303_StateCropData_v1.0.0.xlsx}, accessed on 4 July 2026; the modeled panel is the derived \texttt{yield\_panel.csv}. Fields used are crop area, production, and yield in tonnes per hectare, filtered to the five winter crops and six state-level regions in the main paper. The harvest year is represented by \texttt{year\_start}. Production and area are retained for audit but excluded from model features.

SILO/LongPaddock weather inputs are daily rainfall, maximum and minimum temperature, radiation, evaporation, and vapor pressure. They are aggregated from May through each forecast cutoff. The benchmark uses state-level regional daily series as a monitoring exposure proxy; latitude and longitude metadata are retained, but no crop-area-weighted gridded exposure is claimed. Weather-deviation features use training years only: for feature value $x_{r,w,t}$, the deviation is $x_{r,w,t}-\bar{x}_{r,w}^{train}$ for the same region and forecast window. Soil/ASRIS attributes are static regional background covariates from the Soil and Landscape Grid of Australia, including available water capacity, texture, bulk density, soil organic carbon, nutrients, pH, cation exchange capacity, and depth variables. These features condition regional vulnerability and may partly proxy region identity.

The no-yield-history regime has 73 predictors before one-hot expansion: crop, state-level region, year, 30 stage-weather summaries, 18 train-derived weather-deviation features, and 22 soil background attributes after excluding coordinate metadata. The operational regime adds one-year lag, three-year rolling past mean, and expanding past mean, for 76 predictors before one-hot expansion. Scaling, imputation, encoders, baselines, thresholds, and conformal residuals are fitted without test years; one-hot encoders handle unknown categories in stress tests. Configured weather thresholds are rain day 1 mm, heavy rain 10/25 mm, heat 25/30/35 C, frost 0 C, cold 5 C, and high evaporation 5 mm.

\section{Baseline-Suite Implementation}

The internal baseline suite is run once with random seed 42. This single-seed design is a limitation; model selection is by validation RMSE within each forecast window and feature regime, and test metrics are reported once. Tabular preprocessing uses median imputation and standard scaling for numeric predictors, one-hot encoding for crop and region, and unknown-category handling in held-out stress tests. The suite uses scikit-learn 1.9.0, XGBoost 3.3.0, LightGBM 4.6.0, CatBoost 1.2.10, pyGAM 0.12.0, and PyTorch 2.12.1+cpu in the recorded run.

Candidate models include Ridge with alpha 1.0 or 3.0; ElasticNet with alpha 0.01/l1 ratio 0.2 or alpha 0.03/l1 ratio 0.4; RandomForest with 400 trees and either unrestricted depth/min leaf 3 or depth 8/min leaf 5; SVR-RBF with C 1 or 10 and epsilon 0.1; HistGradientBoosting; XGBoost; LightGBM; CatBoost; and a pyGAM additive model over transformed features. The GRU sequence comparator uses daily weather up to the cutoff window, static covariates concatenated after the sequence encoder, and for the operational regime adds lag/rolling/expanding yield-history features to the static block. Two GRU configurations are validation-selected: hidden size 16 or 32, dropout 0.15 or 0.20, Adam learning rate 0.001, weight decay 1e-4, up to 160 epochs with patience 16.

\section{Classification and Uncertainty Details}

\begin{table}[H]
\centering
\caption{Sensitivity of the low-yield-risk label to crop-specific training shortfall percentiles. The 80th percentile is the main setting.}
\input{tables/table_threshold_sensitivity.tex}
\end{table}

\begin{table}[H]
\centering
\caption{Best threshold-tuned classification metrics by forecast window. Pred.+ is the fraction of test examples flagged positive by the selected threshold.}
\resizebox{\textwidth}{!}{\input{tables/table_classification_summary.tex}}
\end{table}

\begin{table}[H]
\centering
\caption{Low-yield-risk confusion matrix using validation-selected thresholds.}
\resizebox{\textwidth}{!}{\input{tables/table_confusion_matrix.tex}}
\end{table}

\begin{table}[H]
\centering
\caption{Held-out May-Oct ranked watch-list example. Rank scores are uncalibrated classifier scores used for ordering analyst-review lists and should not be interpreted as fully calibrated event probabilities. Observed indicates the held-out low-yield-risk label.}
\resizebox{\textwidth}{!}{\input{tables/table_watch_list_top10.tex}}
\end{table}

Coverage below the nominal 80\% target, especially in later windows, is interpreted as temporal calibration difficulty under small validation samples. These intervals are diagnostics for analyst review rather than reliable calibrated intervals for automatic decisions.

\begin{table}[H]
\centering
\caption{Compact conformal uncertainty diagnostics by forecast window. Coverage and interval width are evaluated on held-out test years after validation-period calibration.}
\resizebox{\textwidth}{!}{\input{tables/table_uncertainty_summary.tex}}
\end{table}

\section{Supplementary Figures}

\begin{figure}[H]
\centering
\includegraphics[width=0.72\textwidth]{figures/fig14_stress_validation_heatmap.png}
\caption{Stress validation heatmap by validation protocol and feature regime. Values use the same best-model, all-window summary as the main-paper stress validation table, not single-window May-Oct scores.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{figures/fig19_feature_importance_split.png}
\caption{Feature group importance measured as mean RMSE increase after group permutation. Operational importance should not be interpreted as within-season weather importance because lag-yield features are allowed in that regime.}
\end{figure}

\end{document}
"""
    (PAPER_DIR / "supplementary.tex").write_text(textwrap.dedent(tex).strip() + "\n", encoding="utf-8")


def polish_latex_outputs() -> None:
    tex_files = [PAPER_DIR / "main.tex", PAPER_DIR / "supplementary.tex", *TABLE_DIR.glob("*.tex")]
    for path in tex_files:
        text = path.read_text(encoding="utf-8")
        text = text.replace("R2", r"$R^2$")
        path.write_text(text, encoding="utf-8")


def make_source_zip() -> Path:
    zip_path = REPORT_DIR / "acml2026_aus_early_warning_anonymous_source.zip"
    include_dirs = [FIG_DIR, TABLE_DIR]
    include_files = [PAPER_DIR / "main.tex", PAPER_DIR / "supplementary.tex", PAPER_DIR / "jmlr.cls", PAPER_DIR / "acml26.bib"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in include_files:
            zf.write(file_path, file_path.relative_to(PAPER_DIR))
        for directory in include_dirs:
            for file_path in sorted(directory.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(PAPER_DIR))
    return zip_path


def write_log(copied_figures: dict[str, str], source_zip: Path) -> None:
    rows = [
        ["paper_dir", str(PAPER_DIR)],
        ["source_zip", str(source_zip)],
        ["safe_round2_snapshot", str(SAFE_DIR)],
        ["figures_copied", str(len(copied_figures))],
        ["anonymous_author_block", "yes"],
        ["main_claim_source", "round2_safe_2026_06_24"],
    ]
    with (REPORT_DIR / "ACML_PAPER_IMPLEMENTATION_LOG.md").open("w", encoding="utf-8") as fh:
        fh.write("# ACML Paper Implementation Log\n\n")
        fh.write("## Generated Artifacts\n\n")
        for key, value in rows:
            fh.write(f"- `{key}`: {value}\n")
        fh.write("\n## Figure Mapping\n\n")
        for src, dst in copied_figures.items():
            fh.write(f"- `{src}` -> `paper_acml/figures/{dst}`\n")
        fh.write("\n## Anonymization Defaults\n\n")
        fh.write("- `main.tex` uses an empty `\\author{}` block.\n")
        fh.write("- The manuscript text avoids local absolute paths, author names, affiliations, and acknowledgements.\n")
        fh.write("- The paper separates no-yield-history weather-soil evidence from operational forecasts that use lag-yield history.\n")
        fh.write("- `acml26.bib` is regenerated from `Tong_hop_ref_bai_Uc_Early_Warning.docx`.\n")


def main() -> None:
    ensure_dirs()
    extract_template_files()
    make_study_region_map_figure()
    make_baseline_lead_time_figure()
    copied_figures = copy_figures()
    make_data_summary_table()
    make_lead_time_table()
    make_ablation_table()
    make_validation_table()
    make_classification_table()
    make_round3_table()
    make_feature_group_table()
    make_naive_baseline_table()
    make_missing_counts_table()
    make_fixed_model_table()
    make_threshold_sensitivity_table()
    make_confusion_matrix_table()
    make_watch_list_table()
    make_uncertainty_table()
    write_bibliography()
    write_main_tex()
    write_supplementary_tex()
    polish_latex_outputs()
    source_zip = make_source_zip()
    write_log(copied_figures, source_zip)
    print(f"Wrote ACML paper assets to {PAPER_DIR}")
    print(f"Wrote source zip to {source_zip}")


if __name__ == "__main__":
    main()
