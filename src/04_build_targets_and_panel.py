from __future__ import annotations

import numpy as np
import pandas as pd

from common import configured_path, ensure_output_dirs, load_config, markdown_table, split_name, write_markdown


def fit_expected_yield(yield_df: pd.DataFrame, config: dict) -> pd.Series:
    train_end = config["splits"]["train_end"]
    train = yield_df[yield_df["year_start"] <= train_end].copy()
    overall_mean = float(train["yield_t_ha"].mean())

    crop_models: dict[str, tuple[float, float]] = {}
    for crop, group in train.groupby("crop"):
        if group["year_start"].nunique() >= 3:
            slope, intercept = np.polyfit(group["year_start"], group["yield_t_ha"], deg=1)
            crop_models[str(crop)] = (float(slope), float(intercept))

    crop_region_models: dict[tuple[str, str], tuple[float, float]] = {}
    for (crop, region), group in train.groupby(["crop", "region"]):
        if group["year_start"].nunique() >= 3:
            slope, intercept = np.polyfit(group["year_start"], group["yield_t_ha"], deg=1)
            crop_region_models[(str(crop), str(region))] = (float(slope), float(intercept))

    expected = []
    for _, row in yield_df.iterrows():
        key = (str(row["crop"]), str(row["region"]))
        crop = str(row["crop"])
        year = float(row["year_start"])
        if key in crop_region_models:
            slope, intercept = crop_region_models[key]
            expected.append(slope * year + intercept)
        elif crop in crop_models:
            slope, intercept = crop_models[crop]
            expected.append(slope * year + intercept)
        else:
            expected.append(overall_mean)
    return pd.Series(expected, index=yield_df.index, dtype=float)


def add_targets(yield_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    yield_df = yield_df.copy()
    yield_df["expected_yield_t_ha"] = fit_expected_yield(yield_df, config)
    yield_df["yield_shortfall"] = yield_df["expected_yield_t_ha"] - yield_df["yield_t_ha"]

    train_mask = yield_df["year_start"] <= config["splits"]["train_end"]
    threshold_rows = []
    thresholds = {}
    for crop, train_crop in yield_df.loc[train_mask].groupby("crop"):
        threshold = float(
            np.percentile(
                train_crop["yield_shortfall"],
                float(config["thresholds"]["low_yield_percentile"]),
            )
        )
        thresholds[str(crop)] = threshold
        threshold_rows.append({"crop": crop, "low_yield_shortfall_threshold": threshold})

    yield_df["low_yield_shortfall_threshold"] = yield_df["crop"].map(thresholds)
    yield_df["low_yield_risk"] = (
        yield_df["yield_shortfall"] >= yield_df["low_yield_shortfall_threshold"]
    ).astype(int)
    return yield_df, pd.DataFrame(threshold_rows)


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)

    yield_df = pd.read_csv(configured_path(config, "yield_panel"))
    yield_df = yield_df[
        yield_df["crop"].isin(config["crops"]) & yield_df["region"].isin(config["regions"])
    ].copy()
    yield_df["split"] = yield_df["year_start"].apply(lambda year: split_name(int(year), config))
    yield_df, thresholds = add_targets(yield_df, config)

    weather = pd.read_csv(configured_path(config, "processed_dir") / "weather_stage_features.csv")
    soil = pd.read_csv(configured_path(config, "processed_dir") / "soil_selected_features.csv")

    panel = yield_df.merge(weather, on=["region", "year_start"], how="left", validate="many_to_many")
    panel = panel.merge(soil, on="region", how="left", validate="many_to_one")
    panel["panel_key"] = (
        panel["region"].astype(str)
        + " | "
        + panel["crop"].astype(str)
        + " | "
        + panel["year_start"].astype(str)
        + " | "
        + panel["forecast_window"].astype(str)
    )

    panel = panel.sort_values(["year_start", "region", "crop", "window_order"]).reset_index(drop=True)
    panel.insert(0, "row_id", np.arange(1, len(panel) + 1))

    output_path = configured_path(config, "processed_dir") / "model_ready_panel.csv"
    panel.to_csv(output_path, index=False)
    thresholds.to_csv(configured_path(config, "tables_dir") / "low_yield_thresholds_by_crop.csv", index=False)

    missing_weather = int(panel["forecast_window"].isna().sum())
    expected_rows = len(yield_df) * len(config["forecast_windows"])
    split_counts = panel.groupby(["split", "forecast_window"], dropna=False).size().reset_index(name="rows")
    risk_summary = (
        panel.groupby(["crop", "region"], as_index=False)
        .agg(
            observed_rows=("yield_t_ha", "size"),
            low_yield_rate=("low_yield_risk", "mean"),
            mean_yield_t_ha=("yield_t_ha", "mean"),
            mean_shortfall=("yield_shortfall", "mean"),
        )
        .sort_values(["crop", "region"])
    )
    risk_summary.to_csv(configured_path(config, "tables_dir") / "crop_region_risk_summary.csv", index=False)

    report = f"""
# Model Ready Panel Report

- Output: `data/processed/model_ready_panel.csv`
- Yield observations: `{len(yield_df)}`
- Forecast windows per observation: `{len(config["forecast_windows"])}`
- Expected panel rows: `{expected_rows}`
- Actual panel rows: `{len(panel)}`
- Missing weather-merge rows: `{missing_weather}`

Targets are created from yield data only. `expected_yield_t_ha` is fit using train years only, and `low_yield_risk` thresholds are computed from train shortfall distributions by crop.

## Split Counts

{markdown_table(split_counts)}

## Low-Yield Thresholds

{markdown_table(thresholds)}

## Leakage Policy

`production_kt` and `area_000ha` are retained as metadata for auditing, but training code excludes them from feature matrices by default.
"""
    write_markdown(configured_path(config, "reports_dir") / "model_ready_panel_report.md", report)
    print(f"Wrote {output_path} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
