from __future__ import annotations

import pandas as pd

from common import (
    configured_path,
    ensure_output_dirs,
    load_config,
    max_consecutive_true,
    rolling_sum_max,
    write_markdown,
)


def summarize_window(df: pd.DataFrame, thresholds: dict) -> dict[str, float]:
    rain_day = float(thresholds["rain_day_mm"])
    heat_25 = float(thresholds["heat_25_c"])
    heat_30 = float(thresholds["heat_30_c"])
    heat_35 = float(thresholds["heat_35_c"])
    frost_c = float(thresholds["frost_c"])
    cold_c = float(thresholds["cold_c"])
    high_evap = float(thresholds["high_evap_mm"])

    rain = df["rain_mm"]
    tmax = df["tmax_c"]
    tmin = df["tmin_c"]
    evap = df["evap_mm"]

    dry_mask = rain < rain_day
    hot_30_mask = tmax >= heat_30
    high_evap_mask = evap >= high_evap

    return {
        "n_weather_days": int(len(df)),
        "lat": float(df["lat"].mean()),
        "lon": float(df["lon"].mean()),
        "rain_sum": float(rain.sum()),
        "rain_mean": float(rain.mean()),
        "rain_days": int((rain >= rain_day).sum()),
        "dry_days": int(dry_mask.sum()),
        "max_3day_rain": rolling_sum_max(rain, 3),
        "max_7day_rain": rolling_sum_max(rain, 7),
        "max_consecutive_dry_days": max_consecutive_true(dry_mask),
        "heavy_rain_days_10": int((rain >= float(thresholds["heavy_rain_10_mm"])).sum()),
        "heavy_rain_days_25": int((rain >= float(thresholds["heavy_rain_25_mm"])).sum()),
        "tmax_mean": float(tmax.mean()),
        "tmax_max": float(tmax.max()),
        "tmin_mean": float(tmin.mean()),
        "tmin_min": float(tmin.min()),
        "heat_days_25": int((tmax >= heat_25).sum()),
        "heat_days_30": int(hot_30_mask.sum()),
        "heat_days_35": int((tmax >= heat_35).sum()),
        "heat_degree_days_30": float((tmax - heat_30).clip(lower=0).sum()),
        "frost_days_0": int((tmin <= frost_c).sum()),
        "cold_days_5": int((tmin <= cold_c).sum()),
        "radiation_sum": float(df["radiation_mj_m2"].sum()),
        "radiation_mean": float(df["radiation_mj_m2"].mean()),
        "evap_sum": float(evap.sum()),
        "evap_mean": float(evap.mean()),
        "vp_mean": float(df["vp_hpa"].mean()),
        "vp_min": float(df["vp_hpa"].min()),
        "vp_max": float(df["vp_hpa"].max()),
        "hot_dry_days_30_rain_lt_1": int((hot_30_mask & dry_mask).sum()),
        "high_evap_dry_days": int((high_evap_mask & dry_mask).sum()),
        "heat_and_low_rain_days": int((hot_30_mask & dry_mask).sum()),
    }


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)

    weather = pd.read_csv(configured_path(config, "daily_weather"), parse_dates=["date"])
    weather["year_start"] = weather["date"].dt.year
    weather["month"] = weather["date"].dt.month

    split = config["splits"]
    weather = weather[
        (weather["year_start"] >= split["train_start"])
        & (weather["year_start"] <= split["test_end"])
        & (weather["region"].isin(config["regions"]))
    ].copy()

    rows = []
    for window in config["forecast_windows"]:
        mask = (weather["month"] >= int(window["start_month"])) & (
            weather["month"] <= int(window["end_month"])
        )
        subset = weather.loc[mask].copy()
        for (region, year_start), group in subset.groupby(["region", "year_start"], sort=True):
            summary = summarize_window(group.sort_values("date"), config["thresholds"])
            summary.update(
                {
                    "region": region,
                    "year_start": int(year_start),
                    "forecast_window": window["name"],
                    "window_start_month": int(window["start_month"]),
                    "window_end_month": int(window["end_month"]),
                    "window_order": int(window["order"]),
                }
            )
            rows.append(summary)

    features = pd.DataFrame(rows)
    key_cols = [
        "region",
        "year_start",
        "forecast_window",
        "window_start_month",
        "window_end_month",
        "window_order",
    ]
    value_cols = [c for c in features.columns if c not in key_cols]
    features = features[key_cols + value_cols].sort_values(
        ["region", "year_start", "window_order"]
    )

    output_path = configured_path(config, "processed_dir") / "weather_stage_features.csv"
    features.to_csv(output_path, index=False)

    expected_rows = len(config["regions"]) * (
        split["test_end"] - split["train_start"] + 1
    ) * len(config["forecast_windows"])
    report = f"""
# Weather Stage Feature Report

- Output: `data/processed/weather_stage_features.csv`
- Rows written: `{len(features)}`
- Expected rows: `{expected_rows}`
- Forecast windows: `{", ".join(w["name"] for w in config["forecast_windows"])}`
- Weather years used: `{split["train_start"]}` to `{split["test_end"]}`

Each row uses only daily weather from May through the cutoff month of the listed `forecast_window`.
"""
    write_markdown(configured_path(config, "reports_dir") / "weather_stage_features_report.md", report)
    print(f"Wrote {output_path} ({len(features)} rows)")


if __name__ == "__main__":
    main()
