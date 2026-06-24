from __future__ import annotations

import pandas as pd

from common import configured_path, ensure_output_dirs, load_config, markdown_table, write_markdown


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        rows.append(
            {
                "column": column,
                "missing": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 3),
                "dtype": str(df[column].dtype),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    return markdown_table(df)


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)

    yield_df = pd.read_csv(configured_path(config, "yield_panel"))
    weather_df = pd.read_csv(configured_path(config, "daily_weather"), parse_dates=["date"])
    soil_wide = pd.read_csv(configured_path(config, "soil_wide"))
    soil_long = pd.read_csv(configured_path(config, "soil_long"))

    yield_expected = {"season", "year_start", "region", "crop", "area_000ha", "production_kt", "yield_t_ha"}
    weather_expected = {"region", "lat", "lon", "date", "rain_mm", "tmax_c", "tmin_c", "radiation_mj_m2", "vp_hpa", "evap_mm"}
    soil_long_expected = {
        "region",
        "lat",
        "lon",
        "soil_attribute",
        "unit",
        "upper_depth_cm",
        "lower_depth_cm",
        "value",
        "lower_uncertainty",
        "upper_uncertainty",
    }

    yield_df["coverage_key"] = (
        yield_df["region"].astype(str)
        + " | "
        + yield_df["crop"].astype(str)
        + " | "
        + yield_df["year_start"].astype(str)
    )
    duplicated_keys = yield_df[yield_df["coverage_key"].duplicated(keep=False)]

    crop_counts = yield_df["crop"].value_counts().sort_index().rename_axis("crop").reset_index(name="rows")
    region_counts = yield_df["region"].value_counts().sort_index().rename_axis("region").reset_index(name="rows")
    crop_region_counts = (
        yield_df.groupby(["crop", "region"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["crop", "region"])
    )

    weather_df["year"] = weather_df["date"].dt.year
    weather_counts = (
        weather_df.groupby(["region", "year"], as_index=False)
        .size()
        .rename(columns={"size": "daily_rows"})
        .sort_values(["region", "year"])
    )
    weather_year_summary = (
        weather_counts.groupby("region", as_index=False)
        .agg(min_year=("year", "min"), max_year=("year", "max"), region_years=("year", "nunique"))
        .sort_values("region")
    )

    notes = []
    if set(config["crops"]) != set(yield_df["crop"].unique()):
        notes.append("- Config crop list differs from yield data crop list.")
    if set(config["regions"]) != set(yield_df["region"].unique()):
        notes.append("- Config region list differs from yield data region list.")
    if duplicated_keys.empty:
        notes.append("- No duplicate `region + crop + year_start` rows in yield data.")
    else:
        notes.append(f"- Found {len(duplicated_keys)} duplicated yield coverage keys.")

    report = f"""
# Data Audit Report

## Dataset Shapes

| Dataset | Rows | Columns |
|---|---:|---:|
| yield_panel.csv | {len(yield_df)} | {yield_df.shape[1] - 1} |
| australia_silo_daily_all_states_1989_present.csv | {len(weather_df)} | {weather_df.shape[1] - 1} |
| soil_features_by_region.csv | {len(soil_wide)} | {soil_wide.shape[1]} |
| soil_features_by_region_long.csv | {len(soil_long)} | {soil_long.shape[1]} |

## Schema Checks

- Yield missing expected columns: `{sorted(yield_expected - set(yield_df.columns))}`
- Weather missing expected columns: `{sorted(weather_expected - set(weather_df.columns))}`
- Soil long missing expected columns: `{sorted(soil_long_expected - set(soil_long.columns))}`

## Yield Scope

- Year range: `{int(yield_df["year_start"].min())}` to `{int(yield_df["year_start"].max())}`
- Crops: `{", ".join(sorted(yield_df["crop"].unique()))}`
- Regions: `{", ".join(sorted(yield_df["region"].unique()))}`
- Analysis unit: `region + crop + year_start`

### Crop Counts

{md_table(crop_counts)}

### Region Counts

{md_table(region_counts)}

### Crop-Region Coverage

{md_table(crop_region_counts)}

## Weather Scope

- Date range: `{weather_df["date"].min().date()}` to `{weather_df["date"].max().date()}`
- Weather years are available beyond yield years; model panel will only use weather years matching yield `year_start`.

{md_table(weather_year_summary)}

## Soil Scope

- Wide soil rows: `{len(soil_wide)}`
- Long soil rows: `{len(soil_long)}`
- Soil attributes: `{", ".join(sorted(soil_long["soil_attribute"].unique()))}`

## Missing Values

### Yield

{md_table(missing_summary(yield_df.drop(columns=["coverage_key"])))}

### Weather

{md_table(missing_summary(weather_df.drop(columns=["year"])))}

### Soil Long

{md_table(missing_summary(soil_long))}

## Notes

{chr(10).join(notes)}
"""

    reports_dir = configured_path(config, "reports_dir")
    tables_dir = configured_path(config, "tables_dir")
    write_markdown(reports_dir / "data_audit.md", report)
    crop_region_counts.to_csv(tables_dir / "yield_crop_region_coverage.csv", index=False)
    weather_year_summary.to_csv(tables_dir / "weather_region_year_summary.csv", index=False)
    print(f"Wrote {reports_dir / 'data_audit.md'}")


if __name__ == "__main__":
    main()
