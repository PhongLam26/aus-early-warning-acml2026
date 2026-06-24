from __future__ import annotations

import pandas as pd

from common import clean_name, configured_path, ensure_output_dirs, load_config, write_markdown


def weighted_depth_average(group: pd.DataFrame, upper: float, lower: float) -> float:
    values = []
    weights = []
    for _, row in group.iterrows():
        row_upper = float(row["upper_depth_cm"])
        row_lower = float(row["lower_depth_cm"])
        overlap = max(0.0, min(row_lower, lower) - max(row_upper, upper))
        if overlap > 0:
            values.append(float(row["value"]))
            weights.append(overlap)
    if not weights:
        return float("nan")
    return float(sum(v * w for v, w in zip(values, weights)) / sum(weights))


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)

    soil = pd.read_csv(configured_path(config, "soil_long"))
    soil = soil[soil["region"].isin(config["regions"])].copy()
    soil["soil_attribute"] = soil["soil_attribute"].astype(str)

    rows = []
    for region, region_df in soil.groupby("region", sort=True):
        out = {
            "region": region,
            "soil_lat": float(region_df["lat"].mean()),
            "soil_lon": float(region_df["lon"].mean()),
        }
        for attr in config["soil"]["attributes"]:
            attr_df = region_df[region_df["soil_attribute"] == attr]
            attr_name = clean_name(attr)
            for depth in config["soil"]["depth_groups"]:
                out[f"soil_{attr_name}_{depth['name']}"] = weighted_depth_average(
                    attr_df,
                    float(depth["upper_cm"]),
                    float(depth["lower_cm"]),
                )

        for attr in config["soil"]["depth_attributes"]:
            attr_df = region_df[region_df["soil_attribute"] == attr]
            if attr_df.empty:
                value = float("nan")
            else:
                value = float(attr_df["value"].mean())
            out[f"soil_{clean_name(attr)}"] = value
        rows.append(out)

    selected = pd.DataFrame(rows).sort_values("region")
    output_path = configured_path(config, "processed_dir") / "soil_selected_features.csv"
    selected.to_csv(output_path, index=False)

    report = f"""
# Soil Selected Feature Report

- Output: `data/processed/soil_selected_features.csv`
- Rows written: `{len(selected)}`
- Regions: `{", ".join(selected["region"].tolist())}`
- Feature columns: `{len(selected.columns) - 1}`

Soil features are aggregated to regional background indicators. Topsoil is 0-30 cm and subsoil is 30-100 cm where depth-specific values are available. These variables should be interpreted as vulnerability-conditioning features, not as causal farm-level effects.
"""
    write_markdown(configured_path(config, "reports_dir") / "soil_selected_features_report.md", report)
    print(f"Wrote {output_path} ({len(selected)} rows)")


if __name__ == "__main__":
    main()
