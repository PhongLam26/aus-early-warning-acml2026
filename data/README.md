# Data Folder

This folder contains the compact data artifacts used by the Australian winter-crop early-warning project.

## Structure

```text
data/
  raw/
    daily_weather/
    soil/
    yield/
  processed/
```

## Raw Data

- `raw/yield/yield_panel.csv`: 966 region-crop-year yield observations.
- `raw/yield/03_AustCropRrt20260303_StateCropData_v1.0.0.xlsx`: original yield workbook kept for auditability.
- `raw/daily_weather/australia_silo_daily_all_states_1989_present.csv`: merged daily weather table for six Australian regions.
- `raw/daily_weather/silo_daily_*_1989_present.csv`: per-region daily weather files.
- `raw/soil/soil_features_by_region.csv`: wide regional soil table.
- `raw/soil/soil_features_by_region_long.csv`: long regional soil table.

The raw weather cache by region-year is excluded from git because it is redundant and can be regenerated from the per-region weather files.

## Processed Data

- `processed/weather_stage_features.csv`: May-Jun to May-Oct weather-stage features by `region`, `year_start`, and `forecast_window`.
- `processed/soil_selected_features.csv`: selected regional soil background features.
- `processed/model_ready_panel.csv`: initial modeling panel.
- `processed/model_ready_panel_improved.csv`: main improved modeling panel used by the ACML manuscript.

The main analysis unit is:

```text
region + crop + year_start + forecast_window
```

The improved panel has 4,830 forecast-window rows from 966 unique region-crop-year yield observations. `production_kt` and `area_000ha` are retained only as metadata and are excluded from the training feature matrices and manuscript evidence.
