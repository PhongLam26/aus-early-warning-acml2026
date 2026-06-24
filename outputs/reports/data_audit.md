# Data Audit Report

## Dataset Shapes

| Dataset | Rows | Columns |
|---|---:|---:|
| yield_panel.csv | 966 | 7 |
| australia_silo_daily_all_states_1989_present.csv | 82008 | 10 |
| soil_features_by_region.csv | 6 | 65 |
| soil_features_by_region_long.csv | 372 | 10 |

## Schema Checks

- Yield missing expected columns: `[]`
- Weather missing expected columns: `[]`
- Soil long missing expected columns: `[]`

## Yield Scope

- Year range: `1989` to `2021`
- Crops: `Barley, Canola, Lupins, Oats, Wheat`
- Regions: `New South Wales, Queensland, South Australia, Tasmania, Victoria, Western Australia`
- Analysis unit: `region + crop + year_start`

### Crop Counts

| crop | rows |
| --- | --- |
| Barley | 198 |
| Canola | 195 |
| Lupins | 177 |
| Oats | 198 |
| Wheat | 198 |

### Region Counts

| region | rows |
| --- | --- |
| New South Wales | 165 |
| Queensland | 153 |
| South Australia | 165 |
| Tasmania | 153 |
| Victoria | 165 |
| Western Australia | 165 |

### Crop-Region Coverage

| crop | region | rows |
| --- | --- | --- |
| Barley | New South Wales | 33 |
| Barley | Queensland | 33 |
| Barley | South Australia | 33 |
| Barley | Tasmania | 33 |
| Barley | Victoria | 33 |
| Barley | Western Australia | 33 |
| Canola | New South Wales | 33 |
| Canola | Queensland | 30 |
| Canola | South Australia | 33 |
| Canola | Tasmania | 33 |
| Canola | Victoria | 33 |
| Canola | Western Australia | 33 |
| Lupins | New South Wales | 33 |
| Lupins | Queensland | 24 |
| Lupins | South Australia | 33 |
| Lupins | Tasmania | 21 |
| Lupins | Victoria | 33 |
| Lupins | Western Australia | 33 |
| Oats | New South Wales | 33 |
| Oats | Queensland | 33 |
| Oats | South Australia | 33 |
| Oats | Tasmania | 33 |
| Oats | Victoria | 33 |
| Oats | Western Australia | 33 |
| Wheat | New South Wales | 33 |
| Wheat | Queensland | 33 |
| Wheat | South Australia | 33 |
| Wheat | Tasmania | 33 |
| Wheat | Victoria | 33 |
| Wheat | Western Australia | 33 |

## Weather Scope

- Date range: `1989-01-01` to `2026-06-03`
- Weather years are available beyond yield years; model panel will only use weather years matching yield `year_start`.

| region | min_year | max_year | region_years |
| --- | --- | --- | --- |
| New South Wales | 1989 | 2026 | 38 |
| Queensland | 1989 | 2026 | 38 |
| South Australia | 1989 | 2026 | 38 |
| Tasmania | 1989 | 2026 | 38 |
| Victoria | 1989 | 2026 | 38 |
| Western Australia | 1989 | 2026 | 38 |

## Soil Scope

- Wide soil rows: `6`
- Long soil rows: `372`
- Soil attributes: `AWC, BULK-DENSITY, CLAY, DEPTH_OF_REGOLITH, DEPTH_OF_SOIL, ECEC, PHC, SAND, SILT, SOC, TOTAL_N, TOTAL_P`

## Missing Values

### Yield

| column | missing | missing_pct | dtype |
| --- | --- | --- | --- |
| season | 0 | 0.0 | object |
| year_start | 0 | 0.0 | int64 |
| region | 0 | 0.0 | object |
| crop | 0 | 0.0 | object |
| area_000ha | 0 | 0.0 | float64 |
| production_kt | 0 | 0.0 | float64 |
| yield_t_ha | 0 | 0.0 | float64 |

### Weather

| column | missing | missing_pct | dtype |
| --- | --- | --- | --- |
| region | 0 | 0.0 | object |
| lat | 0 | 0.0 | float64 |
| lon | 0 | 0.0 | float64 |
| date | 0 | 0.0 | datetime64[ns] |
| rain_mm | 0 | 0.0 | float64 |
| tmax_c | 0 | 0.0 | float64 |
| tmin_c | 0 | 0.0 | float64 |
| radiation_mj_m2 | 0 | 0.0 | float64 |
| vp_hpa | 0 | 0.0 | float64 |
| evap_mm | 0 | 0.0 | float64 |

### Soil Long

| column | missing | missing_pct | dtype |
| --- | --- | --- | --- |
| region | 0 | 0.0 | object |
| lat | 0 | 0.0 | float64 |
| lon | 0 | 0.0 | float64 |
| soil_attribute | 0 | 0.0 | object |
| unit | 48 | 12.903 | object |
| upper_depth_cm | 0 | 0.0 | float64 |
| lower_depth_cm | 0 | 0.0 | float64 |
| value | 0 | 0.0 | float64 |
| lower_uncertainty | 0 | 0.0 | float64 |
| upper_uncertainty | 0 | 0.0 | float64 |

## Notes

- No duplicate `region + crop + year_start` rows in yield data.
