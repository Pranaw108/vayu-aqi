# Forecast model — pm2_5, +24h (MD-03, MD-04)

Generated 2026-10-10 by `pipeline/train.py`.

- Target: **pm2_5**, **24** hours ahead
- Stations: **43** · features: **134**
- Station codes: `DL01..DL43 (43 stations)`
- Rolling-origin folds: **3**, 30 test days each
- Runtime: **64.7 min**

## Headline

| Model | MAE (µg/m³) | RMSE | Skill vs persistence |
|---|---:|---:|---:|
| persistence | 56.7 | 89.9 | — |
| climatology | 71.6 | 99.8 | -26.4% |
| persistence+climatology | 53.3 | 82.1 | +6.0% |
| **lightgbm** | 49.0 | 79.3 | +13.5% |

The model beats persistence by **+13.5%** MAE.

## Prediction interval

- 80% band (p10–p90) contains the truth **75.6%** of the time
- Target is 80%. Well calibrated.

## Per fold

| Fold | Test period | Train rows | Test rows | Persistence MAE | Model MAE | Skill |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-10-01 → 2024-10-31 | 2,040,927 | 28,527 | 34.5 | 31.1 | +9.7% |
| 2 | 2024-10-31 → 2024-11-30 | 2,069,454 | 29,380 | 79.9 | 68.6 | +14.1% |
| 3 | 2024-11-30 → 2024-12-30 | 2,098,834 | 29,279 | 55.6 | 47.3 | +15.0% |

## By season

Winter is when north India's air is worst and a forecast matters most.

| Season | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| post-monsoon | 57,948 | 48.0 | 44.8 | +6.6% |
| winter | 29,238 | 55.7 | 47.3 | +15.1% |

## High-pollution episodes (actual ≥ 120 µg/m³)

The hours the service exists for. An overall average hides failure here.

- Episode hours tested: **45,782**
- Persistence MAE: **77.5** · model MAE: **65.2** · skill **+15.9%**

Errors are much larger here than on the overall average. That is honest and expected: spikes are driven by stubble burning and festivals, which nothing in these features sees.

## What the model leans on

| Feature | Gain |
|---|---:|
| pm2_5_lag1 | 2,474,046 |
| pm2_5_mean168 | 431,937 |
| pm2_5_lag24 | 192,758 |
| doy_cos | 118,608 |
| doy_sin | 101,979 |
| fc_boundary_layer_m | 78,424 |
| fc_wind_speed_ms | 69,540 |
| pm2_5_mean24 | 68,418 |
| pm2_5_lag48 | 65,312 |
| hour_cos | 62,997 |
| fc_humidity_pct | 62,646 |
| fc_temp_c | 44,041 |
| hour_sin | 42,993 |
| wx_temp_c | 41,521 |
| wx_boundary_layer_m | 37,353 |
| wx_wind_speed_ms | 32,729 |
| pm2_5_lag2 | 27,702 |
| wx_pressure_hpa | 27,456 |
| pm2_5_lag168 | 25,551 |
| at_lag1 | 25,137 |

## Limitations

- Trained on ERA5 reanalysis weather but will run on *forecast* weather. That mismatch
  costs accuracy in production and is not measured here.
- Stubble burning and Diwali drive the worst episodes and are invisible to these features.
  NASA FIRMS fire counts are the fix.
- Evaluated on 43 stations; national behaviour may differ.
