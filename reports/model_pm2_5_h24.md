# Forecast model — pm2_5, +24h (MD-03, MD-04)

Generated 2026-10-10 by `pipeline/train.py`.

- Target: **pm2_5**, **24** hours ahead
- Stations: **7** · features: **128**
- Rolling-origin folds: **3**, 30 test days each
- Runtime: **14.4 min**

## Headline

| Model | MAE (µg/m³) | RMSE | Skill vs persistence |
|---|---:|---:|---:|
| persistence | 26.1 | 45.5 | — |
| climatology | 33.9 | 51.2 | -29.9% |
| persistence+climatology | 24.7 | 42.0 | +5.4% |
| **lightgbm** | 23.8 | 40.9 | +8.9% |

The model beats persistence by **+8.9%** MAE.

## Prediction interval

- 80% band (p10–p90) contains the truth **74.6%** of the time
- Target is 80%. Not calibrated — the published band is misleading.

## Per fold

| Fold | Test period | Train rows | Test rows | Persistence MAE | Model MAE | Skill |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-10-01 → 2024-10-31 | 239,439 | 3,056 | 23.2 | 21.6 | +7.0% |
| 2 | 2024-10-31 → 2024-11-30 | 242,495 | 3,416 | 29.2 | 26.3 | +9.9% |
| 3 | 2024-11-30 → 2024-12-30 | 245,911 | 3,030 | 25.9 | 23.5 | +9.6% |

## By season

Winter is when north India's air is worst and a forecast matters most.

| Season | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| post-monsoon | 6,477 | 22.8 | 23.5 | -3.0% |
| winter | 3,025 | 26.0 | 23.5 | +9.6% |

## High-pollution episodes (actual ≥ 120 µg/m³)

The hours the service exists for. An overall average hides failure here.

- Episode hours tested: **1,203**
- Persistence MAE: **73.2** · model MAE: **75.0** · skill **-2.5%**

Errors are much larger here than on the overall average. That is honest and expected: spikes are driven by stubble burning and festivals, which nothing in these features sees.

## What the model leans on

| Feature | Gain |
|---|---:|
| pm2_5_mean24 | 241,753 |
| pm2_5_lag1 | 105,597 |
| pm2_5_mean168 | 20,385 |
| pm10_lag1 | 12,263 |
| doy_cos | 7,428 |
| pm2_5_lag24 | 5,828 |
| pm2_5_mean6 | 5,608 |
| at_lag1 | 4,925 |
| doy_sin | 4,331 |
| ws_lag1 | 2,621 |
| pm2_5_std168 | 2,273 |
| pm2_5_lag48 | 2,268 |
| rh_lag1 | 2,210 |
| o3_mean168 | 2,188 |
| pm2_5_lag168 | 2,120 |
| pm10_mean168 | 2,082 |
| no2_lag1 | 1,994 |
| co_std168 | 1,965 |
| so2_mean168 | 1,964 |
| o3_std168 | 1,901 |

## Limitations

- Trained on ERA5 reanalysis weather but will run on *forecast* weather. That mismatch
  costs accuracy in production and is not measured here.
- Stubble burning and Diwali drive the worst episodes and are invisible to these features.
  NASA FIRMS fire counts are the fix.
- Evaluated on 7 stations; national behaviour may differ.
