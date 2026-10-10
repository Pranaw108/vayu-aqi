# Forecast model — pm2_5, +24h (MD-03, MD-04)

Generated 2026-10-11 by `pipeline/train.py`.

- Target: **pm2_5**, **24** hours ahead
- Stations: **43** · features: **134**
- Station codes: `DL01..DL43 (43 stations)`
- Rolling-origin folds: **3**, 30 test days each
- Runtime: **46.8 min**

## Headline

| Model | MAE (µg/m³) | RMSE | Skill vs persistence |
|---|---:|---:|---:|
| persistence | 56.7 | 89.9 | — |
| climatology | 71.6 | 99.8 | -26.4% |
| persistence+climatology | 53.3 | 82.1 | +6.0% |
| **lightgbm** | 49.1 | 79.1 | +13.3% |

The model beats persistence by **+13.3%** MAE.

## Prediction interval

- 80% band (p10–p90) contains the truth **75.4%** of the time
- Target is 80%. Well calibrated.

## Per fold

| Fold | Test period | Train rows | Test rows | Persistence MAE | Model MAE | Skill |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-10-01 → 2024-10-31 | 2,040,927 | 28,527 | 34.5 | 31.2 | +9.4% |
| 2 | 2024-10-31 → 2024-11-30 | 2,069,454 | 29,380 | 79.9 | 68.3 | +14.5% |
| 3 | 2024-11-30 → 2024-12-30 | 2,098,834 | 29,279 | 55.6 | 47.8 | +14.1% |

## By season

Winter is when north India's air is worst and a forecast matters most.

| Season | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| post-monsoon | 57,948 | 48.0 | 44.9 | +6.5% |
| winter | 29,238 | 55.7 | 47.8 | +14.1% |

## High-pollution episodes (actual ≥ 120 µg/m³)

The hours the service exists for. An overall average hides failure here.

- Episode hours tested: **45,782**
- Persistence MAE: **77.5** · model MAE: **65.3** · skill **+15.7%**

Errors are much larger here than on the overall average. That is honest and expected: spikes are driven by stubble burning and festivals, which nothing in these features sees.

## Performance by what was knowable at prediction time

These splits use only the current reading, so a conclusion can rest on them.

| Conditioned on | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| all hours | 87,186 | 56.7 | 49.1 | +13.3% |
| calm now (< 120) | 40,829 | 32.5 | 29.9 | +7.9% |
| elevated now (>= 120) | 44,478 | 76.7 | 65.7 | +14.4% |

### Episode onset detection

Recall is a legitimate use of the outcome, because here it is the label rather than a filter on which rows get measured.

- True onsets in the test windows: **7,508**
- Model recall: **42.6%** · precision: **52.4%**
- Persistence recall: **0%** — it cannot call an onset by construction

Raising recall is the clearest remaining job, and it is a detection problem rather than a calibration one.

## Transitions (diagnostic only — do NOT read bias from these)

> These rows are selected **by the outcome**, so they cannot show whether the
> model is biased. An unbiased model always looks like it under-predicts rows
> chosen for having a high outcome and over-predicts rows chosen for a low one.
> An earlier version of this report drew a bias conclusion from exactly these
> numbers and it was wrong. They are kept because the error *magnitudes* are
> still useful for sizing the problem.

| Transition | Rows | Persistence MAE | Model MAE |
|---|---:|---:|---:|
| onset — calm now, episode in 24h | 8,300 | 85.4 | 63.0 |
| ongoing — episode now, still an episode | 37,482 | 75.8 | 66.2 |
| decay — episode now, calm by then | 6,996 | 78.3 | 62.3 |
| calm — calm now, still calm | 34,408 | 18.1 | 21.3 |

## What the model leans on

| Feature | Gain |
|---|---:|
| pm2_5_lag1 | 2,472,712 |
| pm2_5_mean168 | 408,882 |
| pm2_5_lag24 | 196,981 |
| doy_cos | 106,346 |
| doy_sin | 76,185 |
| fc_boundary_layer_m | 75,195 |
| pm2_5_mean24 | 73,547 |
| pm2_5_lag48 | 70,601 |
| fc_wind_speed_ms | 66,816 |
| hour_cos | 58,848 |
| fc_humidity_pct | 53,829 |
| hour_sin | 41,998 |
| fc_temp_c | 36,084 |
| wx_temp_c | 35,412 |
| wx_boundary_layer_m | 32,065 |
| wx_wind_speed_ms | 29,507 |
| pm2_5_lag2 | 27,440 |
| pm2_5_lag168 | 26,764 |
| wx_pressure_hpa | 25,829 |
| at_lag1 | 24,026 |

## Limitations

- Trained on ERA5 reanalysis weather but will run on *forecast* weather. That mismatch
  costs accuracy in production and is not measured here.
- Stubble burning and Diwali drive the worst episodes and are invisible to these features.
  NASA FIRMS fire counts are the fix.
- Evaluated on 43 stations; national behaviour may differ.
