# Forecast model — pm2_5, +6h (MD-03, MD-04)

Generated 2026-10-11 by `pipeline/train.py`.

- Target: **pm2_5**, **6** hours ahead
- Stations: **43** · features: **140**
- Station codes: `DL01..DL43 (43 stations)`
- Rolling-origin folds: **3**, 30 test days each
- Runtime: **39.3 min**

## Headline

| Model | MAE (µg/m³) | RMSE | Skill vs persistence |
|---|---:|---:|---:|
| persistence | 63.7 | 97.2 | — |
| climatology | 71.2 | 99.4 | -11.7% |
| persistence+climatology | 54.9 | 83.2 | +13.8% |
| **lightgbm** | 41.6 | 70.5 | +34.7% |

The model beats persistence by **+34.7%** MAE.

## Prediction interval

- 80% band (p10–p90) contains the truth **75.4%** of the time
- Target is 80%. Well calibrated.

## Per fold

| Fold | Test period | Train rows | Test rows | Persistence MAE | Model MAE | Skill |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-10-02 → 2024-11-01 | 2,041,616 | 28,557 | 49.6 | 27.9 | +43.7% |
| 2 | 2024-11-01 → 2024-12-01 | 2,070,173 | 29,373 | 82.3 | 59.6 | +27.6% |
| 3 | 2024-12-01 → 2024-12-31 | 2,099,546 | 29,287 | 59.3 | 37.2 | +37.2% |

## By season

Winter is when north India's air is worst and a forecast matters most.

| Season | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| post-monsoon | 57,251 | 66.2 | 44.1 | +33.4% |
| winter | 29,966 | 59.3 | 35.0 | +41.1% |

## High-pollution episodes (actual ≥ 120 µg/m³)

The hours the service exists for. An overall average hides failure here.

- Episode hours tested: **45,780**
- Persistence MAE: **84.2** · model MAE: **58.3** · skill **+30.7%**

Errors are much larger here than on the overall average. That is honest and expected: spikes are driven by stubble burning and festivals, which nothing in these features sees.

## Performance by what was knowable at prediction time

These splits use only the current reading, so a conclusion can rest on them.

| Conditioned on | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| all hours | 87,217 | 63.7 | 41.6 | +34.7% |
| calm now (< 120) | 40,688 | 43.7 | 26.8 | +38.6% |
| elevated now (>= 120) | 45,012 | 82.9 | 52.8 | +36.3% |

### Episode onset detection

Recall is a legitimate use of the outcome, because here it is the label rather than a filter on which rows get measured.

- True onsets in the test windows: **9,192**
- Model recall: **64.9%** · precision: **73.8%**
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
| onset — calm now, episode in 6h | 9,801 | 102.8 | 55.8 |
| ongoing — episode now, still an episode | 35,979 | 78.4 | 58.6 |
| decay — episode now, calm by then | 9,033 | 94.9 | 32.4 |
| calm — calm now, still calm | 32,404 | 22.5 | 17.7 |

## What the model leans on

| Feature | Gain |
|---|---:|
| pm2_5_mean24 | 2,571,990 |
| pm2_5_mean168 | 672,370 |
| wx_boundary_layer_m | 378,687 |
| pm2_5_lag1 | 229,767 |
| hour_cos | 129,186 |
| wx_temp_c | 118,106 |
| wx_humidity_pct | 102,806 |
| wx_wind_speed_ms | 91,961 |
| hour_sin | 90,835 |
| doy_sin | 85,335 |
| doy_cos | 61,242 |
| fc_boundary_layer_m | 49,237 |
| fire_count_upwind | 43,139 |
| fc_temp_c | 42,999 |
| fire_count_upwind_d1 | 36,191 |
| fire_count_100_300km_d1 | 31,946 |
| pm10_lag1 | 29,514 |
| fc_humidity_pct | 27,752 |
| latitude | 27,680 |
| pm2_5_std168 | 26,677 |

## Limitations

- Trained on ERA5 reanalysis weather but will run on *forecast* weather. That mismatch
  costs accuracy in production and is not measured here.
- Stubble burning and Diwali drive the worst episodes and are invisible to these features.
  NASA FIRMS fire counts are the fix.
- Evaluated on 43 stations; national behaviour may differ.
