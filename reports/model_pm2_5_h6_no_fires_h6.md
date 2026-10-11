# Forecast model — pm2_5, +6h (MD-03, MD-04)

Generated 2026-10-11 by `pipeline/train.py`.

- Target: **pm2_5**, **6** hours ahead
- Stations: **43** · features: **134**
- Station codes: `DL01..DL43 (43 stations)`
- Rolling-origin folds: **3**, 30 test days each
- Runtime: **46.7 min**

## Headline

| Model | MAE (µg/m³) | RMSE | Skill vs persistence |
|---|---:|---:|---:|
| persistence | 63.7 | 97.2 | — |
| climatology | 71.2 | 99.4 | -11.7% |
| persistence+climatology | 54.9 | 83.2 | +13.8% |
| **lightgbm** | 41.4 | 70.1 | +35.0% |

The model beats persistence by **+35.0%** MAE.

## Prediction interval

- 80% band (p10–p90) contains the truth **75.9%** of the time
- Target is 80%. Well calibrated.

## Per fold

| Fold | Test period | Train rows | Test rows | Persistence MAE | Model MAE | Skill |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-10-02 → 2024-11-01 | 2,041,616 | 28,557 | 49.6 | 27.6 | +44.3% |
| 2 | 2024-11-01 → 2024-12-01 | 2,070,173 | 29,373 | 82.3 | 59.7 | +27.5% |
| 3 | 2024-12-01 → 2024-12-31 | 2,099,546 | 29,287 | 59.3 | 36.9 | +37.8% |

## By season

Winter is when north India's air is worst and a forecast matters most.

| Season | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| post-monsoon | 57,251 | 66.2 | 44.0 | +33.6% |
| winter | 29,966 | 59.3 | 35.1 | +40.8% |

## High-pollution episodes (actual ≥ 120 µg/m³)

The hours the service exists for. An overall average hides failure here.

- Episode hours tested: **45,780**
- Persistence MAE: **84.2** · model MAE: **57.3** · skill **+31.9%**

Errors are much larger here than on the overall average. That is honest and expected: spikes are driven by stubble burning and festivals, which nothing in these features sees.

## Performance by what was knowable at prediction time

These splits use only the current reading, so a conclusion can rest on them.

| Conditioned on | Rows | Persistence MAE | Model MAE | Skill |
|---|---:|---:|---:|---:|
| all hours | 87,217 | 63.7 | 41.4 | +35.0% |
| calm now (< 120) | 40,688 | 43.7 | 26.6 | +39.0% |
| elevated now (>= 120) | 45,012 | 82.9 | 52.5 | +36.6% |

### Episode onset detection

Recall is a legitimate use of the outcome, because here it is the label rather than a filter on which rows get measured.

- True onsets in the test windows: **9,192**
- Model recall: **66.5%** · precision: **73.5%**
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
| onset — calm now, episode in 6h | 9,801 | 102.8 | 55.1 |
| ongoing — episode now, still an episode | 35,979 | 78.4 | 57.5 |
| decay — episode now, calm by then | 9,033 | 94.9 | 34.1 |
| calm — calm now, still calm | 32,404 | 22.5 | 17.5 |

## What the model leans on

| Feature | Gain |
|---|---:|
| pm2_5_mean24 | 2,609,810 |
| pm2_5_mean168 | 617,533 |
| wx_boundary_layer_m | 402,189 |
| pm2_5_lag1 | 249,193 |
| hour_cos | 125,786 |
| wx_humidity_pct | 111,863 |
| wx_temp_c | 97,794 |
| doy_sin | 97,048 |
| hour_sin | 88,706 |
| wx_wind_speed_ms | 85,954 |
| doy_cos | 71,185 |
| fc_boundary_layer_m | 53,181 |
| fc_temp_c | 41,486 |
| fc_humidity_pct | 33,551 |
| pm2_5_std168 | 30,633 |
| pm2_5_std24 | 29,454 |
| pm10_lag1 | 29,169 |
| latitude | 26,557 |
| longitude | 21,980 |
| rh_lag1 | 19,876 |

## Limitations

- Trained on ERA5 reanalysis weather but will run on *forecast* weather. That mismatch
  costs accuracy in production and is not measured here.
- Stubble burning and Diwali drive the worst episodes and are invisible to these features.
  NASA FIRMS fire counts are the fix.
- Evaluated on 43 stations; national behaviour may differ.
