# Features — pm2_5, +24h (MD-01)

Generated 2026-10-10 by `pipeline/features.py`.

- Target: **pm2_5**, **24** hours ahead
- Training window starts: **2018-01-01**
- Stations: **8**
- Rows with a known answer: **248,943**
- Features: **128**
- Rows with forecast weather: **100.0%**
- Build time: **0.2 min**

## Persistence baseline

The number any model has to beat. Predicting *the value right now* is a
surprisingly strong forecast for air quality, and a model that cannot beat it
has learned nothing.

- Mean absolute error: **19.2 µg/m³**
- Median absolute error: **11.0 µg/m³**
- RMSE: **36.5 µg/m³**

## Leakage guard

Columns dropped because they could not be known at prediction time:

- `at`
- `bp`
- `co`
- `nh3`
- `no2`
- `o3`
- `pm10`
- `pm2_5`
- `rh`
- `so2`
- `wd`
- `ws`

Every surviving feature is a lag, a rolling statistic over past hours only, a
calendar term, a station constant, or a weather forecast — which is genuinely
available ahead of time.

## Feature coverage

| Feature | % present |
|---|---:|
| rh_lag24 | 78.9 |
| rh_lag1 | 79.0 |
| bp_lag24 | 88.7 |
| bp_lag1 | 88.8 |
| wd_sin_lag24 | 88.8 |
| wd_cos_lag24 | 88.8 |
| wd_sin_lag1 | 88.9 |
| wd_cos_lag1 | 88.9 |
| at_lag24 | 89.2 |
| at_lag1 | 89.3 |
| ws_lag24 | 89.4 |
| ws_lag1 | 89.5 |
| boundary_layer_m | 91.6 |
| neighbour_pm2_5_lag1 | 91.8 |
| o3_lag168 | 95.3 |
| co_lag168 | 95.5 |
| o3_lag48 | 95.6 |
| o3_lag24 | 95.7 |
| o3_lag12 | 95.7 |
| o3_lag6 | 95.7 |
| o3_lag2 | 95.7 |
| o3_lag3 | 95.7 |
| o3_lag1 | 95.7 |
| co_lag48 | 95.8 |
| co_lag24 | 95.8 |
| co_lag12 | 95.9 |
| co_lag6 | 95.9 |
| co_lag3 | 95.9 |
| co_lag2 | 95.9 |
| co_lag1 | 95.9 |
| pm2_5_lag168 | 96.0 |
| co_mean6 | 96.3 |
| co_std6 | 96.3 |
| pm10_lag168 | 96.5 |
| co_mean24 | 96.5 |
| co_std24 | 96.5 |
| pm2_5_lag48 | 96.6 |
| so2_lag168 | 96.6 |
| pm2_5_lag24 | 96.7 |
| pm2_5_lag12 | 96.8 |
| nh3_lag168 | 96.8 |
| pm2_5_lag6 | 96.9 |
| pm10_lag48 | 96.9 |
| pm2_5_lag3 | 96.9 |
| pm2_5_lag2 | 97.0 |
| co_mean168 | 97.0 |
| co_std168 | 97.0 |
| pm10_lag12 | 97.0 |
| pm10_lag24 | 97.0 |
| so2_lag48 | 97.0 |
| pm2_5_lag1 | 97.0 |
| pm10_lag6 | 97.0 |
| o3_mean6 | 97.0 |
| o3_std6 | 97.0 |
| so2_lag24 | 97.1 |
| pm10_lag3 | 97.1 |
| pm10_lag2 | 97.1 |
| so2_lag12 | 97.1 |
| pm10_lag1 | 97.1 |
| so2_lag6 | 97.1 |
| so2_lag3 | 97.1 |
| so2_lag2 | 97.1 |
| so2_lag1 | 97.1 |
| nh3_lag48 | 97.2 |
| nh3_lag24 | 97.2 |
| nh3_lag12 | 97.2 |
| nh3_lag6 | 97.3 |
| nh3_lag3 | 97.3 |
| nh3_lag2 | 97.3 |
| nh3_lag1 | 97.3 |
| no2_lag168 | 97.6 |
| o3_mean24 | 97.7 |
| o3_std24 | 97.7 |
| o3_mean168 | 97.9 |
| o3_std168 | 97.9 |
| no2_lag48 | 97.9 |
| no2_lag24 | 98.0 |
| no2_lag12 | 98.0 |
| no2_lag6 | 98.0 |
| no2_lag3 | 98.0 |
| no2_lag2 | 98.0 |
| no2_lag1 | 98.0 |
| nh3_mean6 | 98.3 |
| nh3_std6 | 98.3 |
| nh3_mean24 | 98.8 |
| nh3_std24 | 98.8 |
| pm10_mean6 | 98.9 |
| pm10_std6 | 98.9 |
| pm2_5_mean6 | 98.9 |
| pm2_5_std6 | 98.9 |
| no2_mean6 | 98.9 |
| no2_std6 | 98.9 |
| so2_mean6 | 99.0 |
| so2_std6 | 99.0 |
| no2_mean24 | 99.3 |
| no2_std24 | 99.3 |
| nh3_mean168 | 99.3 |
| nh3_std168 | 99.3 |
| no2_mean168 | 99.5 |
| no2_std168 | 99.5 |
| pm10_mean24 | 99.5 |
| pm10_std24 | 99.5 |
| pm2_5_mean24 | 99.5 |
| pm2_5_std24 | 99.5 |
| so2_mean24 | 99.6 |
| so2_std24 | 99.6 |
| pm2_5_mean168 | 99.8 |
| pm2_5_std168 | 99.8 |
| so2_mean168 | 99.8 |
| so2_std168 | 99.8 |
| pm10_mean168 | 99.8 |
| pm10_std168 | 99.8 |
| readings | 100.0 |
| latitude | 100.0 |
| longitude | 100.0 |
| elevation_m | 100.0 |
| temp_c | 100.0 |
| humidity_pct | 100.0 |
| wind_speed_ms | 100.0 |
| pressure_hpa | 100.0 |
| precip_mm | 100.0 |
| hour_sin | 100.0 |
| hour_cos | 100.0 |
| doy_sin | 100.0 |
| doy_cos | 100.0 |
| weekday | 100.0 |
| is_weekend | 100.0 |
| is_burning_season | 100.0 |
