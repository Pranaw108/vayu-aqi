# Features — pm2_5, +24h (MD-01)

Generated 2026-10-10 by `pipeline/features.py`.

- Target: **pm2_5**, **24** hours ahead
- Training window starts: **2018-01-01**
- Stations: **43**
- Rows with a known answer: **2,128,147**
- Features: **190**
- Rows with forecast weather: **100.0%**
- Build time: **3.5 min**

## Persistence baseline

The number any model has to beat. Predicting *the value right now* is a
surprisingly strong forecast for air quality, and a model that cannot beat it
has learned nothing.

- Mean absolute error: **39.3 µg/m³**
- Median absolute error: **21.0 µg/m³**
- RMSE: **68.1 µg/m³**

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
| at_lag24 | 68.9 |
| at_lag1 | 69.0 |
| bp_lag24 | 72.9 |
| bp_lag1 | 73.0 |
| ws_lag24 | 86.1 |
| ws_lag1 | 86.1 |
| nh3_lag168 | 87.0 |
| nh3_lag48 | 87.1 |
| nh3_lag24 | 87.2 |
| nh3_lag12 | 87.2 |
| nh3_lag6 | 87.2 |
| nh3_lag3 | 87.2 |
| nh3_lag2 | 87.2 |
| nh3_lag1 | 87.2 |
| wd_sin_lag24 | 88.2 |
| wd_cos_lag24 | 88.2 |
| wd_sin_lag1 | 88.2 |
| wd_cos_lag1 | 88.2 |
| nh3_mean6 | 88.3 |
| nh3_std6 | 88.3 |
| nh3_mean24 | 89.0 |
| nh3_std24 | 89.0 |
| rh_lag24 | 89.0 |
| rh_lag1 | 89.0 |
| nh3_mean168 | 89.7 |
| nh3_std168 | 89.7 |
| so2_lag168 | 89.7 |
| so2_lag48 | 89.9 |
| so2_lag24 | 90.0 |
| so2_lag12 | 90.0 |
| so2_lag6 | 90.0 |
| so2_lag3 | 90.0 |
| so2_lag2 | 90.0 |
| so2_lag1 | 90.0 |
| so2_mean6 | 91.3 |
| so2_std6 | 91.3 |
| so2_mean24 | 91.8 |
| so2_std24 | 91.8 |
| fc_boundary_layer_m | 92.0 |
| wx_boundary_layer_m | 92.0 |
| so2_mean168 | 92.1 |
| so2_std168 | 92.1 |
| pm10_lag168 | 93.4 |
| pm10_lag48 | 93.7 |
| pm10_lag24 | 93.8 |
| pm10_lag12 | 93.8 |
| pm10_lag6 | 93.8 |
| pm10_lag3 | 93.8 |
| pm10_lag2 | 93.9 |
| pm10_lag1 | 93.9 |
| co_lag168 | 95.0 |
| o3_lag168 | 95.0 |
| pm10_mean6 | 95.1 |
| pm10_std6 | 95.1 |
| co_lag48 | 95.2 |
| co_lag24 | 95.2 |
| o3_lag48 | 95.2 |
| co_lag12 | 95.3 |
| co_lag6 | 95.3 |
| co_lag3 | 95.3 |
| o3_lag24 | 95.3 |
| co_lag2 | 95.3 |
| co_lag1 | 95.3 |
| o3_lag12 | 95.3 |
| o3_lag6 | 95.3 |
| o3_lag3 | 95.3 |
| o3_lag2 | 95.3 |
| o3_lag1 | 95.3 |
| pm10_mean24 | 95.8 |
| pm10_std24 | 95.8 |
| pm10_mean168 | 96.2 |
| pm10_std168 | 96.2 |
| no2_lag168 | 96.5 |
| no2_lag48 | 96.7 |
| no2_lag24 | 96.7 |
| no2_lag12 | 96.8 |
| no2_lag6 | 96.8 |
| no2_lag3 | 96.8 |
| no2_lag2 | 96.8 |
| no2_lag1 | 96.8 |
| o3_mean6 | 96.8 |
| o3_std6 | 96.8 |
| co_mean6 | 97.0 |
| co_std6 | 97.0 |
| pm2_5_lag168 | 97.1 |
| o3_mean24 | 97.5 |
| o3_std24 | 97.5 |
| pm2_5_lag48 | 97.5 |
| pm2_5_lag24 | 97.6 |
| pm2_5_lag12 | 97.7 |
| co_mean24 | 97.7 |
| co_std24 | 97.7 |
| pm2_5_lag6 | 97.8 |
| pm2_5_lag3 | 97.8 |
| no2_mean6 | 97.8 |
| no2_std6 | 97.8 |
| pm2_5_lag2 | 97.8 |
| pm2_5_lag1 | 97.9 |
| o3_mean168 | 97.9 |
| o3_std168 | 97.9 |
| co_mean168 | 98.2 |
| co_std168 | 98.2 |
| no2_mean24 | 98.4 |
| no2_std24 | 98.4 |
| no2_mean168 | 98.9 |
| no2_std168 | 98.9 |
| pm2_5_mean6 | 99.1 |
| pm2_5_std6 | 99.1 |
| neighbour_pm2_5_lag1 | 99.3 |
| pm2_5_mean24 | 99.6 |
| pm2_5_std24 | 99.6 |
| pm2_5_mean168 | 99.8 |
| pm2_5_std168 | 99.8 |
| readings | 100.0 |
| latitude | 100.0 |
| longitude | 100.0 |
| elevation_m | 100.0 |
| fc_temp_c | 100.0 |
| fc_humidity_pct | 100.0 |
| fc_wind_speed_ms | 100.0 |
| fc_pressure_hpa | 100.0 |
| fc_precip_mm | 100.0 |
| wx_temp_c | 100.0 |
| wx_humidity_pct | 100.0 |
| wx_wind_speed_ms | 100.0 |
| wx_pressure_hpa | 100.0 |
| wx_precip_mm | 100.0 |
| hour_sin | 100.0 |
| hour_cos | 100.0 |
| doy_sin | 100.0 |
| doy_cos | 100.0 |
| weekday | 100.0 |
| is_weekend | 100.0 |
| is_burning_season | 100.0 |
| fire_count_0_100km | 100.0 |
| fire_frp_0_100km | 100.0 |
| fire_count_100_300km | 100.0 |
| fire_frp_100_300km | 100.0 |
| fire_count_300_500km | 100.0 |
| fire_frp_300_500km | 100.0 |
| fire_count_upwind_0_100km | 100.0 |
| fire_frp_upwind_0_100km | 100.0 |
| fire_count_upwind_100_300km | 100.0 |
| fire_frp_upwind_100_300km | 100.0 |
| fire_count_upwind_300_500km | 100.0 |
| fire_frp_upwind_300_500km | 100.0 |
| fire_count_upwind | 100.0 |
| fire_frp_upwind | 100.0 |
| fire_count_0_100km_d1 | 100.0 |
| fire_count_0_100km_d2 | 100.0 |
| fire_count_0_100km_d3 | 100.0 |
| fire_frp_0_100km_d1 | 100.0 |
| fire_frp_0_100km_d2 | 100.0 |
| fire_frp_0_100km_d3 | 100.0 |
| fire_count_100_300km_d1 | 100.0 |
| fire_count_100_300km_d2 | 100.0 |
| fire_count_100_300km_d3 | 100.0 |
| fire_frp_100_300km_d1 | 100.0 |
| fire_frp_100_300km_d2 | 100.0 |
| fire_frp_100_300km_d3 | 100.0 |
| fire_count_300_500km_d1 | 100.0 |
| fire_count_300_500km_d2 | 100.0 |
| fire_count_300_500km_d3 | 100.0 |
| fire_frp_300_500km_d1 | 100.0 |
| fire_frp_300_500km_d2 | 100.0 |
| fire_frp_300_500km_d3 | 100.0 |
| fire_count_upwind_0_100km_d1 | 100.0 |
| fire_count_upwind_0_100km_d2 | 100.0 |
| fire_count_upwind_0_100km_d3 | 100.0 |
| fire_frp_upwind_0_100km_d1 | 100.0 |
| fire_frp_upwind_0_100km_d2 | 100.0 |
| fire_frp_upwind_0_100km_d3 | 100.0 |
| fire_count_upwind_100_300km_d1 | 100.0 |
| fire_count_upwind_100_300km_d2 | 100.0 |
| fire_count_upwind_100_300km_d3 | 100.0 |
| fire_frp_upwind_100_300km_d1 | 100.0 |
| fire_frp_upwind_100_300km_d2 | 100.0 |
| fire_frp_upwind_100_300km_d3 | 100.0 |
| fire_count_upwind_300_500km_d1 | 100.0 |
| fire_count_upwind_300_500km_d2 | 100.0 |
| fire_count_upwind_300_500km_d3 | 100.0 |
| fire_frp_upwind_300_500km_d1 | 100.0 |
| fire_frp_upwind_300_500km_d2 | 100.0 |
| fire_frp_upwind_300_500km_d3 | 100.0 |
| fire_count_upwind_d1 | 100.0 |
| fire_count_upwind_d2 | 100.0 |
| fire_count_upwind_d3 | 100.0 |
| fire_frp_upwind_d1 | 100.0 |
| fire_frp_upwind_d2 | 100.0 |
| fire_frp_upwind_d3 | 100.0 |
