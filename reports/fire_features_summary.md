# Fire features per station

Generated 2026-10-10 by `pipeline/build_fire_features.py`.

- Detections used: **1,678,383**
- Stations: **43**
- Station-days: **28,583**
- Features: **14** plus 1-, 2- and 3-day lags
- Runtime: **0.1 min**

Upwind figures use ERA5 wind direction, averaged over the day as a circular
mean, and weight each fire by FRP divided by distance within a 60 degree cone.

## Monthly means

October and November should stand out at northern stations — that is paddy
stubble season, and it coincides with the winter inversion that traps the smoke.

| Month | fire_count_0_100km | fire_frp_0_100km | fire_count_100_300km | fire_frp_100_300km | fire_count_300_500km | fire_frp_300_500km | fire_count_upwind_0_100km | fire_frp_upwind_0_100km | fire_count_upwind_100_300km | fire_frp_upwind_100_300km | fire_count_upwind_300_500km | fire_frp_upwind_300_500km | fire_count_upwind | fire_frp_upwind |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jan | 15.6 | 49.3 | 46.9 | 173.7 | 72.7 | 367.5 | 5.1 | 14.4 | 10.2 | 38.5 | 27.8 | 152.7 | 43.1 | 205.5 |
| Feb | 10.7 | 36.2 | 37.4 | 185.1 | 56.5 | 303.9 | 4.8 | 17.5 | 11.4 | 62.2 | 24.9 | 126.9 | 41.1 | 206.6 |
| Mar | 7.5 | 22.3 | 61.5 | 274.9 | 82.6 | 387.4 | 3.4 | 10.5 | 12.9 | 51.7 | 23.8 | 99.7 | 40.1 | 161.9 |
| Apr | 14.1 | 121.5 | 84.9 | 460.6 | 269.1 | 1,384.4 | 5.8 | 61.5 | 25.7 | 173.8 | 55.2 | 291.5 | 86.7 | 526.8 |
| May | 50.4 | 351.0 | 600.0 | 3,184.7 | 862.9 | 5,202.3 | 32.6 | 241.5 | 255.5 | 1,675.5 | 369.0 | 2,616.1 | 657.1 | 4,533.0 |
| Jun | 1.9 | 7.7 | 60.4 | 260.5 | 82.5 | 432.3 | 0.7 | 2.7 | 12.4 | 55.1 | 35.4 | 170.7 | 48.6 | 228.6 |
| Jul | 0.6 | 1.2 | 2.6 | 11.3 | 20.0 | 74.2 | 0.1 | 0.1 | 0.8 | 2.8 | 3.2 | 10.3 | 4.1 | 13.2 |
| Aug | 0.8 | 1.6 | 1.1 | 2.1 | 6.2 | 29.2 | 0.3 | 0.7 | 0.2 | 0.3 | 2.1 | 9.2 | 2.6 | 10.2 |
| Sep | 1.4 | 3.6 | 6.1 | 20.5 | 22.5 | 106.7 | 0.4 | 0.6 | 1.2 | 3.2 | 7.3 | 32.1 | 8.9 | 35.9 |
| Oct | 18.7 | 89.0 | 414.7 | 2,478.2 | 423.0 | 2,585.5 | 6.0 | 26.4 | 267.9 | 1,670.1 | 272.1 | 1,749.4 | 545.9 | 3,446.0 |
| Nov | 16.9 | 63.3 | 1,056.8 | 6,996.0 | 809.8 | 5,156.1 | 6.5 | 25.7 | 613.7 | 4,070.7 | 485.1 | 3,130.9 | 1,105.4 | 7,227.3 |
| Dec | 27.3 | 85.5 | 49.6 | 183.8 | 65.3 | 252.9 | 8.9 | 27.9 | 17.1 | 58.8 | 28.6 | 101.2 | 54.6 | 187.9 |
