# Fire features per station

Generated 2026-10-10 by `pipeline/build_fire_features.py`.

- Detections used: **6,310,535**
- Stations: **43**
- Station-days: **107,263**
- Features: **14** plus 1-, 2- and 3-day lags
- Runtime: **0.3 min**

Upwind figures use ERA5 wind direction, averaged over the day as a circular
mean, and weight each fire by FRP divided by distance within a 60 degree cone.

## Monthly means

October and November should stand out at northern stations — that is paddy
stubble season, and it coincides with the winter inversion that traps the smoke.

| Month | fire_count_0_100km | fire_frp_0_100km | fire_count_100_300km | fire_frp_100_300km | fire_count_300_500km | fire_frp_300_500km | fire_count_upwind_0_100km | fire_frp_upwind_0_100km | fire_count_upwind_100_300km | fire_frp_upwind_100_300km | fire_count_upwind_300_500km | fire_frp_upwind_300_500km | fire_count_upwind | fire_frp_upwind |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jan | 9.8 | 32.2 | 40.2 | 162.1 | 55.0 | 252.4 | 3.5 | 10.5 | 11.6 | 52.7 | 21.7 | 100.7 | 36.7 | 163.9 |
| Feb | 12.3 | 42.2 | 49.8 | 227.6 | 76.5 | 377.6 | 4.9 | 17.0 | 13.7 | 60.1 | 30.2 | 137.4 | 48.8 | 214.5 |
| Mar | 6.3 | 18.8 | 73.3 | 313.2 | 95.7 | 406.2 | 2.5 | 7.8 | 16.2 | 65.5 | 27.8 | 112.4 | 46.6 | 185.7 |
| Apr | 17.7 | 130.8 | 202.6 | 943.1 | 459.0 | 2,206.1 | 9.1 | 75.1 | 44.5 | 286.3 | 76.8 | 431.9 | 130.4 | 793.3 |
| May | 33.8 | 227.4 | 358.9 | 2,027.9 | 620.2 | 3,987.7 | 16.9 | 122.6 | 143.3 | 943.5 | 242.9 | 1,750.2 | 403.1 | 2,816.3 |
| Jun | 1.6 | 6.3 | 59.0 | 283.2 | 93.8 | 475.6 | 0.6 | 2.2 | 7.8 | 40.7 | 31.6 | 158.8 | 40.0 | 201.7 |
| Jul | 0.8 | 2.2 | 3.2 | 13.4 | 28.1 | 120.4 | 0.2 | 0.5 | 0.8 | 3.1 | 5.7 | 22.9 | 6.7 | 26.5 |
| Aug | 0.7 | 1.4 | 1.7 | 4.0 | 9.7 | 43.2 | 0.2 | 0.4 | 0.7 | 1.5 | 3.9 | 16.9 | 4.8 | 18.9 |
| Sep | 1.5 | 3.8 | 5.9 | 21.2 | 31.3 | 160.3 | 0.5 | 1.3 | 2.4 | 9.4 | 14.3 | 75.1 | 17.2 | 85.9 |
| Oct | 14.2 | 56.8 | 320.7 | 1,882.7 | 424.8 | 2,520.7 | 4.9 | 18.9 | 211.4 | 1,309.7 | 273.3 | 1,704.6 | 489.6 | 3,033.1 |
| Nov | 20.3 | 72.3 | 801.2 | 5,493.3 | 735.6 | 4,787.7 | 10.1 | 36.8 | 575.4 | 4,095.8 | 458.7 | 3,150.7 | 1,044.2 | 7,283.3 |
| Dec | 23.3 | 72.4 | 55.8 | 212.2 | 76.8 | 308.7 | 8.6 | 27.8 | 18.4 | 78.4 | 32.7 | 125.5 | 59.6 | 231.7 |
