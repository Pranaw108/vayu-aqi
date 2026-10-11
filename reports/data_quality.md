# Data quality

Generated 2026-10-11 by `pipeline/data_quality.py`.

## Pipeline stages

| Stage | Present | Build with |
|---|:--:|---|
| registry | yes | `pipeline/registry.py` |
| curated 15-min | yes | `pipeline/normalise.py` |
| hourly | yes | `pipeline/build_hourly.py` |
| weather | yes | `pipeline/fetch_weather.py` |
| sensor quarantine | yes | `pipeline/fetch_weather.py` |
| fires | yes | `pipeline/fetch_fires.py` |
| fire features | yes | `pipeline/build_fire_features.py` |

## Outlier stations

A station whose long-run mean PM2.5 sits 3.5 or more robust
deviations from its own state's median. Computed with the median and median
absolute deviation, so one extreme station cannot drag the baseline it is
measured against.

This is the check that was missing when KA06 was found by hand, after it had
already ranked worst in India in a city table.

| Station | State | City | Mean PM2.5 | State median | Score |
|---|---|---|---:|---:|---:|
| AP02 | AP | Kallepalle | 104.0 | 36.5 | +9.29 |
| KA06 | KA | Bengaluru East City Corporation | 118.1 | 30.8 | +8.86 |
| MH19 | MH | Mira-Bhayandar | 78.7 | 43.3 | +5.24 |
| RJ02 | RJ | Bhiwadi | 100.7 | 52.4 | +5.17 |
| KA05 | KA | Davangere | 78.7 | 30.8 | +4.86 |
| TN32 | TN | Housing Unit | 80.6 | 31.5 | +4.66 |
| AP06 | AP | Vegeswarapuram | 67.6 | 36.5 | +4.28 |
| CG01 | CG | Pawaras | 79.0 | 29.1 | +4.26 |
| KL01 | KL | Thrissur | 106.9 | 37.6 | +4.21 |
| RJ08 | RJ | Bharatpur | 88.3 | 52.4 | +3.84 |
| UP15 | UP | Ghaziabad | 116.0 | 48.5 | +3.81 |
| MH32 | MH | Mumbai | 68.9 | 43.3 | +3.79 |
| UN82 | UN | PCBA | 143.3 | 49.2 | +3.76 |
| UP04 | UP | Moradabad | 112.6 | 48.5 | +3.62 |
| MH31 | MH | Mumbai | 67.4 | 43.3 | +3.57 |
| UP05 | UP | Lucknow | 111.1 | 48.5 | +3.53 |
| BR12 | BR | Ratanpur | 37.2 | 80.2 | -3.58 |
| BR08 | BR | Nagara | 22.8 | 80.2 | -4.78 |
| BR01 | BR | Manpur | 19.8 | 80.2 | -5.03 |
| OR03 | OR | Kapileswarpur | 19.0 | 53.6 | -5.30 |

## Feature table health

- Table: `features\target=pm2_5\horizon=6\features.parquet`
- Rows: **2,128,871** · columns: **139** · features: **134**

A constant column teaches nothing and still takes a slot in every tree's
feature sample, which is how 56 fire columns cost 13 points of onset recall.

- No constant, near-constant or mostly-missing features.
