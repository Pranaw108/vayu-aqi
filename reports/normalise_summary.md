# Station data normalisation — DF-01

Generated 2026-09-20 by `pipeline/normalise.py`.

- Stations processed: **546**
- Readings written: **71,862,029**
- Output: `D:\Development\Code\ML_projects\vayu\data\curated`
- Runtime: **9.9 min**

- Time span: **2009-01-01** to **2024-12-31**

## Rows per state

| State | Stations | Readings |
|---|---:|---:|
| Andhra_Pradesh | 14 | 1,376,111 |
| Assam | 7 | 569,783 |
| Bihar | 19 | 2,912,588 |
| Chhattisgarh | 9 | 778,551 |
| Delhi_NCT | 43 | 10,342,500 |
| Gujarat | 18 | 2,590,703 |
| Haryana | 15 | 2,452,510 |
| Jharkhand | 2 | 145,121 |
| Karnataka | 28 | 4,662,642 |
| Kerala | 23 | 2,965,918 |
| Madhya_Pradesh | 20 | 2,165,320 |
| Maharashtra | 90 | 8,356,471 |
| Odisha | 10 | 978,489 |
| Punjab | 6 | 1,297,934 |
| Rajasthan | 31 | 3,308,518 |
| Tamil_Nadu | 37 | 3,559,912 |
| Telangana | 16 | 2,468,238 |
| Unknown_Unassigned | 99 | 12,266,835 |
| Uttar_Pradesh | 40 | 6,256,048 |
| West_Bengal | 19 | 2,407,837 |

## Average completeness by measure

Percent of stored rows that carry a value. Low numbers here are the
real limit on what the model can learn.

| Measure | Mean % present | Stations with >80% |
|---|---:|---:|
| pm2_5 | 89.4 | 483 |
| pm10 | 87.7 | 477 |
| no | 90.5 | 506 |
| no2 | 92.2 | 511 |
| nox | 93.1 | 515 |
| nh3 | 80.6 | 436 |
| so2 | 87.9 | 493 |
| co | 91.1 | 487 |
| o3 | 88.4 | 487 |
| at | 72.7 | 376 |
| rh | 83.6 | 443 |
| ws | 84.7 | 450 |
| wd | 85.3 | 450 |
| vws | 7.8 | 10 |
| bp | 38.3 | 183 |
| benzene | 78.5 | 406 |
| toluene | 61.2 | 311 |
| xylene | 21.0 | 98 |
| mp_xylene | 50.9 | 206 |
| o_xylene | 1.5 | 0 |
| eth_benzene | 50.3 | 196 |

## Values removed by cleaning

| Rule | Values removed |
|---|---:|
| pm2_5 > pm10 (both dropped) | 118,325 |
| duplicate timestamps | 0 |
| unparsable timestamps | 0 |

## Columns absent from source files

These became all-null columns. Not an error -- CPCB stations monitor
different sets of pollutants.

| Column | Files without it |
|---|---:|
| o_xylene | 506 |
| vws | 468 |
| xylene | 386 |
| bp | 284 |
| mp_xylene | 163 |
| eth_benzene | 156 |
| toluene | 144 |
| at | 94 |
| nh3 | 56 |
| benzene | 55 |
| rh | 35 |
| ws | 31 |
| wd | 30 |
| so2 | 21 |
| o3 | 17 |
| pm10 | 8 |
| no | 6 |
| nox | 5 |
| co | 4 |
| pm2_5 | 2 |
| no2 | 1 |

## Stations with weak PM coverage

AQI needs PM2.5 or PM10. Stations below 50% on both are near-useless
for AQI and should be checked before they reach the model.

| Station | State | % pm2_5 | % pm10 | Rows |
|---|---|---:|---:|---:|
| TN28 | Tamil_Nadu | 48.2 | 20.5 | 62,067 |
