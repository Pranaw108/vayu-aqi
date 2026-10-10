# Historical weather — MD-01 input

Generated 2026-10-10 by `pipeline/fetch_weather.py`.

- Window: **2018-01-01** to **2024-12-31**
- Stations covered: **462**
- Grid cells fetched: **245** (of 245 needed)
- Hourly rows: **15,035,160**
- API calls: **0** (free tier allows 10,000/day)
- Runtime: **0.2 min**

Source: [Open-Meteo](https://open-meteo.com/) ERA5 reanalysis, no API key, non-commercial use.

## Coverage per variable

| Variable | % present |
|---|---:|
| temp_c | 100.0 |
| humidity_pct | 100.0 |
| wind_speed_kmh | 100.0 |
| wind_direction_deg | 100.0 |
| pressure_hpa | 100.0 |
| precip_mm | 100.0 |
| boundary_layer_m | 92.9 |
| wind_speed_ms | 100.0 |

## Is the station-to-grid match correct?

CPCB stations measure their own temperature and humidity. ERA5 is an independent
source, so a strong correlation means each station is matched to the right grid
cell. A weak one would mean the coordinates are wrong — a mistake that is
otherwise invisible.

- Stations checked: **35**
- Median temperature correlation: **0.775**
- Median humidity correlation: **0.670**
- Above the 0.7 threshold: **22 / 35**

| Station | Grid cell | Hours | Temp r | Humidity r | Mean temp diff (°C) |
|---|---|---:|---:|---:|---:|
| BR08 | 25.50_87.00 | 17,206 | 0.095 | 0.18 | 0.49 |
| KL07 | 8.50_77.00 | 17,325 | 0.115 | 0.328 | 0.16 |
| KL20 | 8.25_77.00 | 17,302 | 0.16 | 0.114 | 0.25 |
| TG05 | 17.50_78.50 | 17,252 | 0.201 | 0.634 | 0.53 |
| UP09 | 25.00_81.00 | 17,244 | 0.413 | 0.526 | 6.09 |
| AP08 | 16.00_79.50 | 17,316 | 0.445 | 0.243 | -3.23 |
| OR02 | 21.75_86.75 | 17,303 | 0.633 | 0.36 | -0.2 |
| MP07 | 22.75_76.00 | 17,339 | 0.638 | 0.774 | 0.89 |
| MP06 | 23.00_77.25 | 17,273 | 0.66 | 0.756 | 0.49 |
| WB12 | 22.50_88.25 | 17,267 | 0.675 | 0.319 | 1.64 |
| UP14 | 28.75_77.25 | 17,362 | 0.676 | 0.508 | 1.38 |
| WB09 | 22.50_88.25 | 17,346 | 0.677 | 0.329 | 1.19 |
| DL38 | 28.50_77.25 | 17,188 | 0.697 | 0.594 | 2.87 |
| MP10 | 23.25_77.50 | 17,465 | 0.707 | 0.633 | 1.72 |
| MP09 | 24.75_81.75 | 17,402 | 0.738 | 0.681 | 2.35 |
| UP22 | 27.25_78.00 | 17,267 | 0.762 | 0.661 | 1.41 |
| BR09 | 25.50_85.00 | 17,313 | 0.77 | 0.684 | 0.44 |
| UP36 | 29.00_78.75 | 17,237 | 0.775 | 0.652 | 1.8 |
| UP33 | 25.50_78.50 | 17,296 | 0.775 | 0.757 | 1.96 |
| UP25 | 26.75_83.50 | 17,329 | 0.777 | 0.702 | 3.18 |
| MP08 | 26.25_78.25 | 17,210 | 0.779 | 0.723 | 0.1 |
| UP35 | 28.75_78.75 | 17,192 | 0.78 | 0.66 | 1.43 |
| DL20 | 28.75_77.00 | 17,240 | 0.784 | 0.629 | 1.01 |
| UP24 | 25.25_83.00 | 17,287 | 0.788 | 0.716 | 1.66 |
| UP16 | 28.50_77.50 | 17,355 | 0.791 | 0.686 | 3.03 |
| UP10 | 26.75_81.00 | 17,264 | 0.797 | 0.718 | -1.42 |
| BR06 | 25.50_85.00 | 17,483 | 0.8 | 0.691 | -0.43 |
| UP23 | 25.25_83.00 | 17,473 | 0.802 | 0.708 | 0.82 |
| DL13 | 28.75_77.25 | 17,203 | 0.802 | 0.674 | 3.27 |
| DL23 | 28.50_77.25 | 17,210 | 0.803 | 0.673 | 0.62 |
| DL22 | 28.50_77.25 | 17,265 | 0.804 | 0.689 | 0.54 |
| DL16 | 28.50_77.00 | 17,330 | 0.805 | 0.67 | 2.17 |
| UP27 | 27.00_78.25 | 17,217 | 0.807 | 0.735 | 1.13 |
| DL15 | 28.50_77.25 | 17,295 | 0.824 | 0.723 | 0.52 |
| DL29 | 28.75_77.00 | 17,316 | 0.829 | 0.667 | 1.03 |
