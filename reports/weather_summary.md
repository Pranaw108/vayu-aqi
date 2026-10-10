# Historical weather — MD-01 input

Generated 2026-10-10 by `pipeline/fetch_weather.py`.

- Window: **2023-01-01** to **2023-03-31**
- Stations covered: **6**
- Grid cells fetched: **3** (of 3 needed)
- Hourly rows: **6,480**
- API calls: **3** (free tier allows 10,000/day)
- Runtime: **0.1 min**

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
| boundary_layer_m | 100.0 |
| wind_speed_ms | 100.0 |

## Is the station-to-grid match correct?

CPCB stations measure their own temperature and humidity. ERA5 is an independent
source, so a strong correlation means each station is matched to the right grid
cell. A weak one would mean the coordinates are wrong — a mistake that is
otherwise invisible.

- Not enough overlapping station weather to check.
