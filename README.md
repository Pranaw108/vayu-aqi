# Vayu — Indian air quality tracking & forecasting

A free, **non-commercial public-interest** service that shows current air quality across
India and forecasts it 24 hours ahead, built on the full CPCB monitoring archive.

Full engineering plan: [Vayu Build Handbook](https://claude.ai/artifact/Kwm17iCNts9Mop73TZq8Qd)

## Status

| Phase | Module | State |
|---|---|---|
| 1 · Data foundation | DF-01 normalise station CSVs | done |
| 1 · Data foundation | DF-02 station registry | done |
| 2 · AQI engine | AQ-01 CPCB sub-indices | done · 57 tests pass |
| 2 · AQI engine | AQ-02 cross-check vs labelled cities | done · r=0.940 |
| 1 · Data foundation | DF-03 hourly table + DuckDB query layer | done · 15 tests pass |
| 3 · Model | MD-01 feature engineering | next |
| 3 · Model | MD-02..MD-04 baselines, LightGBM, backtest | not started |
| 4 · Live + API + UI | LI, AP, FE, DP | not started |

## Layout

```
pipeline/       offline data prep and model training -- never deployed
api/            FastAPI service + model, deployed to Hugging Face Spaces
web/            React frontend, deployed to Cloudflare Pages
data/           gitignored; raw archive and curated Parquet
reports/        generated markdown reports, committed on purpose
```

## Getting started

```bash
pip install -r requirements.txt

# Point at the raw archive once, so you can skip --raw-root afterwards
export VAYU_RAW_ROOT="D:/Development/Dataset/air quality dataset/India datasets"

python pipeline/registry.py                 # ~1 second
python pipeline/normalise.py --sample 5     # quick check, ~20 seconds
python pipeline/normalise.py                # full run, ~10 minutes
python pipeline/build_hourly.py             # hourly table, ~95 seconds
python pipeline/validate_aqi.py             # AQI cross-check
pytest tests/ -q                            # 72 tests
```

### Reading the data

`api/store.py` is the only way the service touches the archive:

```python
from api.store import Store

store = Store()
store.health()                  # row counts, data age, is_live
store.stations()                # mappable stations + latest AQI (for the map)
store.station("DL01")           # one station, with current pollutant values
store.history("DL01", a, b)     # hourly series for a range
store.cities()                  # (rollup, note) — note says what was excluded
```

AQI is never computed in SQL. The store pulls the readings a station needs and
hands them to `api/aqi.py`, so the CPCB rules have one implementation.

Both scripts write a markdown report to `reports/`. Read those rather than re-running.

## Data sources

| Source | Role | Licence note |
|---|---|---|
| CPCB archive (local) | Training corpus, 2009–2024 | Government open data |
| [CPCB via data.gov.in](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations) | Primary live feed | Free API key |
| [OpenAQ](https://docs.openaq.org/) | Failover live feed | Free API key, 60/min |
| [Open-Meteo](https://open-meteo.com/en/docs/air-quality-api) | Forecast weather covariates | **Non-commercial only** |
| [WAQI](https://aqicn.org/api/) | Optional city cross-check | Attribution required; notify by email before public launch |

This project is non-commercial. Two of the four APIs above would need paid or negotiated
licences if that ever changed.

## Known data defects (handled, not hidden)

- **84 stations** share one coordinate — the geographic centre of India. Flagged
  `is_mappable = False`. Still usable for per-station time-series training.
- **106 stations** have a pollution control board acronym (HSPCB, KSPCB, …) where the city
  name should be. Flagged `has_real_city = False`; excluded from city rollups.
- **113 different header layouts** across the 546 raw files. Normalised to one schema.
- **Pre-2018 data is sparse** — some Delhi stations drop to 28% PM10 coverage. Model
  training is restricted to 2018 onward.

## Non-negotiables

1. **Never train on same-hour PM2.5/PM10 to predict AQI.** AQI is a deterministic function
   of those pollutants; such a model learns arithmetic, not air quality.
2. **Never use a random train/test split.** Rolling-origin validation only.
3. **Never publish an AQI from fewer than 3 pollutants**, one of which must be PM2.5 or
   PM10. CPCB requires it; anything else is a fabricated number on a health service.
4. **Always show data age.** Stale readings shown as current are worse than no reading.
