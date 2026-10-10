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

## Datasets

**No data is redistributed in this repository.** `data/` is gitignored — the raw archive is
6.5 GB and the curated Parquet 2.9 GB. Get the data from the sources below and point
`VAYU_RAW_ROOT` at it.

### What the project reads

| File(s) | What it is | Rows |
|---|---|---|
| `<state>/<CODE>.csv` × 546 | Station readings at 15-minute resolution, 2009–2024. CPCB CAAQMS format: PM2.5, PM10, NO, NO₂, NOx, NH₃, SO₂, CO, O₃, plus weather (temp, humidity, wind, pressure) and BTX volatiles | 71,862,029 |
| `stations.csv` | Station registry: code, city, state, PIN, latitude, longitude, elevation, and three derived terrain features | 553 |
| `IndiaPopularCity.csv` | Daily city-level pollutants **with a published AQI and category** — used only to cross-check the AQI engine | 25,555 |

### Where this data comes from

**Station readings — Central Pollution Control Board (CPCB), India.** The national
Continuous Ambient Air Quality Monitoring Station network. Official access points:

- [CPCB CCR portal](https://app.cpcbccr.com/ccr/) — advance search, downloadable per station
  and pollutant. Note the portal limits a 15-minute request to roughly one week per station,
  so assembling a multi-year, all-station archive means many requests.
- [Real time AQI on data.gov.in](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations) — free API key, current readings only, no history
- [OpenAQ](https://openaq.org/) — mirrors Indian CPCB data with a documented API

> **Provenance gap, needs Pranaw to confirm.** The local archive arrived without any source,
> licence or README file, and its 2009 start predates the CCR portal's usual coverage. It is
> therefore very likely a compiled redistribution rather than a direct CCR export. If you know
> where it was downloaded from, record it here — a dataset nobody can trace is a dataset nobody
> else can reuse.
>
> `stations.csv` also carries `elevation_m`, `topo_complexity`, `coastal_proximity` and
> `valley_factor`, which are **not** CPCB fields. Someone derived them. How, is unrecorded.

**City-level AQI file.** The columns, date range and category labels match
[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
by Rohan Rao on Kaggle (its `city_day.csv`). The copy here is a subset: 22 cities and 25,555 rows
against the original's 26 cities and 29,531 rows, and it ends 2020-04-10 rather than July 2020.
Check the licence on the dataset page before reusing it.

### Known problems in the source data

Measured, not assumed — see `reports/` for the full numbers.

| Problem | Scale | How this project handles it |
|---|---|---|
| Inconsistent headers | **113** different header layouts across 546 files; column counts 4–22 | Normalised to one 22-column schema |
| Placeholder coordinates | **84** stations share `22.6876, 79.3704` — the geographic centre of India | Flagged `is_mappable = False`; kept for time-series training, excluded from the map |
| Agency names instead of cities | ~**106** stations named HSPCB, KSPCB, UPPCB… | Flagged `has_real_city = False`; excluded from city rollups |
| Impossible PM pairs | **118,325** readings where PM2.5 > PM10 | Both values dropped — one sensor is wrong and you cannot tell which |
| Sparse early years | Some Delhi stations fall to 28% PM10 coverage pre-2018 | Model training restricted to 2018 onward |
| Stations stop at different dates | 61 of 546 are more than a day behind the archive's newest hour; one is 8 years behind | `store.cities()` only ranks stations inside a 24-hour comparability window |
| Published AQI above the scale | **538** rows in the city file exceed the CPCB maximum of 500 (highest: 2049) | Excluded from the cross-check |
| Wrong CO unit | Ahmedabad's median CO is 17.12 against ~0.89 everywhere else | That city excluded from the cross-check |


**Two open items that are not yet fixed:**

- **KA06 "Bengaluru East City Corporation" looks faulty** and currently ranks first nationally
  in `store.cities()`. Its lifetime mean PM2.5 is 118 and its December mean 215 — far above what
  Bengaluru actually experiences — and its lifetime maximum is exactly 1000.0, which looks like a
  sensor ceiling rather than a reading. Not excluded yet: one suspect station is not proof, and
  silently dropping a station is worse than showing it. Needs a station-level plausibility check
  against regional peers **before any ranking is published**.
- **The terrain features in `stations.csv` are unexplained.** Until someone can say how they were
  derived, treat them as untrusted inputs to any model.

### Live sources (used at runtime, not for training)

| Source | Role | Licence note |
|---|---|---|
| [CPCB via data.gov.in](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations) | Primary live feed | Free API key |
| [OpenAQ](https://docs.openaq.org/) | Failover live feed | Free API key, 60/min |
| [Open-Meteo](https://open-meteo.com/en/docs/air-quality-api) | Forecast weather covariates | **Non-commercial only** |
| [WAQI](https://aqicn.org/api/) | Optional city cross-check | Attribution required; notify by email before public launch |

This project is non-commercial. Two of the four APIs above would need paid or negotiated
licences if that ever changed.

### Citing this

If the station archive helps you, cite CPCB as the originating authority. For the city file,
cite Rohan Rao's Kaggle dataset as well.

## Non-negotiables

1. **Never train on same-hour PM2.5/PM10 to predict AQI.** AQI is a deterministic function
   of those pollutants; such a model learns arithmetic, not air quality.
2. **Never use a random train/test split.** Rolling-origin validation only.
3. **Never publish an AQI from fewer than 3 pollutants**, one of which must be PM2.5 or
   PM10. CPCB requires it; anything else is a fabricated number on a health service.
4. **Always show data age.** Stale readings shown as current are worse than no reading.
