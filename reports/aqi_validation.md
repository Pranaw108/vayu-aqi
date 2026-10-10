# AQI engine cross-check — AQ-02

Generated 2026-09-20 by `pipeline/validate_aqi.py`.

> **What this proves.** Exact correctness of the CPCB formula is proved by
> `tests/test_aqi.py`, where every expected value is hand-computed from the
> official breakpoint table. This report is a *consistency* check against
> published city AQI values, which are computed from per-station AQIs rather
> than from the city-average pollutants in the same file. The two are not the
> same calculation, so they track each other closely but do not match exactly.

## Result

| Check | Target | Actual | Pass |
|---|---:|---:|:--:|
| Correlation with published AQI | ≥0.90 | 0.940 | ✅ |
| Within ±50 AQI points | ≥85% | 89.4% | ✅ |
| Same CPCB category | ≥65% | 71.6% | ✅ |
| Within ±25 AQI points | — | 70.9% | — |

- Rows in file: **25,555**
- Rows with a published AQI: **21,304**
- Rows comparable after exclusions: **19,873**

## Defects found in the reference file

Worth knowing before anyone trusts this dataset's AQI column.

**1. 538 rows have a published AQI above 500** (highest: 2049). The CPCB index
stops at 500, so whoever produced that column extrapolated past the Severe band
instead of capping. Those rows are excluded here.

| City | Rows above 500 |
|---|---:|
| Ahmedabad | 412 |
| Delhi | 43 |
| Patna | 25 |
| Gurugram | 22 |
| Lucknow | 15 |
| Jorapokhar | 7 |
| Amritsar | 4 |
| Hyderabad | 4 |

**2. CO looks like the wrong unit for: Ahmedabad.** CPCB reports CO in mg/m³, where a
city median sits near 1. These cities are far above that, which inflates their
CO sub-index and therefore their AQI. Excluded from the comparison.

| City | Median CO |
|---|---:|
| Ahmedabad | 17.12 |

## Rows where we refused to compute an AQI

The file publishes an AQI for these, but they fail CPCB's own validity rules.
Refusing is correct behaviour.

| Reason | Rows |
|---|---:|
| fewer_than_3_pollutants | 54 |
| no_pm2_5_or_pm10 | 4 |

## Agreement by AQI band

| Published band | Rows | Within ±50 | Mean difference |
|---|---:|---:|---:|
| Good | 912 | 96.3% | +3.0 |
| Satisfactory | 6,536 | 97.6% | -4.9 |
| Moderate | 7,338 | 88.1% | -8.3 |
| Poor | 2,433 | 71.0% | -6.6 |
| Very Poor | 2,091 | 89.5% | -4.5 |
| Severe | 563 | 80.3% | -16.4 |

## Agreement by dominant pollutant

| Dominant pollutant | Rows | Within ±50 |
|---|---:|---:|
| co | 2,129 | 86.7% |
| nh3 | 12 | 100.0% |
| no2 | 248 | 92.3% |
| o3 | 487 | 80.9% |
| pm10 | 7,510 | 93.9% |
| pm2_5 | 9,445 | 86.8% |
| so2 | 42 | 100.0% |

## Category distribution

| Category | Published | Ours |
|---|---:|---:|
| Good | 912 | 1,877 |
| Satisfactory | 6,536 | 6,650 |
| Moderate | 7,338 | 6,474 |
| Poor | 2,433 | 1,849 |
| Very Poor | 2,091 | 2,496 |
| Severe | 563 | 527 |
