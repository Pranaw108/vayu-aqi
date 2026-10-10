# Station registry — DF-02

Generated 2026-09-20 by `pipeline/registry.py`.

- Stations in registry: **545**
- Usable on a map: **462**
- With a real city name: **439**
- Output: `D:\Development\Code\ML_projects\vayu\data\registry.parquet`

## Why stations were excluded from the map

| Reason | Stations |
|---|---:|
| Coordinates shared with many other stations (placeholder) | 82 |
| Coordinates missing | 1 |
| Coordinates outside India | 0 |

### Placeholder coordinate points

| Latitude | Longitude | Stations | States |
|---:|---:|---:|---|
| 22.6876 | 79.3704 | 76 | UN |
| 19.0436 | 73.0185 | 3 | MH |
| 30.7229 | 76.7686 | 3 | UN |

## Stations named after an agency, not a city

City rollups must skip these until someone maps them by hand.

| Name in `city` | Stations |
|---|---:|
| HSPCB | 16 |
| KSPCB | 12 |
| BSPCB | 10 |
| RSPCB | 10 |
| MPCB | 10 |
| UPPCB | 9 |
| TNPCB | 7 |
| MPPCB | 5 |
| OSPCB | 5 |
| IMD | 4 |
| CECB | 4 |
| APPCB | 3 |
| WBPCB | 3 |
| PPCB | 2 |
| PUNE | 1 |
| JMC | 1 |
| GPCB | 1 |
| PPCC | 1 |
| NTPC | 1 |
| PCBA | 1 |

## Files with no metadata row

These have readings but no location, so they cannot be registered.

- `UN001`

## Mappable stations per state

| State code | Mappable | Total |
|---|---:|---:|
| AP | 14 | 14 |
| AS | 7 | 7 |
| BR | 18 | 19 |
| CG | 9 | 9 |
| DL | 43 | 43 |
| GJ | 18 | 18 |
| HR | 15 | 15 |
| JH | 2 | 2 |
| KA | 28 | 28 |
| KL | 23 | 23 |
| MH | 87 | 90 |
| MP | 20 | 20 |
| OR | 10 | 10 |
| PB | 6 | 6 |
| RJ | 31 | 31 |
| TG | 16 | 16 |
| TN | 37 | 37 |
| UN | 19 | 98 |
| UP | 40 | 40 |
| WB | 19 | 19 |
