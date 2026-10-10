# Is the fire signal real, or seasonal?

Generated 2026-10-11 by `pipeline/diagnose_fires.py` for state `DL`.

A feature can correlate strongly with pollution and still add nothing, if what it
really encodes is the time of year. Fires and Delhi's PM2.5 both climb through
October, so the raw correlation cannot separate cause from calendar.

The last column is what matters: the share of the raw correlation that survives
once day-of-year and yesterday's PM2.5 — both of which the model already has — are
controlled for.

| Feature | Month | Rows | Raw | After season | After season + lag | Kept |
|---|---|---:|---:|---:|---:|---:|
| `fire_count_upwind` | September | 7,586 | +0.257 | +0.196 | +0.157 | 61% |
| `fire_count_upwind` | October | 8,018 | +0.481 | +0.171 | +0.075 | 16% |
| `fire_count_upwind` | November | 7,790 | +0.268 | +0.171 | +0.139 | 52% |
| `fire_count_upwind` | December | 8,276 | +0.083 | +0.127 | +0.091 | 110% |
| `fire_count_100_300km` | September | 7,586 | +0.229 | +0.148 | +0.004 | 2% |
| `fire_count_100_300km` | October | 8,018 | +0.610 | +0.266 | +0.123 | 20% |
| `fire_count_100_300km` | November | 7,790 | +0.272 | +0.148 | +0.049 | 18% |
| `fire_count_100_300km` | December | 8,276 | -0.002 | +0.020 | +0.051 | 2269% |
| `fire_frp_100_300km` | September | 7,586 | +0.185 | +0.099 | -0.031 | 17% |
| `fire_frp_100_300km` | October | 8,018 | +0.599 | +0.263 | +0.104 | 17% |
| `fire_frp_100_300km` | November | 7,790 | +0.254 | +0.129 | +0.025 | 10% |
| `fire_frp_100_300km` | December | 8,276 | +0.009 | +0.018 | +0.069 | 790% |

## What this says

October has the strongest raw correlation of any month (+0.481) and the
weakest once controlled (+0.075) — only 16% survives. Through October both fires and PM2.5 rise
together, so the raw figure is mostly the shared ramp. The model already knows it is
late October and that yesterday was bad, and the fire count adds little on top.

November keeps 52% (+0.139).
There both fires and pollution are high and fluctuating day to day, so the
correlation reflects real coupling rather than a trend.

This is why October's onset recall did not improve with fire features. The problem
is not that the fires are missing or that the wind points the wrong way — it is that
in October the fire signal is largely redundant with the calendar.
