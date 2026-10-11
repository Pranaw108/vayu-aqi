# no_fires_h6 vs core_fires_h6

Generated 2026-10-11 by `pipeline/compare_models.py`.

Both models scored on the **same rows** with the **same code**. That is the whole
point of this script: every mis-comparison in this project came from two numbers
produced by two different pieces of code.

- Window: **2024-10-01** to **2024-12-30**
- Rows scored: **85,673** (identical for both)
- `no_fires_h6`: 134 features, 400 trees
- `core_fires_h6`: 140 features, 397 trees

## Accuracy

| Metric | no_fires_h6 | core_fires_h6 | Δ |
|---|---:|---:|---:|
| MAE (µg/m³) | 41.23 | 41.36 | +0.13 |
| Skill vs persistence | 35.41% | 35.20% | -0.21 |
| MAE, calm now | 24.67 | 24.80 | +0.12 |
| MAE, elevated now | 56.48 | 56.63 | +0.14 |

## Onset detection, free of any threshold choice

Recall and precision trade off against each other, so a single operating point
cannot say which model detects better. These two summaries do not depend on where
the threshold is set.

| Measure | no_fires_h6 | core_fires_h6 | Δ |
|---|---:|---:|---:|
| Average precision | 0.7866 | 0.7801 | -0.0065 |
| ROC AUC | 0.9155 | 0.9135 | -0.0020 |

Measured on 41,083 calm hours containing 9,021 onsets.

### The full curve

| Threshold | no_fires_h6 recall | no_fires_h6 precision | core_fires_h6 recall | core_fires_h6 precision |
|---:|---:|---:|---:|---:|
| 80 | 94.4% | 43.6% | 94.3% | 43.7% |
| 90 | 89.5% | 50.5% | 89.2% | 50.5% |
| 100 | 83.1% | 57.8% | 81.9% | 58.0% |
| 110 | 74.9% | 65.4% | 73.6% | 65.7% |
| 120 | 65.9% | 73.1% | 64.0% | 73.2% |
| 130 | 56.6% | 80.3% | 54.5% | 80.3% |
| 140 | 47.5% | 85.8% | 46.0% | 86.2% |
| 160 | 31.8% | 92.7% | 31.2% | 92.4% |

The 120 µg/m³ threshold the reports quote is a choice, not a fact. A health warning
may well prefer a lower one: more false alarms, far fewer missed episodes.

## Verdict

`core_fires_h6` does not detect measurably better or worse than `no_fires_h6`: average precision moves by -0.0065.
