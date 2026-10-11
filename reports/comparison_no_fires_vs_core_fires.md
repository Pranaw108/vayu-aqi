# no_fires vs core_fires

Generated 2026-10-11 by `pipeline/compare_models.py`.

Both models scored on the **same rows** with the **same code**. That is the whole
point of this script: every mis-comparison in this project came from two numbers
produced by two different pieces of code.

- Window: **2024-10-01** to **2024-12-30**
- Rows scored: **85,317** (identical for both)
- `no_fires`: 134 features, 157 trees
- `core_fires`: 140 features, 241 trees

## Accuracy

| Metric | no_fires | core_fires | Δ |
|---|---:|---:|---:|
| MAE (µg/m³) | 48.59 | 49.08 | +0.50 |
| Skill vs persistence | 13.99% | 13.11% | -0.88 |
| MAE, calm now | 26.79 | 26.64 | -0.14 |
| MAE, elevated now | 68.69 | 69.77 | +1.09 |

## Onset detection, free of any threshold choice

Recall and precision trade off against each other, so a single operating point
cannot say which model detects better. These two summaries do not depend on where
the threshold is set.

| Measure | no_fires | core_fires | Δ |
|---|---:|---:|---:|
| Average precision | 0.4935 | 0.4992 | +0.0057 |
| ROC AUC | 0.8321 | 0.8339 | +0.0018 |

Measured on 40,928 calm hours containing 7,154 onsets.

### The full curve

| Threshold | no_fires recall | no_fires precision | core_fires recall | core_fires precision |
|---:|---:|---:|---:|---:|
| 80 | 87.4% | 33.1% | 86.8% | 33.9% |
| 90 | 77.7% | 37.2% | 76.6% | 38.2% |
| 100 | 66.4% | 41.9% | 64.5% | 42.9% |
| 110 | 53.2% | 46.5% | 51.1% | 48.0% |
| 120 | 40.3% | 52.6% | 38.0% | 53.7% |
| 130 | 27.6% | 58.5% | 26.5% | 60.2% |
| 140 | 17.4% | 64.4% | 17.0% | 65.3% |
| 160 | 5.5% | 72.5% | 5.7% | 75.6% |

The 120 µg/m³ threshold the reports quote is a choice, not a fact. A health warning
may well prefer a lower one: more false alarms, far fewer missed episodes.

## Verdict

`core_fires` does not detect measurably better or worse than `no_fires`: average precision moves by +0.0057.
