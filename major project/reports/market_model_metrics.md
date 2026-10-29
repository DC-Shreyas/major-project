# Synthetic property model report

**Status:** synthetic_dataset_model_ready

This workflow uses **3,000 paired generated property/deed records** as the project’s canonical dataset. All prices and labels are synthetic. Results describe performance on generated records only and are not evidence of real Bengaluru market accuracy.

## Data and split

- Training: 2023Q2, 2023Q3 (1,507 rows).
- Interval calibration: 2023Q4 (79 flagged generated reference prices).
- Holdout: 2024Q1 (724 rows).
- Holdout coverage: 91.7% for nominal 90% intervals; mean width INR 3,491,576.

## Holdout results against generated reference prices

| Method | MAE (INR) | MAPE | R² |
|---|---:|---:|---:|
| Random forest with bounded price signals | 987,502 | 3.62% | 0.967 |
| Geometric mean of asking and declared signals | 1,519,211 | 5.96% | 0.982 |

## Interpretation

Training targets and interval calibration references are generated values. Estimates are constrained by the guidance reference, declared amount, and asking amount. Read this as a reproducible software demonstration on this dataset, not as a bank, surveyor, or market valuation.
