# Synthetic anchor-only baselines

**Status:** `synthetic_anchor_only_baselines`

**Source:** `dataset/synthetic_property_valuation_data.csv + dataset/synthetic_registered_deeds_validation.csv`

**Target:** generated latent reference price; not observed asking or transaction prices

## Metrics

### Hedonic Gam Ridge Anchor Proxy

- MAPE: 11.163%
- MAE: INR 2,591,726
- Hit rate within 10%: 48.20%

### Hist Gradient Boosting Anchor Proxy

- MAPE: 11.299%
- MAE: INR 2,660,775
- Hit rate within 10%: 48.48%

Only the two canonical synthetic CSV files were used. The historical real-listing baseline is disabled in this synthetic-only project.
