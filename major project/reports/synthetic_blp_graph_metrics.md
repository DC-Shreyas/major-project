# BLP-Graph simulation prototype

**Status:** `simulation_only_architecture_prototype`

> ALL model training labels, paired signals, and validation values in this run are generated synthetic fixtures; not observed Bengaluru transactions.

## Simulation design

- Training: 2023Q2, 2023Q3 (1,507 generated rows; 143 generated exact-price anchors).
- Conformal calibration: 2023Q4 (79 generated validation anchors).
- Holdout: 2024Q1 (724 generated rows).
- Graph: 613 micro-market nodes and 1,917 undirected functional similarity edges.
- Model: Native PyTorch additive spline GAM hedonic head plus multi-head attention over training-profile-only functional micro-market similarity edges and a GRU over prior-quarter asking/guidance and declared/guidance summaries. Unseen markets map to a shared OOV node. Training combines interval-censored likelihood with generated exact-price anchors; inference uses a bounded truncated-lognormal posterior mean.

## Synthetic holdout results

Metrics below compare model outputs with generated latent prices in the simulation fixture. They are not empirical Bengaluru price-prediction results.

| Signal/model | MAE (INR lakh) | MAPE | Median APE | Hit within 10% | Hit within 20% |
|---|---:|---:|---:|---:|---:|
| Asking price only | 39.74 | 16.36% | 16.32% | 22.4% | 65.5% |
| Declared value only | 28.51 | 11.91% | 10.39% | 48.5% | 71.1% |
| Naive geometric pool | 15.19 | 5.96% | 5.72% | 84.7% | 100.0% |
| Bounded latent graph | 14.96 | 5.86% | 5.32% | 85.4% | 99.9% |

## Baselines, ablations, and stability

| Variant | MAPE |
|---|---:|
| Full graph + GRU (five-seed mean prediction) | 5.86% |
| No functional graph edges | 6.00% |
| No GRU temporal module | 5.79% |
| no censored likelihood | 6.42% |
| no asking signal | 192.51% |
| no deed signal | 13.32% |
| hedonic gam ridge anchor proxy | 11.16% |
| hist gradient boosting anchor proxy | 11.30% |

Five-seed MAPE standard deviation: 0.119 percentage points. Paired market-level Wilcoxon comparison with the geometric-pool baseline: p=0.06959 across 445 synthetic micro-markets.
Leave-micro-market-out simulation: 8.54% MAPE on 146 held-out rows from 93 held-out markets.

## Synthetic 90% conformal intervals

- Prior-quarter calibration (2023Q4 to 2024Q1): 96.0% coverage (nominal 90%).
- Same-quarter calibration on flagged validation anchors: 91.8% coverage on 646 remaining synthetic rows.
- Calibration is stratified by functional-similarity cluster, not geography; the data contains no coordinates.
- Same-quarter calibration illustrates exchangeability; prior-quarter undercoverage shows that future-quarter coverage is not established.

## Scope and limitations

- Synthetic fixture values are generated and cannot establish real-market accuracy or policy indices.
- The graph has unit and training-known micro-market representations plus functional covariate-similarity edges; unseen markets map to an OOV node. No coordinates, POI nodes, RERA project nodes, metro-event edges, or true spatial-neighbor edges are available.
- Functional profiles and profile scaling use training-period covariates only; markets absent from training use a shared OOV node.
- Conformal strata are six functional market clusters built from synthetic property profiles because flagged anchors are too sparse per market; these are not geographic clusters.
- Synthetic latent truth is used only on the final quarter to score the simulation and is never used to fit the model outside flagged synthetic anchors.
- MugRep, ReGram, ST-RAP, LightGBM, and SHAP benchmarks are not run in this prototype; their packages or licensed data/implementations are not present.

The run implements the synthetic-data versions of the dual-signal censored likelihood, unit-attribute spline GAM, functional micro-market attention, prior-quarter GRU, learned asking markup and under-declaration heads, an additive hedonic/locality decomposition, and cluster-stratified split conformal intervals. It does not complete the proposal's real spatial graph or empirical validation because RERA, POI/metro, geocodes, and observed transaction records are absent.

Rebuild with `python scripts/run_blp_graph_simulation.py`.
