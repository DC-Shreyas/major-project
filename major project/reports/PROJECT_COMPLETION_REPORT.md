# BLP-Graph synthetic project completion report

**Status:** complete for the two-file synthetic-data workflow described below. This is not complete empirical validation of the research proposal.

Inputs: `dataset/synthetic_property_valuation_data.csv + dataset/synthetic_registered_deeds_validation.csv` (3,000 paired generated property/deed rows). No other dataset files are read by the primary pipeline or website.

## Model and evaluation

- Architecture: Native PyTorch additive spline GAM hedonic head plus multi-head attention over training-profile-only functional micro-market similarity edges and a GRU over prior-quarter asking/guidance and declared/guidance summaries. Unseen markets map to a shared OOV node. Training combines interval-censored likelihood with generated exact-price anchors; inference uses a bounded truncated-lognormal posterior mean.
- Chronological split: train on 2023Q2, 2023Q3; calibrate on flagged generated anchors in 2023Q4; hold out 2024Q1.
- Five-seed mean holdout MAPE: 5.86% against generated latent price.
- Holdout MAE: INR 1,496,264.
- Nominal 90% interval coverage on the generated holdout: 96.0%.
- Leave-micro-market-out proxy MAPE: 8.54%.

All values above score against generated synthetic targets. Treat them as pipeline diagnostics, not as real-world valuation accuracy or guaranteed coverage.

## Research proposal coverage

Implemented on the available synthetic pair: censored dual-signal training, an additive spline hedonic head, multi-head attention over functional market-similarity edges, a prior-quarter GRU, anchor-supervised asking-markup and under-declaration heads, cluster-stratified conformal intervals, chronological and leave-market-out simulations, tabular/signal baselines, and component ablations.

- No listing portal originals, observed registered deeds, RERA project table, property coordinates, POI data, IT park data, or dated metro milestones are present in the two canonical CSVs.
- The graph therefore uses functional similarity edges between synthetic micro-market nodes, not spatial adjacency or a full project/city/amenity graph.
- The five-seed and holdout scores measure recovery of generated latent prices only; they do not establish real-market accuracy or conformal coverage under temporal shift.
- LightGBM, MugRep, ReGram, ST-RAP, SHAP, and the proposal's formal Diebold-Mariano comparison are not implemented; available comparisons include anchor-only Ridge and histogram-gradient-boosting proxies, signal baselines, ablations, and paired Wilcoxon diagnostics.

## Interactive website

Run `python scripts/serve_web_app.py` and open `http://127.0.0.1:8765`. The app trains the graph estimator at startup and exposes the same two synthetic CSVs through its records, summary, and estimate interfaces.

## Rebuild

Run `python scripts/run_project.py` to rebuild the model experiments, predictions, graph/index outputs, and reports.
