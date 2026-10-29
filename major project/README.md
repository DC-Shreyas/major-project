# BLP-Graph Synthetic Property Valuation

This project implements a synthetic-data version of the BLP-Graph methodology in the research proposal. The standard pipeline and website read only these two files:

- `dataset/synthetic_property_valuation_data.csv`
- `dataset/synthetic_registered_deeds_validation.csv`

They are joined by `synthetic_property_id`. No other project dataset is read by `scripts/run_project.py` or the website.

## Run the complete workflow

Install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Build experiments, predictions, and reports:

```powershell
python scripts/run_project.py
```

Start the interactive local website:

```powershell
python scripts/serve_web_app.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The website includes a filtered overview, searchable synthetic records, and a property estimator. The estimator uses the graph model and returns the bounded point estimate, nominal interval, learned asking markup, learned under-declaration probability, locality component, and generated feature comparables.

## Implemented workflow

1. Join and validate the paired synthetic property and generated deed/validation tables.
2. Represent properties with additive spline bases and categorical property/SRO terms; represent micro-markets with synthetic feature profiles and functional-similarity edges.
3. Encode market relationships with multi-head attention and prior-quarter asking/declaration signals with a GRU.
4. Estimate latent log price as an additive hedonic component plus graph/temporal locality component.
5. Train with interval-censored likelihood between the guidance/declaration lower bound and asking upper bound. Flagged generated reference prices anchor the latent price and supervise asking-markup and under-declaration heads.
6. Calibrate nominal 90% intervals on flagged generated anchors, stratified by functional-similarity cluster.
7. Evaluate chronologically on 2024 Q1, run a leave-micro-market-out simulation, five random seeds, signal and anchor-only model baselines, and component ablations.

Full metrics and experiment details are written to `reports/synthetic_blp_graph_metrics.md` and `.json`. Prediction rows, functional graph edges, and market-quarter diagnostics are written under `data/processed/`.

## Proposal coverage limits

This is the closest end-to-end implementation possible with the two supplied synthetic CSVs. Those files have no geocoordinates, RERA project records, POI/IT park data, or dated metro milestones. The available graph therefore connects synthetic micro-markets by functional covariate similarity; it is not a spatial H3 or project/amenity graph. The proposal's LightGBM, MugRep, ReGram, ST-RAP, SHAP, and Diebold-Mariano experiments are not included. Available comparisons are explicitly labeled proxies.

The generated latent prices and flagged references are not observed sale prices or independent valuations. MAPE, MAE, hit rates, and interval coverage measure recovery of generated values only. The results are not real-world Bengaluru valuation accuracy, and future-quarter conformal coverage is not established.

## Useful files

- `scripts/run_project.py` — full experiment and report runner.
- `scripts/serve_web_app.py` — local interactive website server.
- `scripts/build_data_foundation.py` — prepares only the joined canonical synthetic pair.
- `scripts/build_listing_baseline.py` — runs synthetic-anchor tabular baselines; it does not read listing files.
- `scripts/build_spatial_readiness.py` — reports which spatial features are unavailable from the canonical pair; it does not read map files.
- `scripts/validate_empirical_data.py` — confirms the synthetic-only data scope; it does not inspect external deed or validation files.
- `scripts/run_blp_graph_simulation.py` — direct entry point for the BLP-Graph experiment runner.
- `reports/PROJECT_COMPLETION_REPORT.md` — concise project outcome and coverage limits.
- `docs/PROJECT_PLAN.md` — methodology and implementation map.
- `docs/RESEARCH_REPORT_DRAFT.md` — research write-up template with synthetic-data caveats.
