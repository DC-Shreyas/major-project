# BLP-Graph project plan and implementation status

## Objective

Build the proposal's dual-signal property valuation workflow using only the two synthetic CSV files currently available. The architecture is aligned with the proposal, while missing real spatial and administrative sources are recorded as unimplemented inputs.

## Canonical data and boundaries

- `dataset/synthetic_property_valuation_data.csv`: 3,000 generated property rows with property attributes, guidance references, asking and declared values, generated latent price, market label, and quarter.
- `dataset/synthetic_registered_deeds_validation.csv`: 3,000 linked generated deed/validation rows with flagged generated reference prices.
- Join: one-to-one by `synthetic_property_id`.
- Not used: all other files in `dataset/`, including older listing exports, guidance documents, and spatial layers.
- Not present in the two canonical files: coordinates, RERA project records, POIs, IT parks, dated metro milestones, or observed transaction prices.

## Workflow implemented

1. Validate and join the same two synthetic CSVs; preserve generated values as synthetic.
2. Build additive spline features for area, bedrooms, age, and floor, plus property-type and SRO categorical terms.
3. Build synthetic micro-market profiles and top-five functional-similarity edges. Multi-head attention propagates information over those edges.
4. Use a GRU to encode prior-quarter asking/guidance, declared/guidance, and record-count summaries. Chronological scaling is fitted on training-quarter histories.
5. Predict latent log price as an additive hedonic component plus graph/temporal locality premium.
6. Optimize a Gaussian interval-censored likelihood between guidance/declaration and asking bounds, anchored by flagged generated exact references. Auxiliary heads estimate asking markup and under-declaration probability from the flagged generated anchors.
7. Calibrate nominal 90% intervals on 2023 Q4 flagged generated anchors by functional market cluster. Inference bounds remain within the input guidance/declaration and asking values.
8. Evaluate on chronological 2024 Q1 and through a leave-micro-market-out split. Run five seeds, signal baselines, anchor-only Ridge and histogram-gradient-boosting proxies, paired Wilcoxon diagnostics, and ablations removing the graph, GRU, censoring, asking signal, and deed signal.
9. Serve the graph model and synthetic records through the interactive local website.

## Items the supplied data cannot support

- Actual hierarchical unit/project/H3/city/amenity graph nodes and spatial-distance edges.
- RERA supply pipelines, POI/IT park accessibility, or dated metro event sequences.
- Independently observed registered prices or real asking listings.
- The proposal's LightGBM, MugRep, ReGram, ST-RAP, SHAP and Diebold-Mariano benchmarks.
- A claim of real-market conformal coverage. The chronological split is distribution shifted and all reference targets are generated.

Functional similarity is a synthetic proxy, not a substitute for geographical proximity. Cluster-stratified calibration is similarly functional, not geographic. The leave-market-out evaluation remains transductive with respect to market covariate profiles because no geographic/project attributes are available.

## Run commands

- `python scripts/build_data_foundation.py` — join and prepare the two files without fitting a model.
- `python scripts/run_project.py` — full synthetic BLP-Graph experiments and reports.
- `python scripts/serve_web_app.py` — local website at `http://127.0.0.1:8765`.
