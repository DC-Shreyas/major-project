# Dual-Signal Micro-Market Property Valuation: Synthetic Implementation

## Abstract

This software prototype implements a BLP-Graph-style latent property-price model using only the two synthetic property/deed CSVs supplied with the project. The system combines interval-censored asking and declared price signals, flagged generated price anchors, an additive spline hedonic component, attention over functionally similar synthetic micro-markets, a prior-quarter GRU, anchor-supervised asking-markup and under-declaration heads, and cluster-stratified split conformal intervals. Its chronological holdout and leave-micro-market-out metrics measure recovery of generated latent prices only. They do not establish the model's performance on observed Bengaluru transactions.

## Data

The property fixture contains 3,000 generated records and is linked one-to-one by `synthetic_property_id` to 3,000 generated deed/validation records. The supplied records cover JP Nagar, Jayanagar, and BTM Layout across 2023 Q2 to 2024 Q1. The primary pipeline reads only these two CSV files.

All prices and reference values are generated. The flagged validation rows are generated anchors, not independent real transaction evidence. The data has no coordinates, RERA projects, POIs/IT parks, or dated metro milestones.

## Methods

The chronological experiment trains on 2023 Q2 and Q3, calibrates on flagged generated anchors from 2023 Q4, and evaluates on 2024 Q1. A multi-head attention network propagates messages over top-five functional profile-similarity edges. A GRU summarizes earlier quarter market-level asking/guidance and declared/guidance ratios. The price head adds a spline-based hedonic component to a graph and temporal locality premium.

The likelihood treats each record as interval-censored between the maximum of guidance and declared values and the asking price. Flagged generated exact references supervise latent log price and auxiliary asking-markup and under-declaration heads. Split conformal quantiles are estimated using six functional profile clusters, with a global fallback when a cluster has fewer than ten anchors.

## Experiments

The primary experiment uses five random seeds. A chronological holdout and a random leave-micro-market-out experiment are reported. Available comparators include asking-only, declared-only, geometric pooling, anchor-only spline Ridge, and anchor-only histogram-gradient-boosting proxies. Ablations remove functional graph edges, the GRU, the censored likelihood, asking signal, or deed signal. A paired Wilcoxon test compares per-row absolute percentage errors against geometric pooling.

These proxies do not reproduce LightGBM, MugRep, ReGram, or ST-RAP. No SHAP analysis or Diebold-Mariano test is reported. The results file includes the estimates for each implemented experiment.

## Results

See `reports/synthetic_blp_graph_metrics.md` for generated holdout metrics, interval coverage, ablations, baseline scores, and leave-market-out results. Quote them as synthetic simulation diagnostics only. Do not describe them as sale-price accuracy, independent appraisal accuracy, or evidence of Bengaluru market performance. Future-quarter conformal coverage is not guaranteed by the exchangeability result for split conformal prediction.

## Limitations and next data requirements

The synthetic fixture does not support the proposal's complete heterogeneous spatial graph or empirical validation. A full empirical study requires the original portal listing and registered deed records, matched property-level locations, RERA project records, POI/IT park data, dated metro milestones, and independent valuation/transaction references. With those inputs, market nodes, spatial-distance edges, temporal events, genuinely spatial calibration strata, and real-market evaluation can replace the current synthetic proxies.

## Interactive application

Run `python scripts/serve_web_app.py` and open `http://127.0.0.1:8765`. The site exposes a synthetic-data overview, record search, and an interactive bounded BLP-Graph estimate. Estimates, comparables, and intervals are demonstrations and not appraisals.
