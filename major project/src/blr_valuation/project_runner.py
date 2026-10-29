"""Run the synthetic-only BLP-Graph workflow and write a completion report."""

from __future__ import annotations

import json
from pathlib import Path

from .synthetic_blp_graph import run_synthetic_blp_graph, _load_synthetic


def run_full_project(project_root: Path) -> dict:
    frame = _load_synthetic(project_root)
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "property_records.csv", index=False, encoding="utf-8-sig")
    graph = run_synthetic_blp_graph(project_root)
    holdout = graph["holdout_metrics_against_synthetic_latent_truth"]["bounded_latent_graph"]
    interval = graph["conformal_90_metrics_against_synthetic_latent_truth"]
    report = {
        "project_status": "synthetic_blp_graph_workflow_complete",
        "data_source": "dataset/synthetic_property_valuation_data.csv + dataset/synthetic_registered_deeds_validation.csv",
        "records": int(len(frame)),
        "scope": sorted(frame["sro_area"].astype(str).unique().tolist()),
        "quarters": sorted(frame["synthetic_quarter"].astype(str).unique().tolist()),
        "model": graph["model_description"],
        "model_evaluation": {
            "training_quarters": graph["training_quarters"],
            "training_rows": graph["training_rows"],
            "calibration_quarter": graph["conformal_calibration_quarter"],
            "calibration_anchor_rows": graph["synthetic_validation_anchors_for_conformal"],
            "holdout_quarter": graph["evaluation_quarter"],
            "holdout_rows": graph["test_rows"],
            "holdout_mape_percent": holdout["mape_percent"],
            "holdout_mae_inr": holdout["mae_inr"],
            "interval_nominal_coverage_percent": interval["nominal_coverage_percent"],
            "interval_observed_coverage_percent": interval["empirical_coverage_percent"],
        },
        "experiments": {
            "random_seeds": graph["five_seed_summary"]["seeds"],
            "ablation_variants": sorted(graph["ablation_metrics"]),
            "anchor_only_baselines": sorted(graph["anchor_only_baseline_metrics"]),
            "leave_micro_market_out_mape_percent": graph["leave_micro_market_out_evaluation"]["metrics_against_synthetic_latent_truth"]["mape_percent"],
        },
        "website": {
            "entry_point": "scripts/serve_web_app.py",
            "local_url": "http://127.0.0.1:8765",
            "features": [
                "filtered overview and quarterly signals",
                "searchable synthetic records",
                "interactive graph model estimates and intervals",
                "asking markup and under-declaration diagnostics",
                "generated feature comparables",
            ],
        },
        "data_status": graph["data_status"],
        "proposal_coverage_gaps": [
            "No listing portal originals, observed registered deeds, RERA project table, property coordinates, POI data, IT park data, or dated metro milestones are present in the two canonical CSVs.",
            "The graph therefore uses functional similarity edges between synthetic micro-market nodes, not spatial adjacency or a full project/city/amenity graph.",
            "The five-seed and holdout scores measure recovery of generated latent prices only; they do not establish real-market accuracy or conformal coverage under temporal shift.",
            "LightGBM, MugRep, ReGram, ST-RAP, SHAP, and the proposal's formal Diebold-Mariano comparison are not implemented; available comparisons include anchor-only Ridge and histogram-gradient-boosting proxies, signal baselines, ablations, and paired Wilcoxon diagnostics.",
        ],
        "outputs": [
            "data/processed/property_records.csv",
            "data/processed/synthetic_blp_graph_holdout_predictions.csv",
            "data/processed/synthetic_functional_market_graph.csv",
            "data/processed/synthetic_market_quarter_indices.csv",
            "reports/synthetic_blp_graph_metrics.json",
            "reports/synthetic_blp_graph_metrics.md",
            "reports/project_completion_status.json",
            "reports/PROJECT_COMPLETION_REPORT.md",
        ],
    }
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "project_completion_status.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# BLP-Graph synthetic project completion report",
        "",
        "**Status:** complete for the two-file synthetic-data workflow described below. This is not complete empirical validation of the research proposal.",
        "",
        f"Inputs: `{report['data_source']}` ({report['records']:,} paired generated property/deed rows). No other dataset files are read by the primary pipeline or website.",
        "",
        "## Model and evaluation",
        "",
        f"- Architecture: {report['model']}",
        f"- Chronological split: train on {', '.join(report['model_evaluation']['training_quarters'])}; calibrate on flagged generated anchors in {report['model_evaluation']['calibration_quarter']}; hold out {report['model_evaluation']['holdout_quarter']}.",
        f"- Five-seed mean holdout MAPE: {report['model_evaluation']['holdout_mape_percent']:.2f}% against generated latent price.",
        f"- Holdout MAE: INR {report['model_evaluation']['holdout_mae_inr']:,.0f}.",
        f"- Nominal 90% interval coverage on the generated holdout: {report['model_evaluation']['interval_observed_coverage_percent']:.1f}%.",
        f"- Leave-micro-market-out proxy MAPE: {report['experiments']['leave_micro_market_out_mape_percent']:.2f}%.",
        "",
        "All values above score against generated synthetic targets. Treat them as pipeline diagnostics, not as real-world valuation accuracy or guaranteed coverage.",
        "",
        "## Research proposal coverage",
        "",
        "Implemented on the available synthetic pair: censored dual-signal training, an additive spline hedonic head, multi-head attention over functional market-similarity edges, a prior-quarter GRU, anchor-supervised asking-markup and under-declaration heads, cluster-stratified conformal intervals, chronological and leave-market-out simulations, tabular/signal baselines, and component ablations.",
        "",
        *[f"- {item}" for item in report["proposal_coverage_gaps"]],
        "",
        "## Interactive website",
        "",
        "Run `python scripts/serve_web_app.py` and open `http://127.0.0.1:8765`. The app trains the graph estimator at startup and exposes the same two synthetic CSVs through its records, summary, and estimate interfaces.",
        "",
        "## Rebuild",
        "",
        "Run `python scripts/run_project.py` to rebuild the model experiments, predictions, graph/index outputs, and reports.",
        "",
    ]
    (reports / "PROJECT_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
