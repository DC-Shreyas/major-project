"""Synthetic anchor-only baseline entrypoint for the BLP-Graph workflow."""

from __future__ import annotations

import json
from pathlib import Path

from .synthetic_blp_graph import _anchor_only_baselines, _load_synthetic


def run_listing_baseline(project_root: Path) -> dict:
    """Run tabular baselines on generated anchors without reading listing files."""
    frame = _load_synthetic(project_root)
    metrics = _anchor_only_baselines(frame)
    report = {
        "status": "synthetic_anchor_only_baselines",
        "data_source": "dataset/synthetic_property_valuation_data.csv + dataset/synthetic_registered_deeds_validation.csv",
        "target": "generated latent reference price; not observed asking or transaction prices",
        "metrics": metrics,
        "scope_note": "Only the two canonical synthetic CSV files were used. The historical real-listing baseline is disabled in this synthetic-only project.",
    }
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "listing_baseline_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Synthetic anchor-only baselines",
        "",
        f"**Status:** `{report['status']}`",
        "",
        f"**Source:** `{report['data_source']}`",
        "",
        f"**Target:** {report['target']}",
        "",
        "## Metrics",
        "",
    ]
    for name, result in metrics.items():
        lines.extend([
            f"### {name.replace('_', ' ').title()}",
            "",
            f"- MAPE: {result['mape_percent']:.3f}%",
            f"- MAE: INR {result['mae_inr']:,.0f}",
            f"- Hit rate within 10%: {result['hit_rate_within_10_percent']:.2f}%",
            "",
        ])
    lines.extend([report['scope_note'], ""])
    (reports_dir / "listing_baseline_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    return report
