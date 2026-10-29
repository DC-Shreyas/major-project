"""Prepare the canonical synthetic property/deed pair for project workflows."""

from __future__ import annotations

import json
from pathlib import Path

from .synthetic_blp_graph import _load_synthetic


def run_pipeline(project_root: Path) -> dict:
    """Join and write only the two canonical synthetic CSV files."""
    frame = _load_synthetic(project_root)
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed_dir / "property_records.csv", index=False, encoding="utf-8-sig")
    report = {
        "project_status": "synthetic_pair_prepared",
        "data_source": "dataset/synthetic_property_valuation_data.csv + dataset/synthetic_registered_deeds_validation.csv",
        "records": int(len(frame)),
        "micro_markets": int(frame["micro_market_id"].nunique()),
        "quarters": sorted(frame["synthetic_quarter"].astype(str).unique().tolist()),
        "scope_note": "Only the two canonical synthetic CSV files were read. This workflow does not inspect original listings, deeds, guidance PDFs, or spatial files.",
    }
    (reports_dir / "data_readiness.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports_dir / "data_readiness.md").write_text(
        "# Synthetic data foundation\n\n"
        f"**Status:** `{report['project_status']}`\n\n"
        f"**Source:** `{report['data_source']}`\n\n"
        f"**Rows:** {report['records']:,} paired synthetic property/deed records.\n\n"
        f"**Micro-markets:** {report['micro_markets']:,}.\n\n"
        f"**Quarters:** {', '.join(report['quarters'])}.\n\n"
        f"{report['scope_note']}\n",
        encoding="utf-8",
    )
    return report
