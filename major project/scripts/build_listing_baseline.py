from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blr_valuation.listing_baseline import run_listing_baseline


if __name__ == "__main__":
    report = run_listing_baseline(PROJECT_ROOT)
    report_path = PROJECT_ROOT / "reports" / "listing_baseline_metrics.json"
    print(f"Synthetic anchor-only baseline metrics: {report_path}")
