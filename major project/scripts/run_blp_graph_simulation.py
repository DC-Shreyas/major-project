from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blr_valuation.synthetic_blp_graph import run_synthetic_blp_graph


if __name__ == "__main__":
    report = run_synthetic_blp_graph(PROJECT_ROOT)
    print(f"Status: {report['status']}")
    print(f"Synthetic test rows: {report['test_rows']:,}")
    print(f"Synthetic metrics: {PROJECT_ROOT / 'reports' / 'synthetic_blp_graph_metrics.md'}")
