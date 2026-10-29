from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blr_valuation.spatial_readiness import build_spatial_readiness


if __name__ == "__main__":
    report = build_spatial_readiness(PROJECT_ROOT)
    print(f"Spatial status: {report['status']}")
    print(f"Canonical rows with coordinates: {report['assets']['canonical_rows_with_coordinates']:,}")
    print(f"Report: {PROJECT_ROOT / 'reports' / 'spatial_readiness.md'}")
