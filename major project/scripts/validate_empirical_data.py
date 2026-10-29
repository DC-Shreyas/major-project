from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blr_valuation.empirical_intake import run_empirical_intake


if __name__ == "__main__":
    report = run_empirical_intake(PROJECT_ROOT)
    print(f"Synthetic source check passed: {report['canonical_synthetic_records']:,} paired property/deed records.")
    print(f"Empirical status: {report['status']} (no other files read).")
