from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blr_valuation.data_foundation import run_pipeline


if __name__ == "__main__":
    report = run_pipeline(PROJECT_ROOT)
    output = PROJECT_ROOT / "data" / "processed" / "property_records.csv"
    print(f"Prepared {report['records']:,} paired generated records from the two canonical synthetic CSV files.")
    print(f"Prepared records: {output}")
