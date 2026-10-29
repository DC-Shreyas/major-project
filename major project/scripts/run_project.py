from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from blr_valuation.project_runner import run_full_project

if __name__ == "__main__":
    report = run_full_project(PROJECT_ROOT)
    result = report["model_evaluation"]
    print(f"Project status: {report['project_status']}")
    print(f"Canonical paired records: {report['records']:,}")
    print(f"Generated-reference holdout MAPE: {result['holdout_mape_percent']:.2f}%")
    print(f"Completion report: {PROJECT_ROOT / 'reports' / 'PROJECT_COMPLETION_REPORT.md'}")
    print(f"Website: {report['website']['local_url']} (run scripts/serve_web_app.py)")
