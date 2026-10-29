"""Synthetic-only intake status; never reads separate empirical candidates."""

from __future__ import annotations

import json
from pathlib import Path

from .synthetic_blp_graph import _load_synthetic


def run_empirical_intake(
    project_root: Path,
    deeds_path: Path | None = None,
    validation_path: Path | None = None,
) -> dict:
    """Report that only the canonical synthetic fixture is in scope.

    Optional empirical paths are rejected so this helper cannot silently read
    files outside the user's requested two-file data scope.
    """
    if deeds_path is not None or validation_path is not None:
        raise ValueError(
            "This project is scoped to its two canonical synthetic CSV files; "
            "additional deed or validation paths are not read."
        )
    frame = _load_synthetic(project_root)
    report = {
        "status": "synthetic_data_only_no_empirical_intake",
        "ready_for_linkage_review": False,
        "canonical_synthetic_records": int(len(frame)),
        "registered_deeds": {"status": "generated_fixture_only", "accepted_as_empirical": False},
        "independent_validation": {"status": "generated_fixture_only", "accepted_as_empirical": False},
        "scope_note": "Only the canonical synthetic pair was inspected. No separate deed or independent-validation files were read; generated references are not empirical observations.",
    }
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "empirical_data_intake.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports_dir / "empirical_data_intake.md").write_text(
        "# Empirical data intake\n\n"
        f"**Status:** `{report['status']}`\n\n"
        f"Canonical generated records: {report['canonical_synthetic_records']:,}.\n\n"
        f"{report['scope_note']}\n",
        encoding="utf-8",
    )
    return report
