"""Report spatial readiness using only the canonical synthetic CSV pair."""

from __future__ import annotations

import json
from pathlib import Path


def build_spatial_readiness(project_root: Path) -> dict:
    from .synthetic_blp_graph import _load_synthetic

    records = _load_synthetic(project_root)
    coordinate_columns = {"latitude", "longitude"}.issubset(records.columns)
    property_coordinate_rows = (
        int(records[["latitude", "longitude"]].notna().all(axis=1).sum())
        if coordinate_columns else 0
    )

    spatial_features = {
        "property_latitude_longitude": False,
        "h3_resolution_9_cell": False,
        "nearest_metro_distance": False,
        "nearest_bmtc_distance": False,
        "poi_or_it_park_distance": False,
        "dated_metro_event_features": False,
    }
    assets = {
        "canonical_synthetic_rows": int(len(records)),
        "canonical_rows_with_coordinates": property_coordinate_rows,
        "coordinate_columns_in_canonical_data": coordinate_columns,
        "metro_network_rows": 0,
        "bmtc_route_stop_occurrences": 0,
        "bbmp_ward_polygons": 0,
        "poi_or_it_park_data_available": False,
        "dated_metro_event_data_available": False,
    }
    assets["property_coordinates_available"] = property_coordinate_rows > 0
    ready = assets["property_coordinates_available"]
    report = {
        "status": "property_geocoding_inputs_available" if ready else "blocked_missing_property_coordinates",
        "property_geospatial_linkage_ready": ready,
        "study_area": ["JP Nagar", "Jayanagar", "BTM Layout"],
        "assets": assets,
        "derived_features": spatial_features,
        "blockers": [
            "The canonical synthetic files do not provide property coordinates.",
            "No metro, bus-stop, ward, amenity/IT-park, or dated metro-event data is used by this synthetic-only workflow.",
        ] if not ready else [],
        "data_source": "The two canonical synthetic CSV files only.",
        "policy": "No additional spatial files are inspected. H3 and proximity features remain unavailable until spatial data is explicitly added to the permitted project inputs.",
    }
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "spatial_readiness.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Spatial linkage readiness",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "## Available assets",
        "",
    ]
    for key, value in assets.items():
        if isinstance(value, bool):
            display_value = str(value).lower()
        elif isinstance(value, int):
            display_value = f"{value:,}"
        else:
            display_value = str(value)
        lines.append(f"- {key.replace('_', ' ').capitalize()}: {display_value}.")
    lines.extend(["", "## Proposal features", ""])
    lines.extend(f"- {name.replace('_', ' ')}: `{available}`." for name, available in spatial_features.items())
    lines.extend(["", "## Remaining blockers", ""])
    lines.extend(f"- {item}" for item in report["blockers"])
    lines.extend(["", report["policy"], ""])
    (reports / "spatial_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    return report
