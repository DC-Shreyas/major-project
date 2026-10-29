# Spatial linkage readiness

**Status:** `blocked_missing_property_coordinates`

## Available assets

- Canonical synthetic rows: 3,000.
- Canonical rows with coordinates: 0.
- Coordinate columns in canonical data: false.
- Metro network rows: 0.
- Bmtc route stop occurrences: 0.
- Bbmp ward polygons: 0.
- Poi or it park data available: false.
- Dated metro event data available: false.
- Property coordinates available: false.

## Proposal features

- property latitude longitude: `False`.
- h3 resolution 9 cell: `False`.
- nearest metro distance: `False`.
- nearest bmtc distance: `False`.
- poi or it park distance: `False`.
- dated metro event features: `False`.

## Remaining blockers

- The canonical synthetic files do not provide property coordinates.
- No metro, bus-stop, ward, amenity/IT-park, or dated metro-event data is used by this synthetic-only workflow.

No additional spatial files are inspected. H3 and proximity features remain unavailable until spatial data is explicitly added to the permitted project inputs.
