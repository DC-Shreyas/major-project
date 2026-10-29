# Dataset inventory and use boundary

## The only files used by the primary model and website

| File | Rows | Use |
|---|---:|---|
| `synthetic_property_valuation_data.csv` | 3,000 | Generated property attributes, asking price, declared value, guidance reference, generated latent reference, and quarter. |
| `synthetic_registered_deeds_validation.csv` | 3,000 | Matching generated deed and flagged validation rows linked by property ID. |

The two files are synthetic and form the complete allowed input set for `scripts/run_project.py`, `scripts/build_data_foundation.py`, and the website. All labels, price signals, and flagged references are generated.

## Other repository files

Other files remain in this folder from earlier work. The current primary pipeline and website do not open, merge, or use them. In particular, no older property listing export, PDF schedule, or geospatial layer is included in the graph model or website.

## Coverage and limits

- Generated SRO labels: JP Nagar, Jayanagar, and BTM Layout.
- Generated period: 2023 Q2 through 2024 Q1.
- No coordinates, RERA, POIs/IT parks, dated metro milestones, or observed sale prices are in the canonical pair.
- Synthetic micro-market similarity edges are not real spatial edges.
- Model metrics are generated-price recovery diagnostics, not real-market accuracy.
