# Empirical data intake schema

Use this schema when real records become available through an authorized source. Keep the registered-deed signal, guidance reference, asking-price signal, and independent validation label in separate fields or tables. The synthetic CSVs in `dataset/` do not satisfy this schema.

## Registered-deed sample

Recommended grain: one row per residential property transfer (or one row per property line if a deed covers multiple properties), with the source's document identifier replaced by a non-identifying research key.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `transaction_id` | string | Yes | Random or salted pseudonymous key; never a party's personal identifier. |
| `registration_date` | ISO date | Yes | Date registered, with source precision preserved. |
| `sro_area` | string | Yes | Sub-Registrar Office name or code. |
| `locality_raw` | string | Yes | Non-personal locality/village/ward as supplied. |
| `micro_market_id` | string | If available | Stable project-defined area key after geocoding/review. |
| `property_type` | category | Yes | Residential site, apartment/flat, villa/row house, or documented other category. |
| `area_value` | number | If available | Recorded property area; retain the original value and unit. |
| `area_unit` | category | If available | Source unit, such as square feet or square metres. |
| `declared_consideration_inr` | integer/decimal | Yes | Consideration amount declared in the registered document, INR. |
| `guidance_value_inr` | integer/decimal | If available | Guidance/stamp-duty valuation amount used for that deed, if present in the source. |
| `guidance_rate_inr_per_sqm` | integer/decimal | If joined | Schedule rate; retain separately from total guidance value and declared consideration. |
| `source_record_key` | string | Yes | Non-identifying provenance key that supports deduplication and audit. |
| `source_name` | string | Yes | Provider/department and extract name. |
| `source_terms_reference` | string | Yes | File or record documenting permission, license, and permitted research use. |

Do not include party names, phone numbers, government ID numbers, full residential addresses, signatures, or deed-image scans in the modeling table. Store any temporary linkage material separately with restricted access, and remove it after the approved linkage process.

## Independent validation sample

Recommended grain: one independently verified value for a property and valuation date. This label must not be synthetic and must not simply copy the registered declared consideration being evaluated.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `validation_id` | string | Yes | Pseudonymous validation record key. |
| `property_link_key` | string | Yes | Approved non-identifying link to the relevant listing/deed record, where permitted. |
| `valuation_date` | ISO date | Yes | Date the independent value refers to. |
| `verified_price_inr` | integer/decimal | Yes | Independently established transaction/valuation amount. |
| `verification_method` | string/category | Yes | For example, verified completed transaction, bank valuation, or qualified appraisal. |
| `source_reference` | string | Yes | Audit reference to the permitted evidence, stored without unnecessary personal data. |
| `review_status` | category | Yes | Include only records that pass a documented verification review. |

## Linkage and quality rules

- Define and document the time window and match confidence rules before matching records. A matching locality alone is not a property-level match.
- Preserve each source's original price and area units alongside normalized INR and square-foot values.
- Keep declared consideration, total guidance value, guidance rate, listing asking price, and independent validation price distinct.
- Record coverage by year, SRO, locality, property type, and area; report unmatched and excluded rows.
- Obtain and retain written terms that allow the intended analysis and publication of aggregated results before using a source in model training or evaluation.
- Never report metrics from synthetic validation prices as empirical validation results.
