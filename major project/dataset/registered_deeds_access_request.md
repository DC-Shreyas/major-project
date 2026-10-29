# Bengaluru registered-deed data access request

## Current status

No registered-deed transaction file has been added to this project. The three local PDFs are guidance-value schedules, not sale-deed records.

I could not find a free, licensed, downloadable bulk extract of Bengaluru residential sale consideration values. Karnataka's KAVERI 2.0 service covers registration, Encumbrance Certificates, certified copies, and property valuation, but the public material I found does not provide a bulk research file/API. The individual record workflow generally requires property identifiers and authentication.

## Ready-to-send inquiry

**To:** aigrcomp@karnataka.gov.in  
**Subject:** Request for anonymized Bengaluru property-registration data for academic research

Dear Sir/Madam,

I am preparing an academic research project on residential property valuation and micro-markets in Bengaluru. I would like to ask whether the Department of Stamps and Registration can provide an existing machine-readable, anonymized extract or sample of residential sale-registration records for Bengaluru Urban for research use.

The useful fields would be:

- registration date or quarter;
- Sub-Registrar Office and non-personal locality/village/ward or project identifier;
- property type and recorded area/area unit, where available;
- declared consideration amount;
- guidance/market value used for registration or stamp-duty assessment, where available;
- document type and a non-identifying record key for deduplication.

Please exclude party names, phone numbers, personal identifiers, full residential addresses, and deed-image scans. If an extract is available, please also share its data dictionary, coverage period, known limitations, access fees, and the terms for academic analysis and publication of aggregated results.

If the department does not provide a bulk extract, please advise the appropriate procedure for obtaining a small anonymized research sample or the relevant office/contact for a formal information request. I am requesting existing records or exports; I am not asking the department to create new analyses.

Thank you,

[Your name]
[Institution / course, if applicable]
[Contact email]

## Sources and constraints

- Official KAVERI 2.0 overview: https://negd.gov.in/isl/Directory/statedata/416
- Karnataka KAVERI portal: https://kaveri.karnataka.gov.in/
- Department contact above is published on the KAVERI 2.0 overview page; verify it before sending.
- A third-party API advertises Karnataka deed records but requires an account/API key and paid calls. Its terms prohibit using its data to train, fine-tune, or evaluate AI models, so it is not suitable for this project's model dataset without a separate written license: https://indiapropertyregister.com/api/docs and https://indiapropertyregister.com/terms
- KAVERI document/EC lookup is property-specific; the current listings file has locality and society text but no survey number/document number. Do not treat guidance values as registered transactions.

## After a response arrives

1. Save the department's data, data dictionary, and reuse terms in this folder.
2. Remove or hash any accidental personal identifiers before analysis.
3. Keep declared consideration, guidance value, and independently verified transaction price as separate fields.
4. Check whether the data's location and period overlap the listing sample before attempting record linkage.
