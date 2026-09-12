# Zenodo software deposit plan

## Decision

After the creator list is supplied, make a metadata-only `v1.1.2` follow-up release and archive it as an open-access Zenodo **Software** record. Use its version DOI in the manuscript and `CITATION.cff`. This keeps the public `v1.1.1` code tag immutable while allowing the required creator metadata and reserved DOI to be added together. The GitHub repository remains the development location; Zenodo provides the immutable preservation record and DOI.

## Metadata fixed for the deposit

- Upload type: Software
- Title: Direction-specific wrist-step algorithm crosswalk reproducibility package
- Version: 1.1.2 (metadata-only follow-up to the validated `v1.1.1` code release)
- Publication date: date on which the creator-complete follow-up is published
- Access: Open access
- License: MIT License
- Language: English
- Keywords: accelerometry; step count; isotonic regression; reproducibility; NHANES; wrist-worn accelerometry
- Related URL: the immutable GitHub `v1.1.2` release URL created after creator completion
- Description: Code, locked environment specifications, exact direction-specific isotonic resources, aggregate reference results, automated tests, and validation evidence for the NHANES 2011–2014 wrist-step crosswalk study. Participant-level source data and manuscript Word files are not redistributed.
- Files: `step-crosswalk-reproducibility-v1.1.2.zip` and `step-crosswalk-reproducibility-v1.1.2.zip.sha256`
- Archive SHA-256: copy the digest from the final sibling `.sha256` asset after the archive is built

## Sole unresolved required field

Creator names and identifiers are intentionally deferred to the authors. Zenodo requires at least one creator, so the record must not be published and a DOI must not be inserted into the manuscript until the final creator list has been supplied and checked. Once supplied, reserve the DOI in the Zenodo draft, add the creators and DOI to `CITATION.cff` and the English and Chinese Code Availability statements, create the metadata-only `v1.1.2` release, recheck all hashes and links, upload its two archive assets, and then publish the Zenodo record.
