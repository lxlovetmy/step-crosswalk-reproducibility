# Zenodo software deposit plan

## Decision

Make a metadata-only `v1.1.2` follow-up release and archive it as an open-access Zenodo **Software** record. Use its version DOI in the manuscript and `CITATION.cff`. This keeps the public `v1.1.1` code tag immutable while adding checked creator metadata and the DOI without changing the validated scientific resources. The GitHub repository remains the development location; Zenodo provides the immutable preservation record and DOI.

## Metadata fixed for the deposit

- Upload type: Software
- Title: Direction-specific wrist-step algorithm crosswalk reproducibility package
- Version: 1.1.2 (metadata-only follow-up to the validated `v1.1.1` code release)
- Publication date: 2026-09-12
- Access: Open access
- License: MIT License
- Language: English
- Keywords: accelerometry; step count; isotonic regression; reproducibility; NHANES; wrist-worn accelerometry
- Related URL: the immutable GitHub `v1.1.2` release URL created after creator completion
- Description: Code, locked environment specifications, exact direction-specific isotonic resources, aggregate reference results, automated tests, and validation evidence for the NHANES 2011–2014 wrist-step crosswalk study. Participant-level source data and manuscript Word files are not redistributed.
- Files: `step-crosswalk-reproducibility-v1.1.2.zip` and `step-crosswalk-reproducibility-v1.1.2.zip.sha256`
- Archive SHA-256: copy the digest from the final sibling `.sha256` asset after the archive is built

## Creators

1. Xiang Luo — School of Sport Science, Beijing Sport University — ORCID `0009-0006-2113-2397`
2. Qinlong Li — China Institute of Sport Science — ORCID `0000-0001-5989-8968`
3. Qiang Dong — School of Sport Science, Beijing Sport University — ORCID `0009-0003-9462-3304`
4. Yue Zhou — School of Sport Science, Beijing Sport University — ORCID `0000-0003-2603-7234`

## Reserved DOI

- Version DOI: `10.5281/zenodo.22723129`
- Reservation date: 2026-09-12
- Status: reserved in the creator-complete Zenodo draft; registration remains pending publication

## Remaining release actions

Complete the Zenodo metadata form, create and verify the matching GitHub `v1.1.2` release and archive assets, upload those two files to the Zenodo draft, and publish only after a final cross-check. Then verify that the DOI resolves and that the Zenodo citation, `CITATION.cff`, GitHub release, and manuscript Code Availability statement agree.
