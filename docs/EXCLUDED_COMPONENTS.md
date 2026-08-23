# Explicitly excluded components

The following material is outside the current code release and must not be copied into it.

## Data and privacy exclusions

- Raw NHANES or PhysioNet files.
- Participant identifiers, person-day or minute-level tables.
- Fold assignments, participant-level OOF predictions, and bootstrap replicate records.
- Any derived table that permits participant-level reconstruction.

## Superseded scientific branches

- Metabolic-outcome regressions and medication, diabetes, or glycaemia branches.
- ROC, Youden, AUC, spline-risk, and outcome-prediction analyses.
- Historical Tier 1/2/3, 20/20/2, conversion-permission, or combined E/H grading.
- The retired post hoc cluster-count/PAM branch, its tables, and its supplemental figure; it is not part of the current manuscript or release pipeline.
- Any statement treating a step algorithm as ground truth or a crosswalk as health-threshold equivalence.

## Historical engineering and writing artifacts

- Old manuscripts, supplementary Word files, reviewer reports, submission-package builders, and Word-version migration scripts.
- Internal absolute paths, local Git-state locks, and checks tied to obsolete manuscript versions.
- Old tier-based figures and the historical four-panel Figure 4.
- Cached files, environments, logs, temporary files, and the full dirty private repository.

## Provenance-only exceptions

Historical script and aggregate-result paths may appear as relative text entries in provenance manifests so the release can be audited. That does not make those files part of the release. The current three-panel Figure 4 is copied as a small aggregate visual reference because it is the manuscript authority that future plotting code must match.
