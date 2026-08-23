# Data provenance and licensing boundary

The computation uses only twelve frozen public-use files listed in `data/expected_inputs.csv`:

- four NHANES 2011–2012/2013–2014 DEMO and BMX XPT files for eligibility, demographics, BMI, survey weights, strata, and PSU;
- seven compressed minute-level step-series files and one Troiano wear-indicator file from the PhysioNet v1.0.1 study snapshot.

The raw files are not redistributed. The PhysioNet snapshot stored with the private study source includes a CC0 1.0 notice; NHANES public-use terms remain governed by the official source. A person obtaining the data is responsible for reviewing the current official terms and preserving source notices.

Each run hashes all twelve files before reading them. A mismatch stops the pipeline. The analysis then builds the eligible cohort in memory and writes only aggregate summaries or direction-specific fitted-model resources. These include exact isotonic thresholds and direction-level metadata without participant identifiers or participant-level predictions. Reference results are bundled solely for terminal numeric comparison and cannot substitute for the raw inputs.

Provenance manifests retain relative historical lineage and hashes. They do not redistribute historical code, raw data, Word files, or private repository state.
