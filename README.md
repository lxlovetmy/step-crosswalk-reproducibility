# Step-algorithm crosswalk reproducibility package

Version 1.1.1 release candidate for the crosswalk-only manuscript. The package rebuilds all reported aggregate analyses, three main-table data layers, S1–S8 machine-readable resources, and Figures 1–4/S1–S4 from the twelve frozen public-use inputs. It also exports the exact fitted isotonic thresholds, direction metadata, and a strict converter. It does not rebuild Word files.

## Scope and interpretation

The analysis covers 8,646 NHANES 2011–2014 adults (57,080 valid person-days), seven wrist-step algorithms, a 21-row sample-marginal alignment lookup, and 42 separately fitted direction-specific isotonic crosswalks. Exact fitted thresholds are the sole formal machine-readable conversion parameters; the 42-row direction metadata file supplies labels, source P05–P95 use ranges, knot counts, E/H evidence, and integrity fields. E is participant-grouped out-of-fold target-output reproduction error; H is a support-sensitive subgroup-spread diagnostic. Neither is accuracy against true steps, a clinical threshold, or an acceptability grade.

Historical metabolic, ROC/AUC, Tier, 20/20/2, and manuscript-migration branches are deliberately excluded.

## Quick start

1. Create the Conda base environment, then install and verify the exact R-package versions:

       conda env create -f environment/environment.yml
       conda activate step-crosswalk-release
       Rscript environment/install_r_packages.R
       Rscript environment/validate_runtime.R

   Alternatively, run `bash environment/create_and_validate.sh` from any directory. The R packages are locked separately in `environment/r-requirements.lock` because the requested R 4.6.0 and those historical package versions are not available together as a solvable macOS arm64 conda-forge specification. The installer retrieves the exact released R-package versions from CRAN; source-package compilation may require the standard macOS build tools.

2. Obtain the twelve public-use files and place them under a data root exactly as shown in `data/README.md`. Raw data are not included.

3. Run the complete pipeline from this directory:

       python run_all.py --mode full --data-root /path/to/data/raw --output-dir /path/to/empty/output

The runner first verifies all twelve SHA-256 values. Reference tables are read only by the final validator, after computation is complete. A smoke run uses two bootstrap repetitions and stops after the core/boundary modules:

       python run_all.py --mode smoke --data-root /path/to/data/raw --output-dir /path/to/empty/smoke-output

The complete run is computationally intensive because it repeats participant-grouped cross-fitting inside 300 stratified PSU-within-stratum cluster bootstrap samples. The primary mapping remains unweighted; this is not a fully weighted design-based population estimator.

## Outputs

- `output/tables/`: full aggregate analysis tables.
- `output/publication_tables/`: main Tables 1–3 and S1–S8 table-ready files, including the exact-threshold and direction-index resources for S2.
- `output/figures/`: Figures 1–4 and S1–S4.
- `output/wear/`: aggregate wear-threshold sensitivity outputs.
- `output/tables/crosswalk_exact_knots.csv`: canonical direction-specific mapping parameters.
- `output/tables/crosswalk_direction_metadata.csv`: units, support, errors, version, and hashes for each direction.
- `output/release_validation.json`: contract and numeric-reference audit.

The package-bundled `results/reference/` directory contains aggregate numeric and visual authorities only. It is never an analysis input.

## Tests and integrity

Run lightweight package tests with:

    python -m unittest discover -s tests -v
    python tools/build_manifest.py --package-root . --check MANIFEST.sha256

Scientific contracts, output definitions, provenance, exclusions, and clean-room evidence are documented under `docs/`.

For strict conversion from the bundled references, use `analysis/crosswalk_converter.py`; it rejects unsupported directions and values outside the released source P05–P95 range rather than clipping or extrapolating.

## Privacy and data boundary

No raw data, participant identifiers, person-day/minute tables, fold assignments, participant-level OOF predictions, or bootstrap-replicate rows are redistributed or written. Figures S1/S2 use participant-level arrays only in memory.

## Human completion items

The authors must choose a code license and complete author/repository metadata in `LICENSE_TO_BE_SELECTED.md` and `CITATION.cff` before public archiving. This package does not upload files, mint a DOI, or grant a license by implication.
