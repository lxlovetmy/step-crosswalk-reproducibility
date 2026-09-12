# Release checklist

- [x] Current crosswalk-only scientific scope frozen.
- [x] Twelve input filenames and SHA-256 values recorded.
- [x] One-command smoke/full runner implemented.
- [x] Sample-flow set assertion, sample alignment, exact individual mappings, E/H, bootstrap, boundary, wear, P05–P95 release-domain audit, table, and figure code included.
- [x] Exact-threshold converter and 42-row direction metadata/index included.
- [x] Aggregate numeric and current visual references bundled.
- [x] No raw data or Word files included.
- [x] Historical v1.1.1 300-replicate calculation and terminal validator evidence retained in `validation/`.
- [x] **STEP 2:** run the current single-grid candidate from empty smoke/full output directories, including 300 bootstrap repetitions.
- [x] **STEP 2:** refresh only validated references, Figure 4, provenance, and `MANIFEST.sha256`, then pass final clean-room acceptance.
- [x] **STEP 3:** verify the final internal manifest and versioned ZIP before GitHub release as `v1.1.1`.
- [x] MIT License selected and inserted for the software release.
- [x] Make only `lxlovetmy/step-crosswalk-reproducibility` public after the final history/privacy audit.
- [x] Fix the non-author Zenodo deposit metadata in `docs/ZENODO_DEPOSIT.md`.
- [x] Replace citation author/repository placeholders with the checked four-author metadata.
- [x] Reserve Zenodo DOI `10.5281/zenodo.22723129` and insert it into the package and manuscript candidate.
- [ ] Publish the creator-complete Zenodo software record and verify the DOI after the matching GitHub v1.1.2 release is public.
