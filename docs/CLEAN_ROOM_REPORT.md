# Clean-room reproduction report

Status: **RELEASE CANDIDATE PASS**

The formal calculation was run from an empty output directory using only twelve checksum-verified public-use inputs. It rebuilt the 8,646-participant/57,080-person-day cohort, all 42 directed crosswalks, three independent sets of 300 bootstrap repetitions, wear-threshold sensitivity outputs, and the manuscript table/figure layers. Every reported bootstrap-success field was 300/300, and the corrected terminal harness passed 83/83 checks.

Version 1.1.1 then used that validated formal output as a controlled base because the release change did not alter cohort construction, participant-grouped OOF models, bootstrap algorithms, common-support evaluation, cross-cycle evaluation, fixed-threshold analyses, or wear analyses. From the frozen raw inputs, it rebuilt 11,416 exact isotonic thresholds and 42 direction-metadata rows, regenerated the affected S2 publication files and Figure 2, and removed the formal 101-node dissemination grid and fidelity table. The separate 101-point common-support evaluation grids remain unchanged because they are evaluation coordinates, not conversion parameters.

The v1.1.1 release layer passed 79/79 terminal checks with zero failures. Sixteen package tests passed, including 210 converter boundary checks across all 42 directions. Exact thresholds reconstruct the fitted full-sample models with a maximum error of 0 steps in the validated output. The exact-threshold resource is therefore the sole formal machine-readable conversion parameter set.

An independent v1.1.1 smoke run from another empty output directory verified all twelve input hashes and completed all eight runner modules with two bootstrap repetitions. It reproduced the 8,646/57,080 cohort, all 42 directions, 11,416 exact-knot rows, and 42 direction-metadata rows without generating either retired formal 101-node file.

Bundled aggregate references are used only by the terminal validator after computation. No raw file, Word document, participant identifier, person-day/minute table, fold assignment, participant-level OOF prediction, or bootstrap-replicate row is included. The three Word files were modified separately under user authorization and are recorded only by SHA-256 in the authority manifest; they are not bundled in the code archive. Machine-readable evidence is in `validation/full_run_report.json`, `validation/smoke_report.json`, `validation/test_report.txt`, and `provenance/release_validation.json`.
