# Common-support H integration findings

## F-001 — Git baseline (confirmed)

Repository baseline is `main...origin/main` at `4e07d0f`. There are eleven tracked modifications and three untracked Figure 1 assets. No staged changes exist. The sole remote is `origin` and no remote operation has been performed.

The modified formal-method candidates are `analysis/build_stage6_common_support.py`, `analysis/build_wear_threshold_sensitivity.py`, and the common-support portions of `analysis/build_tables_figures.py`. `MANIFEST.sha256`, provenance, Figure 4, and its test are associated release-integrity candidates. Figure 1, Figure 2, Figure S3, the Figure 1 vector/raster assets, and unrelated visual-formatting portions of the table/figure builder are protected pre-existing work.

## F-002 — method trace (confirmed)

`compute_pair()` fits each eligible subgroup model once, evaluates the original P05–P95 grid with those models, and evaluates `np.linspace(common_low, common_high, 101)` with the same model objects. `grouped_folds()` is reached only from grouped OOF MAE code; it is not used to build common-support curves. The wear runner delegates continuous H to this same `compute_pair()` implementation.

## F-003 — remaining formal old branch (confirmed)

The retired identifiers and wording remain in these formal release surfaces: `analysis/validate_release.py`, `tests/test_numeric_reference.py`, and five reference CSV files (`common_support_detail`, `common_support_summary`, `continuous_bootstrap`, `wear_pair`, and `wear_comparison`). Current contracts expect 378 common-support detail rows and 126 continuous-bootstrap rows, whereas the approved single-grid output has 252 and 84 respectively.

## F-004 — current integrity failure (confirmed)

`python -m unittest discover -s tests -v` runs 18 tests and fails only `test_figure4_authority`: the actual Figure 4 SHA-256 is `266f8b2982c3dcc7b8a65836ca035b44e4ffd86691f6628ba810d8ecc4ad86e4`, while the test and both provenance records state `cbf39adbea610dfdf7711a6a17a6f91f36ebfe80fc55d9e49e8754817c47d8df`. `tools/build_manifest.py --check MANIFEST.sha256` currently passes, so the discrepancy is specifically between Figure 4 and non-manifest authorities.

## F-005 — frozen writing files (confirmed)

The user-specified English manuscript, Chinese manuscript, and SDC hashes match exactly at baseline. They are outside the package and must be rechecked only, never written, during release integration.

## F-006 — review-only evidence (confirmed)

The supplied single-grid review bundle reports the requested cohort, bootstrap, method, and Figure 4 display values. Its candidate common-support and wear tables differ from the package references exactly where the retired branch remains. The bundle is not a replacement for a new clean-room execution.

## F-007 — Chinese frozen writing-file hash changed after diagnosis (confirmed)

At the Phase 1 protection checkpoint, the Chinese manuscript's SHA-256 was `25046e4dd51078f8df25a98224321bf4bc3666306458124d1797fe0f9d975af6`; its modification time was 2026-09-09T22:35:56Z. This differs from both the user-provided frozen hash and the diagnosis-time value. The English manuscript and SDC still match their user-provided hashes. No writing file was modified by this work.

## F-008 — frozen-writing baseline resolution (confirmed)

The user approved the current English and Chinese manuscripts as the frozen baseline. The SDC remains frozen at its original approved hash. No writing file is in scope for edits.

## F-009 — Phase 1 contracts are intentionally ahead of stale references (confirmed)

After the Phase 1 validator/test/documentation update, the two static method guards pass and `git diff --check` passes. The full package suite runs 21 tests, with five expected failures confined to old formal references: detail rows `378` rather than `252`; continuous-bootstrap rows `126` rather than `84`; Figure 4's file bytes versus the old authority hash; and the retired reference values yielding displayed common-versus-full counts `37/0/5` rather than the required `32/6/4`. No source-method guard fails. These references must not be edited until a new full candidate has passed Phase 2.

## F-010 — 2026-09-11 protected baseline (confirmed)

The repository remains `main...origin/main` at `4e07d0f72f4f96459e8781c0363cf0aeb91dd3cd`, with no staged changes and no remote operation. At this audit's start there were 18 modified tracked files and seven untracked files. The English manuscript hash was `82942d485f4d5baea8565a47479e0a3062e81b479fa0015d67a88827f63aab2c`; the SDC hash was `2af1b22c6bc552d7fe38dc758e9bb1b65d92dd5526461ecbfb4be695616602fc`. Both remained unchanged. Existing modifications to reference figures, provenance, and `MANIFEST.sha256` were recorded and not written by this audit. No physical `AGENTS.md` exists under the current workspace/repository; the instructions supplied in the user message were followed.

## F-011 — three current-method contract gaps corrected (confirmed)

The formal configuration still declared `pooled_empirical_nodes`; the shared subgroup threshold was 100 although the manuscript and configuration specify at least 200; and Stage 6 plus the wear reason helper rejected otherwise valid common-support intersections when fewer than 101 observations fell inside them. The configuration now defines one equally spaced 101-node grid, the shared eligibility threshold is 200, and every positive eligible-subgroup P05–P95 intersection is evaluated at 101 coordinates regardless of its descriptive observation count.

## F-012 — formal call chain and ownership (confirmed)

`run_all.py` verifies the twelve frozen input hashes before analysis and rejects package-internal or nonempty output directories. It then invokes, in order, cohort/crosswalk construction, subgroup H, sample alignment, direction-level assembly, exact resources, Stage 4 bootstrap, Stage 6 common support, cross-cycle/fixed-threshold bootstrap, and the release-domain audit. Full mode alone adds wear sensitivity, tables/figures, and terminal validation. Common-support H has one computational owner, `compute_pair()` in `analysis/build_stage6_common_support.py`; wear sensitivity delegates to it, and Table 3/Figure 4/terminal validation consume only `common_support_linear_H`. Configured 300 repetitions and Stage 4/Stage 6 seeds are passed explicitly by the runner.

## F-013 — Step 1 validation result (confirmed)

All formal script `--help` import checks and the static release-path scan passed. The locked Python 3.13.9 dependency versions match the environment records, and the R 4.6.0/haven 2.5.5/survey 4.5/dplyr 1.2.1 runtime check passes. Eight focused method/configuration/runner tests passed, including direct synthetic evidence that eight eligible subgroup curves are fitted eight times—not once per grid—and each fitted object is used on both 101-point grids; duplicated subject IDs never cross Stage 6 folds; PSU resampling retains every cluster member with the same multiplicity; and the formal runner stage order is complete. The full suite now contains 26 tests: 21 pass and five fail only because bundled authorities still encode the retired method. `git diff --check` passes.

## F-014 — failures reserved for Step 2 (confirmed)

The stale references still contain 378 common-support detail rows instead of 252, 126 continuous-bootstrap rows instead of 84, and displayed lower/equal/higher counts `37/0/5` instead of `32/6/4`. Figure 4 is `266f8b2982c3dcc7b8a65836ca035b44e4ffd86691f6628ba810d8ecc4ad86e4`, while `provenance/reference_manifest.csv` records `cbf39adbea610dfdf7711a6a17a6f91f36ebfe80fc55d9e49e8754817c47d8df`. The package manifest check also fails after the source/documentation updates. These are reference/authority/hash failures, not reasons to modify results manually in Step 1.

## F-015 — Step 2 protected-writing checkpoint (confirmed)

Immediately before execution, the two documents explicitly designated by the user as read-only authorities remained unchanged: English manuscript SHA-256 `82942d485f4d5baea8565a47479e0a3062e81b479fa0015d67a88827f63aab2c` and SDC SHA-256 `2af1b22c6bc552d7fe38dc758e9bb1b65d92dd5526461ecbfb4be695616602fc`. The separately maintained Chinese manuscript had changed externally to `4783e6218aee64609b8bdce7248b30051cdffd10c5e7019fee84eb8353b56ba9`; it was preserved, excluded from the two-document authority manifest, and not used to block this scientific reproduction.

## F-016 — fresh smoke and full reproduction (confirmed)

The smoke run from an empty output directory passed all nine stages with two bootstrap repetitions. The independent retained full candidate verified all twelve input hashes, rebuilt 8,646 participants/57,080 valid person-days and 42 directions, and completed every required bootstrap family at 300/300. Stage 6 emitted 252 detail, 42 summary, and 84 continuous-bootstrap rows. Maximum E and complete-sample H reproduction differences were both `8.326672684688674e-17`, within the machine-precision contract.

## F-017 — Figure 1 formal-name contract defect fixed (confirmed)

The first full candidate completed all scientific stages but the terminal validator stopped because the builder emitted `Figure1_fixed8000_attainment.png` without the formal `Figure1.png` alias. The builder now emits a byte-identical alias, and a regression test locks that contract. Only the presentation layer was rebuilt; no scientific result was changed or recomputed for this fix.

## F-018 — controlled promotion and terminal validation (confirmed)

Before promotion, 8 of 96 checks failed and every failure was confined to the five deliberately stale common-support/wear reference CSVs. After copying only those five validated aggregate CSVs and the regenerated Figure 4, the terminal validator passed 96/96. Displayed common-versus-complete-sample H counts are `32/6/4`, median change `-0.024`, and range `-0.112` to `0.024`. For ADEPT→Step-RF at 1,296 minutes, common-support H changed from `0.357671` at the 960-minute baseline to `0.5221004`.

## F-019 — clean-room acceptance (confirmed)

The final clean-room command ran the complete 300-replicate pipeline in a new temporary output directory, finished in 3,568.166 seconds, returned code 0, and passed the terminal validator at 96/96 with zero failures. The temporary output was removed after the report was written. Repeated controlled Figure 4 rendering reproduced SHA-256 `2ac669ebf2c26412630f9e8984d7550fc40cb37f6231712f82cf2b3eb6ffca6c`, identical to the promoted visual reference.

## F-020 — GitHub release completed (confirmed)

Step 3 produced three scoped commits ending at `a1ae3b7`, pushed `main`, created annotated tag `v1.1.1`, and published the GitHub Release at `https://github.com/lxlovetmy/step-crosswalk-reproducibility/releases/tag/v1.1.1`. The uploaded versioned ZIP was 5,498,887 bytes and its local, checksum-file, and GitHub asset SHA-256 values all matched `6dbc107e0e25e70ab29cca1a30c0e5f366e62fdf22be8785118aeaa6e2902bec`. No participant data or Word manuscript was included.

## F-021 — Step 4 non-author decisions (confirmed)

The user delegated all Step 4 decisions except author identity. MIT was selected as the software license because the release is intended for broad reuse with attribution and warranty limitation. A version-specific Zenodo DOI was selected for preservation and citation. Zenodo publication cannot be completed before at least one creator is supplied, so the non-author deposit metadata is frozen in `docs/ZENODO_DEPOSIT.md` and DOI insertion remains intentionally pending. English and Chinese manuscript copies were created without overwriting their source documents; their updated Code Availability statements point to the exact GitHub `v1.1.1` release, identify the MIT License and checksum asset, and preserve the participant-data and Word-file exclusions.

## F-022 — release-archive helper hardened (confirmed)

The published Step 3 ZIP was produced from the Git tag and contains no `.git` path. A separate audit found that `tools/create_release_archive.py`, if used directly in the future, did not explicitly exclude repository internals and did not reject an output path inside the package. The helper now excludes `.git` and common cache directories and requires the archive to be written outside the package root. The final archive inspection remains the release authority.

## F-023 — public-release boundary audit (confirmed)

The complete Git history contains no raw-data/Word/spreadsheet/archive path, personal absolute path, credential signature, or blob larger than 50 MB. The two `/private/tmp` occurrences are intentional generic runtime references in the clean-room tool and report, not user-specific paths. The public visibility decision applies only to `lxlovetmy/step-crosswalk-reproducibility`; all other repositories in the account remain unchanged.

## F-024 — creator metadata and Zenodo DOI reserved (confirmed)

The approved creator sequence is Xiang Luo, Qinlong Li, Qiang Dong, and Yue Zhou. All four ORCIDs and affiliations were entered into `CITATION.cff` and the creator-complete Zenodo Software draft. Zenodo reserved version DOI `10.5281/zenodo.22723129` on 2026-09-12. The v1.1.2 follow-up is metadata-only: no scientific source, aggregate reference result, or 300-replicate validation evidence was changed. Before release finalization, all 28 package tests and all 96 terminal validation checks passed.
