# Common-support H release integration plan

**Status:** SCIENTIFIC WORK COMPLETE; RELEASE METADATA FOLLOW-UP
**Version:** 2026-09-11.7
**Owner:** Codex with user review
**Approver:** user
**Approval timestamp:** 2026-09-11T15:59:29Z
**Current phase:** Step 4 — release metadata and manuscript-link alignment; author metadata intentionally deferred
**Baseline:** `main` at `4e07d0f`; dirty working tree recorded in `findings.md` (F-001–F-004).

## Purpose

Finish the reproducibility-package integration for the approved common-support H correction: fit every eligible subgroup isotonic curve once on the complete sample (or each bootstrap resample), retain those same curves for both complete-sample and common-support H, and evaluate common-support H only on 101 equally spaced points in the intersection of eligible subgroup source P05–P95 intervals. Preserve E, complete-sample H, mappings, frozen writing files, and unrelated dirty/untracked work.

## Evidence status

### Confirmed facts

- The formal Stage 6 entry point is `main_release()` in `analysis/build_stage6_common_support.py`; it now defines only `original_grid_stage3` and `common_support_linear` and reuses fitted subgroup models at lines 375–423.
- The formal wear runner calls `support.compute_pair()` and reads only `common_support_linear` for its common-support H field.
- The manuscript and configuration require at least 200 participants in each eligible nonmissing subgroup. The shared `MIN_SUBGROUP_CURVE_N` constant is now 200 for complete-sample H, common-support H, and their bootstrap refits.
- A valid common-support intersection is always evaluated on 101 equally spaced coordinates. The retired requirement for at least 101 pooled observations inside that interval has been removed; the observed count remains descriptive only.
- The current full candidate reported 8,646 participants, 57,080 valid person-days, 42 directions, 300/300 successful common-support bootstrap repetitions, and maximum E and complete-sample H reproduction differences of `8.33e-17` (machine precision).
- Formal reference tables, `analysis/validate_release.py`, and `tests/test_numeric_reference.py` still contain the retired empirical common-support branch. The existing reference row contracts still expect 378 detail rows and 126 continuous-bootstrap rows rather than 252 and 84.
- Current Figure 4 bytes hash is `266f…866e`, while the figure test and two provenance files record `cbf3…7df`; this is a release-integrity failure independent of the scientific calculation.
- The two documents explicitly designated in the current request as read-only authorities are the English manuscript (`82942d485f4d5baea8565a47479e0a3062e81b479fa0015d67a88827f63aab2c`) and SDC (`2af1b22c6bc552d7fe38dc758e9bb1b65d92dd5526461ecbfb4be695616602fc`). Both remained unchanged during Step 2.
- The separately maintained Chinese manuscript changed externally to `4783e6218aee64609b8bdce7248b30051cdffd10c5e7019fee84eb8353b56ba9`; it was preserved and was not used as a current authority or modified by this work.

### Inferences to test

- The review-only full outputs can serve as a comparison oracle, but a new clean-room run from the current integrated package is still required.
- Existing changes to Figure 1, Figure 2, Figure S3, and three `Figure1_fixed8000_attainment.*` assets predate this task's formal integration and are reproducible from the existing figure code; they must not be reverted or silently discarded.

### Unknowns resolved only by execution

- Whether the current source produces the same numeric candidate outputs in a new clean-room environment.
- Which exact, reproducible Figure 4 bytes result from the controlled final render; the final test and provenance must use that actual file, not a remembered hash.

## Scope and protection rules

**In scope:** authoritative Stage 6/common-support release code, wear sensitivity's common-support fields, terminal validator, numeric/contract/figure tests, README and method/output/bootstrap/clean-room documentation, formal aggregate references affected by the new method, Figure 4, provenance, validation evidence, and the package manifest.

**Forbidden:** changes to frozen DOCX files; raw data; mappings; E; complete-sample H; primary models; remote GitHub state; commits; pushes; releases; deletion or rollback of unrelated dirty/untracked files; deletion of historical/archive material solely because it contains old terminology.

## Phase 1 — source and release-contract integration

**Allowed implementation files:** source code, `run_all.py`, `config/analysis.json`, validator, tests, README, current method/output/clean-room/release-checklist documentation, and the three execution records. No reference result, provenance, manifest, Figure, raw input, or DOCX path is allowed in this phase.

1. Snapshot the dirty-tree file list and frozen-writing hashes before editing.
2. Narrowly update the formal validator, tests, and documentation to the single-grid definition. Replace retired formal empirical fields with `common_support_linear` only; leave historical code or archives intact unless they are on a formal release path.
3. Update row/schema contracts to detail=252, summary=42, continuous bootstrap=84; verify 42 directions and 300-success requirements.
4. Add a static release guard that rejects empirical common-support fields, pooled empirical grid wording, and five-fold curve-ensemble construction on formal paths while retaining five-fold subject-grouped OOF for E/fixed-threshold/cross-cycle evaluation.
5. Make Figure 4 authority checks derive from the final reference-manifest record and verify that the record and file agree.

**Validation:** package unit tests (except expected reference drift before Phase 3), static token inventory, syntax/import checks, `git diff --check`.

**Success criteria:** no formal branch calls the retired curve/grid method; validator and tests encode `32/6/4`, displayed median `-0.024`, and the stated method definition; no protected or unrelated file is overwritten.

## Phase 2 — isolated smoke and full candidate reproduction

1. Run `run_all.py --mode smoke` into a new empty directory outside the package with 2 bootstrap reps.
2. Run `run_all.py --mode full` into a second new empty directory with 300 reps. Until references are refreshed, a terminal-reference mismatch is expected only for intentionally changed common-support/wear outputs; retain the generated analysis output for comparison.
3. Compare the candidate to the approved review-only evidence and assert: 8,646/57,080, 42 directions, E and complete-sample H maximum differences <= `1.11e-16`, 300/300 common-support bootstrap success, displayed lower/equal/higher `32/6/4`, median `-0.024`, range `-0.112` to `0.024`, and the specified wear findings including `adept→scrf` at 1296 minutes changing from approximately `0.358` to `0.522`.
4. Audit subject grouping, resampling, curve-fitting calls, and output headers for leakage, OOF misuse, fold-ensemble reconstruction, or mixed old/new outputs.

**Commands:** use the approved locked Python runtime, `PYTHONDONTWRITEBYTECODE=1`, an approved raw-data root supplied through `--data-root`, and fresh output directories outside the package. No output directory may be the package or a package descendant.

**Success criteria:** smoke completes; every scientific result matches the frozen contract; no aggregate output contains participant, day, fold, or replicate identifiers.

### Step 2 execution authorization and checkpoint

The user explicitly authorized the complete second step on 2026-09-11. For this execution, internal Phases 2–4 form one bounded approved phase: fresh smoke, fresh 300-replicate candidate, controlled promotion, provenance/manifest refresh, and a second fresh 300-replicate clean-room acceptance. Execution must stop after local acceptance and must not commit, push, create a release, edit a DOCX, or enter publication/metadata Steps 3–4.

**Allowed writes:** fresh outputs and logs under the workspace-local `.step2_reproduction_20260911/` directory; only the five method-affected reference CSVs and `results/reference/figures/Figure4_current.png`; `provenance/reference_manifest.csv`, `provenance/authority_manifest.json`, `provenance/release_validation.json`; `validation/smoke_report.json`, `validation/full_run_report.json`, and `validation/clean_room_report.json`; current release/method/output/clean-room documentation and the three execution records; `MANIFEST.sha256`; and a narrowly scoped provenance-generation helper/test if required for deterministic regeneration. All other pre-existing dirty/untracked paths remain protected.

**Pre-promotion rollback:** preserve byte-for-byte copies of every allowed formal target under `.step2_reproduction_20260911/pre_promotion_backup/`. If promotion validation fails, restore only those named targets from that backup; never use Git reset or checkout.

**Validation gate:** verify all twelve input hashes before computation; require smoke PASS; require the frozen scientific contract before promotion; require terminal validator PASS, clean-room PASS, all unit tests PASS, manifest verification PASS, privacy/static-path scans PASS, and unchanged frozen-writing hashes.

## Phase 3 — controlled reference and provenance refresh

1. Copy only validated, method-affected candidate artifacts into formal references: common-support detail/summary/bootstrap, wear pair/comparison, and Figure 4. Do not copy unrelated figure outputs merely because a full run created them.
2. Update Table 3 and all generated S4/S7/S8 layers through the normal table builder; verify their common-support fields and row counts. Preserve internal algorithm variable names where output dictionaries distinguish them from display labels.
3. Regenerate `provenance/reference_manifest.csv` and `provenance/authority_manifest.json` from actual local hashes, including the two documents explicitly designated as read-only authorities in the current request. Record method/version/date and final controlled-output hashes without private absolute paths; record but do not absorb unrelated external writing-file drift.
4. Regenerate `MANIFEST.sha256` only after all tracked package outputs and provenance are final.

**Validation:** manifest check; reference-manifest-to-file hash check; Figure 4 file/test/provenance agreement; protected-writing rehash; no unapproved change to unrelated dirty files.

## Phase 4 — final clean-room acceptance

1. Run the terminal validator on the retained full candidate and save the detailed result as package provenance.
2. Run `tools/clean_room_check.py` for a fresh full 300-rep reproduction; it must return `CLEAN_ROOM_CHECK=PASS`.
3. Run package tests, manifest verification, static old-branch scan, output/privacy scan, and before/after diff inventory.
4. Update validation reports and this plan/progress record with evidence, then provide the required PASS/FAIL acceptance matrix and a no-push/no-commit file list for user review.

## Reproducibility controls

- Inputs: checksum-verified public-use inputs only; no bundled reference is an analysis input.
- Seeds: Stage 2 point seed `20260607`; Stage 6 seed `20260715`; 300 PSU-within-stratum bootstrap repetitions.
- Grouping: subject IDs remain together in every OOF fold; bootstrap multiplicities remain grouped by subject.
- Leakage checks: no participant/day/fold/prediction/replicate table; references opened only by terminal validation; common-support curves are full-resample subgroup fits, not OOF or fold averages.

## Risks, retry, and re-plan triggers

- If the full run differs from frozen E/H results beyond tolerance, stop before reference refresh and preserve the candidate/logs for diagnosis.
- If an unrelated dirty file changes, restore only the newly generated target from the immediately prior protected snapshot; do not use broad Git reset or checkout.
- If a raw-input checksum, frozen DOCX hash, or clean-room run fails, stop and report the exact file/check before any release metadata is finalized.
- If final Figure 4 rendering is non-deterministic across two controlled renders, diagnose renderer metadata/version behavior before pinning an authority hash.

## Progress

| Phase | State | Evidence |
|---|---|---|
| Baseline diagnosis | complete | F-001–F-004 |
| 1 source/release-contract integration | complete | static guards and synthetic method tests pass; release contracts/docs updated; expected stale-reference failures recorded in F-009 and F-013 |
| 2 isolated reproduction | complete | smoke PASS; retained full candidate completed all required bootstrap families at 300/300 |
| 3 controlled reference/provenance refresh | complete | only five method-affected CSV references and Figure 4 promoted; terminal validation 96/96 PASS |
| 4 clean-room acceptance | complete | fresh full clean-room run PASS in 3,568.166 s; terminal validation 96/96; final local gates recorded in `progress.md` |

## Decision log

- 2026-09-09: Treat the review-only single-grid output as evidence, not as a formal reference replacement.
- 2026-09-09: Preserve all pre-existing dirty/untracked files and make only narrow, attributable formal-output changes.
- 2026-09-09: Use the final locally rendered Figure 4 bytes as the only authority after controlled regeneration; do not trust the currently conflicting hashes.
- 2026-09-10: User confirmed the current English and Chinese manuscripts as the frozen baseline; the SDC retains its prior frozen hash. Phase 1 is approved.
- 2026-09-10: Phase 1 completed without writing any reference, provenance, manifest, Figure, raw-data, or DOCX file. The five remaining package-test failures are limited to deliberately stale formal references/authority and will be resolved only after a validated Phase 2 candidate is promoted in Phase 3.
- 2026-09-11: A new Step 1 audit found and corrected three code/contract gaps: the configuration still exposed the retired empirical grid, the shared eligibility constant was 100 rather than the manuscript/configured 200, and a valid common-support interval was incorrectly gated on at least 101 observations. Runtime configuration, validator, documentation, and synthetic tests now lock the current method. No protected output was regenerated.
- 2026-09-11: Historical v1.1.1 clean-room evidence is retained but no longer presented as acceptance for the current method. Step 2 requires explicit user confirmation.
- 2026-09-11: User explicitly authorized the complete local Step 2 and requested uninterrupted execution. Internal Phases 2–4 are therefore one bounded execution phase; GitHub and manuscript/publication metadata remain out of scope.
- 2026-09-11: Step 2 completed. Fresh smoke and retained full candidate passed; the five validated method-affected CSV references and Figure 4 were promoted; terminal validation and an independent 300-replicate clean-room run passed 96/96. The task stops for user review before any Step 3 Git operation.
- 2026-09-11: The user authorized Step 3. The validated changes were organized into three commits, pushed to `main`, tagged `v1.1.1`, and published as a GitHub Release with a versioned ZIP and matching SHA-256 asset.
- 2026-09-11: The user authorized Step 4 except author metadata. MIT was selected for the software; a Zenodo version DOI was selected for long-term citation, but publication remains dependent on the creator list that Zenodo requires.

## Outcomes and retrospective

Step 1 passed at the candidate-code level. Step 2 passed local scientific reproduction and clean-room acceptance. Step 3 created the versioned GitHub release. Step 4 is complete for all non-author metadata that can be finalized before Zenodo receives its required creator list.
