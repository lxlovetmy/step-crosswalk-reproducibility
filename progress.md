# Execution progress

## 2026-09-09 — diagnosis complete

- Read repository instructions and release documentation.
- Recorded Git branch, remote, unstaged/untracked inventory, full relevant diff, and baseline tests.
- Verified the user-specified frozen writing hashes without modifying the files.
- Traced the formal common-support and wear implementations and located all remaining old formal references.
- Wrote `PLAN.md` and `findings.md`; no scientific code, formal output, manifest, provenance, test, raw input, or DOCX file was changed in this diagnosis phase.

Next action requires explicit approval of `PLAN.md`, beginning with Phase 1 only.

## 2026-09-10 — Phase 1 protection checkpoint blocked

- User authorized execution of the plan.
- Pre-phase checkpoint: commit `4e07d0f72f4f96459e8781c0363cf0aeb91dd3cd`; existing dirty/untracked inventory preserved; locked Python is 3.13.9.
- English manuscript and SDC hashes match their frozen contract.
- Chinese manuscript hash is now `25046e4dd51078f8df25a98224321bf4bc3666306458124d1797fe0f9d975af6`, differing from the frozen contract and diagnosis-time value; see F-007.
- No Phase 1 implementation file was edited. Execution is paused pending user direction on the Chinese frozen baseline.

## 2026-09-10 — Phase 1 approved

- User approved the current English and Chinese manuscripts as frozen; SDC remains frozen at its pre-existing approved hash.
- Plan version 2026-09-10.3 is APPROVED for Phase 1 only. Allowed implementation files are listed in `PLAN.md`.

## 2026-09-10 — Phase 1 source and release-contract integration complete

- Updated only the approved validator, tests, README, and method/output/bootstrap documentation paths. No reference output, Figure, manifest, provenance record, raw input, or DOCX file was written by this phase.
- The validator and contracts now require the one-curve `common_support_linear` definition, 101 equally spaced common-support points, 42 directions, 252 detail rows, 84 continuous-bootstrap rows, 300 successful bootstrap repetitions, and displayed common-versus-full counts `32/6/4` with median difference `-0.024`.
- Added source guards confirming that `grouped_folds()` is called only for OOF error evaluation and that neither an empirical common-support grid nor a fold-ensemble curve is selectable on the formal Stage 6/wear path.
- Validation: `git diff --check` passed; the two static release-guard tests passed; syntax/import checks passed. The full suite ran 21 tests with five expected stale-reference/authority failures documented in F-009; no source-method test failed.
- Exact next approved phase: Phase 2 — isolated smoke and full candidate reproduction. It has not started in this execution phase.

## 2026-09-11 — Step 1 candidate-code audit complete

- Re-recorded HEAD, branch/upstream, staged/unstaged/untracked inventory, protected writing hashes, existing reference-figure hashes, provenance hashes, and manifest hash. No existing change was reverted or overwritten.
- Read the current English manuscript and SDC as read-only authorities for subgroup eligibility, curve fitting, common-support evaluation, OOF scope, bootstrap design, and output privacy.
- Corrected three code/contract gaps: removed the retired empirical grid from configuration; unified eligible subgroups at n>=200; and removed the unsupported `pooled_common_n >= 101` estimation gate while retaining 101 equally spaced evaluation coordinates.
- Made `run_all.py` enforce the current configuration before creating outputs and pass the locked 300-repetition/seed settings explicitly to bootstrap stages.
- Expanded the validator and tests to enforce the single grid, n>=200, grid endpoints, model reuse, subject grouping, and PSU-cluster multiplicity. Updated current method/output/clean-room/release-checklist documentation so historical v1.1.1 evidence is not mistaken for acceptance of this candidate.
- Validation: syntax and all formal `--help` import checks passed; locked Python and R dependency checks passed; static formal-path scan passed; eight focused method/configuration/runner tests passed; `git diff --check` passed. The full suite ran 26 tests with 21 passes and five expected stale-reference/authority failures (F-013–F-014).
- Did not run smoke or the 300-replicate full analysis. Did not modify raw data, either specified DOCX, `results/reference`, existing figures, provenance, `MANIFEST.sha256`, Git history, or GitHub.

Next action requires explicit user confirmation: Step 2, beginning with a smoke run into a fresh external output directory, followed by the full 300-replicate reproduction only if smoke passes.

## 2026-09-11 — Step 2 authorized and started

- User explicitly authorized the complete second step and requested uninterrupted execution; no authority was given for GitHub, commits, releases, or manuscript/SDC edits.
- Pre-execution checkpoint: HEAD remains `4e07d0f72f4f96459e8781c0363cf0aeb91dd3cd`; the staged area is empty; all pre-existing dirty and untracked files remain protected.
- Disk availability was 188 GiB. Prior evidence indicates approximately 24 minutes for smoke and 74 minutes for each 300-replicate full run.
- Execution order is fixed as fresh smoke, fresh full candidate, frozen-contract comparison, controlled reference/provenance refresh, terminal validation, fresh clean-room full run, and final package acceptance.

## 2026-09-11 — Step 2 completed

- Verified all twelve public-use input hashes. The fresh smoke run completed all nine stages with two bootstrap repetitions in 1,515.735 seconds.
- The retained full candidate started from an empty output directory, rebuilt 8,646 participants/57,080 valid days and all 42 directions, and completed all required bootstrap families at 300/300.
- A presentation-contract defect was found after scientific completion: `Figure1.png` was missing although its fixed-8000 source asset existed. Added the byte-identical formal alias and a regression test, then rebuilt only tables/figures.
- Pre-promotion terminal validation produced 8/96 failures, all confined to the five intentionally stale common-support/wear references. Promoted only those five aggregate CSVs and Figure 4. Final terminal validation passed 96/96.
- Regenerated the reference and authority manifests from actual files. The designated English manuscript and SDC hashes remained unchanged; no DOCX or raw input was modified.
- The independent clean-room full run completed in 3,568.166 seconds and passed 96/96 terminal checks with zero failures. A repeated Figure 4 render matched the promoted SHA-256 `2ac669ebf2c26412630f9e8984d7550fc40cb37f6231712f82cf2b3eb6ffca6c`.
- Final gates passed: 28/28 package tests; 96/96 terminal checks; Python AST syntax checks across 29 source/test files; all twelve input hashes; aggregate-output privacy headers; formal-path static guards; deterministic Figure 4 hash; and `git diff --check`. `MANIFEST.sha256` was regenerated as the final package write and verified successfully.

Next action after final local gates: stop and obtain explicit user confirmation before Step 3. No commit, push, tag, release, DOI action, or manuscript edit has been performed.
