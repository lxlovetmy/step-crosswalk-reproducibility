# Clean-room reproduction report

Status: **PASS — CURRENT SINGLE-GRID METHOD**

On 2026-09-11, the package completed a smoke run and two independent full computations from empty output directories using only the twelve checksum-verified public-use inputs. The retained candidate full run and the final clean-room run each used 300 bootstrap repetitions. The clean-room run used a new temporary directory under `/private/tmp`, completed in 3,568.166 seconds, passed the terminal validator with 96/96 checks, and removed its temporary analysis output after recording the report.

The runs rebuilt the 8,646-participant/57,080-person-day cohort, all 42 directed crosswalks, 11,416 exact isotonic knots, common-support outputs, cross-cycle and fixed-threshold analyses, wear-threshold sensitivity, and all manuscript table/figure layers. All required bootstrap-success fields were 300/300.

Common-support H used one eligible-subgroup isotonic fit per full sample or bootstrap resample. The same fitted subgroup curves were used for complete-sample and common-support H, and common-support H was evaluated on 101 equally spaced points in the intersection of eligible subgroup source P05–P95 intervals. The formal path did not use pooled empirical-quantile coordinates or five-fold curve averaging. Five-fold subject-grouped OOF remained limited to E, fixed-threshold, and cross-cycle evaluations.

The first candidate reached the final presentation stage before exposing one output-contract defect: the figure builder emitted the fixed-8000 Figure 1 asset but not the formal `Figure1.png` alias expected by the validator. A byte-identical alias and regression test were added; scientific stages were not rerun for that naming-only fix. Before reference promotion, all eight terminal-validation failures were confined to the five intentionally stale method-affected references. After promoting only those five CSVs and Figure 4, terminal validation passed 96/96.

No raw file, Word document, participant identifier, person-day/minute table, fold assignment, participant-level OOF prediction, or bootstrap-replicate row is included. Machine-readable evidence is in `validation/smoke_report.json`, `validation/full_run_report.json`, `validation/clean_room_report.json`, and `provenance/release_validation.json`.
