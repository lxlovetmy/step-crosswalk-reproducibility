# Methods-to-code and results mapping

This map records the implemented release entry points. Historical source lineage and hashes remain in `provenance/source_manifest.csv`; numeric and visual authorities are listed in `provenance/reference_manifest.csv`.

| Manuscript component | Release implementation | Principal generated outputs |
|---|---|---|
| Valid-day cohort and seven daily-step series | `analysis/gate0_d9_wear_audit.py`, `analysis/build_mvp_exposure_qc.py`, `analysis/build_stage2_all7_crosswalk_matrix.py` | cohort, distribution, agreement, OOF residual, and reclassification tables |
| Weighted baseline Table 1 | `analysis/build_tables_figures.py` | `publication_tables/main_table1.csv` |
| Sample-level alignment lookup (S1) | `analysis/build_sample_alignment.py` | `tables/sample_alignment.csv` |
| Direction-specific individual resource (S2) | `analysis/build_stage3_release_assembly.py`, `analysis/build_crosswalk_resource.py` | exact isotonic thresholds and 42-row direction metadata/index |
| Strict direction-specific converter | `analysis/crosswalk_converter.py` | in-range conversion from exact thresholds with explicit status codes |
| OOF reproduction error E | `analysis/build_stage2_all7_crosswalk_matrix.py`, `analysis/build_stage3_release_assembly.py` | `tables/stage3_translatability_map.csv` |
| Complete-sample source P05-P95 and common-support H | `analysis/build_stage2c_all7_subgroup_stability.py`, `analysis/build_stage6_common_support.py` | subgroup and common-support tables |
| Fold/cycle-derived P05-P95 release-domain audit | `analysis/build_release_domain_audit.py` | Table S9 aggregate sensitivity files |
| E/H uncertainty | `analysis/build_stage4_crosswalk_uncertainty_ci.py` | `tables/stage4_crosswalk_uncertainty_ci.csv` |
| Cross-cycle and fixed-threshold checks | `analysis/build_stage6_cycle_threshold.py` | cross-cycle and fixed-threshold point/CI tables |
| Wear-time sensitivity | `analysis/build_wear_threshold_sensitivity.py` | six aggregate wear-threshold tables and a run record |
| Main and supplemental table data layers | `analysis/build_tables_figures.py` | `publication_tables/` |
| Figures 1–4 and S1–S4 | `analysis/build_tables_figures.py` | eight PNG files; Figure 4 has panels A–C |
| Terminal scientific, numeric, figure, and privacy audit | `analysis/validate_release.py` | `release_validation.json` |

## Implementation rules

1. Input hashes are verified before analysis; bundled references are opened only by the terminal validator.
2. Sample-level marginal alignment is distinct from direction-specific individual prediction.
3. Source-to-target and target-to-source models are fitted separately; inverse symmetry is never assumed.
4. Full-sample fits generate the exact released mappings. Supplemental Table S2 is a 42-direction index and use-boundary table; formal conversion uses only the exact fitted thresholds. Participant-grouped five-fold OOF predictions evaluate E and fixed-threshold consequences.
5. E remains the primary target-output reproduction error; H remains a support-sensitive subgroup diagnostic. They are not combined into a grade.
6. Only aggregate tables, mapping nodes, figures, and run records are written. Participant, day, minute, fold, OOF-person, and replicate rows are not released.
7. Eligible nonmissing subgroups require at least 200 participants. Common-support H uses the same full-data eligible-subgroup isotonic curves as complete-sample H. It changes only the evaluation grid to 101 equally spaced points in the intersection of eligible subgroup source P05–P95 intervals; observed density within a valid intersection does not change those coordinates, and the method never uses a fold-averaged curve or empirical-quantile grid.
8. Figure 4 uses exact values for plotting and the manuscript's three-decimal direction-level display convention for its 32-lower/6-unchanged/4-higher, median -0.024 annotation; the displayed counts are tested.
