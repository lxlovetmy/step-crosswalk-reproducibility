# Output dictionary

All released CSV files are aggregate or mapping resources. `source_algorithm` and `target_algorithm` use `acti`, `adept`, `oak`, `scrf`, `scssl`, `vs`, and `vsrev`; arrows are directional and are never assumed invertible.

| Resource | Core fields | Interpretation |
|---|---|---|
| `sample_alignment.csv` / Table S1 | Oak anchor, empirical percentile, algorithm-specific equivalent | Sample-marginal position alignment; not an individual prediction |
| `crosswalk_exact_knots.csv` / Table S2 machine-readable resource | direction, exact source thresholds, fitted target thresholds | Sole formal machine-readable mapping; interpolate only between exact thresholds and only within released P05–P95 support |
| `crosswalk_direction_metadata.csv` / Table S2 index | direction, labels, units, support, exact-knot count, E/H, version, hash | Direction-level use contract, Word-table index, and provenance |
| `stage3_translatability_map.csv` / Table S3 | E, H, component subgroup spreads | Continuous direction-level evidence; no grade |
| common-support files / Table S4 | support bounds, subgroup curves, full/common H | Support-sensitivity audit |
| cross-cycle files / Table S5 | train/test cycles, transported and within-test error, penalty | Temporal application audit, not external validation |
| fixed-threshold files / Table S6 | target threshold, OOF classifications, discordance, kappa, degeneracy | Same numerical value on the target-algorithm scale; not health equivalence |
| bootstrap files / Table S7 | point estimate, percentile CI, successful/requested repetitions | Stratified PSU-within-stratum cluster-bootstrap uncertainty; no replicate rows are released |
| wear files / Table S8 | wear threshold, aggregate point estimates and change from 960 | Point-estimate sensitivity; bootstrap not recomputed |
| release-domain files / Table S9 | training-derived source P05–P95 coverage, raw MAE/P95, E, and delta E | Range-restricted sensitivity; model fitting remains untrimmed and the range is not validated |
Main Table 1 has 22 display rows (including three categorical section headings), Table 2 has seven algorithms, and Table 3 has nine illustrative directions. `publication_tables/supplemental_table_file_index.csv` maps the S1–S9 data layer; Table S2 is represented by the exact-knot file and the direction-metadata/index file.
