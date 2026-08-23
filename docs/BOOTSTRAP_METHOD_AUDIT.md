# Bootstrap Method Audit

## Scope and verdict

This audit covers the three release modules that generate bootstrap intervals:

- `analysis/build_stage4_crosswalk_uncertainty_ci.py`
- `analysis/build_stage6_common_support.py` (`main_release`)
- `analysis/build_stage6_cycle_threshold.py`

**Code-level verdict: PASS with a terminology correction.** The implementation is a stratified PSU-level cluster bootstrap that respects the NHANES stratum/PSU structure. It is not a fully weighted design-based population estimator: the primary crosswalk relation remains unweighted, and WTMEC4YR enters only the weighted fixed-threshold evaluation sensitivity. Therefore, `design-based bootstrap` is stronger than the code supports; use `stratified PSU-within-stratum cluster bootstrap` (or `design-aware cluster bootstrap`) in the manuscript and supplement.

The requested replicate count remains **300**. This audit does not recommend increasing it to 1,000 and does not alter any point estimate, confidence interval, seed, fold count, mapping, or threshold.

## Implementation findings

| Audit item | Finding | Status |
|---|---|---|
| Sampling unit | The observed number of PSUs is sampled with replacement within each SDMVSTRA stratum. | PASS |
| Cluster multiplicity | When a PSU is selected more than once, every participant index in that PSU is repeated once per draw. | PASS |
| Subject grouping | Bootstrap copies of the same participant retain the same identifier; fold assignment is made on unique identifiers, so copies remain in one fold. | PASS |
| Mapping refit | Applicable isotonic mappings are refitted in every replicate. | PASS |
| Fold reconstruction | Subject-grouped folds are rebuilt with a replicate-specific deterministic seed. | PASS |
| Standardization | The target-algorithm IQR is recomputed from the bootstrap sample in every applicable replicate. | PASS |
| Common-support analysis | Support limits, subgroup fits, E, and H are recomputed from each resample. | PASS |
| Cross-cycle analysis | 2011-2012 and 2013-2014 are resampled independently within their own stratum/PSU structures before transport evaluation. | PASS |
| Threshold analysis | The mapping remains unweighted; threshold metrics are evaluated as the unweighted primary analysis and a WTMEC4YR-weighted sensitivity. | PASS |
| Interval construction | Continuous intervals are percentile intervals from the 2.5th and 97.5th percentiles. | PASS |
| Partial-replicate protection | Non-finite metric values are omitted, but the release validator requires every reported bootstrap-success field to equal 300; a hard computation error aborts the pipeline. | PASS at release gate |
| Replicate-level output | No participant, prediction, or replicate-level table is released. | PASS |

## Interpretation boundary

The bootstrap propagates uncertainty from PSU-level resampling, refitting, fold reconstruction, and the replicate-specific IQR denominator for the eligible analytic sample. Because the primary mapping is unweighted and no survey replicate weights or finite-population rescaling are used, it should not be described as establishing nationally representative, fully design-based inference. WTMEC4YR cannot correct selection into the complete all-seven-algorithm analytic cohort.

## Recommended manuscript wording

> Sampling uncertainty was quantified with 300 stratified PSU-level cluster bootstrap replicates. Within each NHANES stratum, the observed number of PSUs was sampled with replacement, with all participants in a selected PSU retained with multiplicity. For each applicable replicate, subject-grouped folds were reconstructed, the relevant isotonic mappings were refitted, and the target-algorithm IQR was recomputed. Percentile 95% confidence intervals were defined by the 2.5th and 97.5th percentiles. The primary crosswalk remained unweighted; WTMEC4YR was used only for the weighted fixed-threshold evaluation sensitivity.

Suggested Chinese companion:

> 抽样不确定性采用 300 次按分层进行的 PSU 级聚类 bootstrap 进行量化。在每个 NHANES 分层内，按该分层实际 PSU 数有放回地抽取 PSU；同一 PSU 被重复抽中时，其所含参与者按抽中次数重复进入重抽样样本。对每个适用的重抽样，重新构建按受试者分组的交叉验证折、重新拟合相关等渗映射，并重新计算目标算法 IQR。95% 置信区间取 bootstrap 分布的第 2.5 和第 97.5 百分位数。主要换算关系仍采用未加权估计；WTMEC4YR 仅用于固定阈值评价的加权敏感性分析。

## Final-output check

**PASS.** The completed formal clean-room output reported `bootstrap_requested=300`, and the minimum and maximum successful replicate counts were both 300 for Stage 4 E/H, Stage 6 continuous common-support, cross-cycle transport, and fixed-threshold interval tables. The original formal validator passed 83/83 checks; after integrating the exact-resource and release-layer changes, the expanded validator passed 86/86 with zero failures. The merged release retains this success-count gate.
