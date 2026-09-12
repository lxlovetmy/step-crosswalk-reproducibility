#!/usr/bin/env python3
"""Validate scientific contracts, numeric references, figures, and privacy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from release_common import ALGORITHM_LABELS


KEYS = [
    "source_algorithm", "target_algorithm", "algorithm", "reference_threshold_steps",
    "source_input_steps", "metric_family", "group_type", "group_level", "metric",
    "support_definition", "train_cycle", "test_cycle", "threshold_target_steps",
    "analysis_weighting", "wear_threshold", "section", "knot_index", "mapping_id",
]


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    def equal_rows(self, name: str, generated: pd.DataFrame, reference: pd.DataFrame) -> None:
        self.check(name, len(generated) == len(reference), f"generated={len(generated)}, reference={len(reference)}")

    def numeric_compare(self, name: str, generated: pd.DataFrame, reference: pd.DataFrame, atol: float = 1e-8) -> None:
        common_keys = [key for key in KEYS if key in generated and key in reference]
        if common_keys:
            generated = generated.sort_values(common_keys).reset_index(drop=True)
            reference = reference.sort_values(common_keys).reset_index(drop=True)
        common_numeric = [
            column for column in generated.columns.intersection(reference.columns)
            if pd.api.types.is_numeric_dtype(generated[column]) and pd.api.types.is_numeric_dtype(reference[column])
        ]
        if len(generated) != len(reference):
            self.check(name, False, f"row mismatch: {len(generated)}/{len(reference)}")
            return
        if not common_numeric:
            exact = generated.equals(reference)
            self.check(name, exact, f"no numeric columns; exact table match={exact}")
            return
        differences = []
        for column in common_numeric:
            left = pd.to_numeric(generated[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(reference[column], errors="coerce").to_numpy(float)
            valid = np.isfinite(left) & np.isfinite(right)
            nan_match = np.array_equal(np.isnan(left), np.isnan(right))
            maximum = float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else 0.0
            if not nan_match or maximum > atol:
                differences.append(f"{column}:{maximum:.3g}")
        self.check(name, not differences, "max absolute differences=" + (", ".join(differences) if differences else "within tolerance"))


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().eq("true")


def manuscript_median(values: pd.Series) -> float:
    displayed = np.round(values.astype(float).to_numpy(), 3)
    median = Decimal(str(float(np.median(displayed)))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return float(median)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    package, generated = args.package_root.resolve(), args.generated_dir.resolve()
    refs, tables = package / "results" / "reference" / "tables", generated / "tables"
    audit = Audit()
    config = json.loads((package / "config" / "analysis.json").read_text(encoding="utf-8"))
    config_contract_ok = (
        config.get("schema_version") == 4
        and config["individual_crosswalk"].get("oof_folds") == 5
        and config["individual_crosswalk"].get("oof_split_unit") == "participant"
        and config["individual_crosswalk"].get("oof_seed") == 20260607
        and config["evaluation"].get("minimum_subgroup_n") == 200
        and config["evaluation"].get("common_support_grid")
        == {"type": "equally_spaced", "nodes": 101}
        and config["evaluation"].get("common_support_curve_fitting")
        == "one_full_resample_fit_per_eligible_subgroup_reused_for_complete_and_common_support_H"
        and "common_support_grids" not in config["evaluation"]
        and config["bootstrap"].get("replicates") == 300
        and config["bootstrap"].get("stage4_seed") == 20260609
        and config["bootstrap"].get("stage6_seed") == 20260715
    )
    audit.check(
        "analysis_configuration_contract",
        bool(config_contract_ok),
        "participant-grouped 5-fold OOF; n>=200; one equally-spaced 101-point common-support grid; locked seeds",
    )

    mapping = {
        "cohort": (tables / "stage2_all7_cohort_check.csv", refs / "cohort.csv", None),
        "algorithm_distribution": (tables / "stage2_all7_distribution.csv", refs / "algorithm_distribution.csv", "daily_steps"),
        "pairwise_agreement": (tables / "stage2_all7_pairwise_agreement.csv", refs / "pairwise_agreement.csv", None),
        "crosswalk_error": (tables / "stage2_all7_crosswalk_residual_matrix.csv", refs / "crosswalk_error.csv", None),
        "subgroup_counts": (tables / "stage2c_all7_subgroup_counts.csv", refs / "subgroup_counts.csv", None),
        "subgroup_oof": (tables / "stage2c_all7_subgroup_fullsample_oof_residuals.csv", refs / "subgroup_oof_summary.csv", None),
        "subgroup_spread": (tables / "stage2c_all7_subgroup_curve_spread.csv", refs / "subgroup_spread.csv", None),
        "subgroup_grid": (tables / "stage2c_all7_subgroup_curve_grid_daily_headline.csv", refs / "subgroup_grid.csv", None),
        "sample_alignment": (tables / "sample_alignment.csv", refs / "sample_alignment.csv", None),
        "exact_knots": (tables / "crosswalk_exact_knots.csv", refs / "crosswalk_exact_knots.csv", None),
        "direction_metadata": (tables / "crosswalk_direction_metadata.csv", refs / "crosswalk_direction_metadata.csv", None),
        "pair_summary": (tables / "stage3_translatability_map.csv", refs / "pair_summary.csv", None),
        "bootstrap_uncertainty": (tables / "stage4_crosswalk_uncertainty_ci.csv", refs / "bootstrap_uncertainty.csv", None),
        "common_support_detail": (tables / "stage6_common_support_subgroup_stability.csv", refs / "common_support_detail.csv", None),
        "common_support_summary": (tables / "stage6_common_support_summary.csv", refs / "common_support_summary.csv", None),
        "continuous_bootstrap": (tables / "stage6_continuous_bootstrap.csv", refs / "continuous_bootstrap.csv", None),
        "cross_cycle": (tables / "stage6_cross_cycle_transport_validation.csv", refs / "cross_cycle.csv", None),
        "cross_cycle_ci": (tables / "stage6_cross_cycle_transport_ci.csv", refs / "cross_cycle_ci.csv", None),
        "fixed_threshold": (tables / "stage6_fixed_threshold_oof_reclassification.csv", refs / "fixed_threshold.csv", None),
        "fixed_threshold_ci": (tables / "stage6_fixed_threshold_oof_reclassification_ci.csv", refs / "fixed_threshold_ci.csv", None),
        "release_domain_whole": (tables / "release_domain_whole_cohort.csv", refs / "release_domain_whole_cohort.csv", None),
        "release_domain_cross_cycle": (tables / "release_domain_cross_cycle.csv", refs / "release_domain_cross_cycle.csv", None),
        "wear_cohort": (generated / "wear" / "wear_threshold_cohort_summary.csv", refs / "wear_cohort.csv", None),
        "wear_pair": (generated / "wear" / "wear_threshold_pair_metrics.csv", refs / "wear_pair.csv", None),
        "wear_comparison": (generated / "wear" / "wear_threshold_comparison_vs_960.csv", refs / "wear_comparison.csv", None),
        "wear_attainment": (generated / "wear" / "wear_threshold_algorithm_attainment.csv", refs / "wear_attainment.csv", None),
        "wear_fixed": (generated / "wear" / "wear_threshold_fixed_threshold_metrics.csv", refs / "wear_fixed_threshold.csv", None),
        "wear_cross_cycle": (generated / "wear" / "wear_threshold_cross_cycle_transport.csv", refs / "wear_cross_cycle.csv", None),
    }
    for name, (actual_path, reference_path, exposure_filter) in mapping.items():
        actual, reference = read(actual_path), read(reference_path)
        if exposure_filter and "exposure" in actual:
            actual = actual.loc[actual["exposure"].eq(exposure_filter)].reset_index(drop=True)
        audit.equal_rows(f"{name}_row_count", actual, reference)
        tolerance = 1e-8
        if name == "pairwise_agreement":
            tolerance = 1e-6  # rank-correlation tie handling is stable to six decimal places
        elif name.startswith("wear"):
            tolerance = 1e-7
        audit.numeric_compare(f"{name}_numeric_reference", actual, reference, atol=tolerance)

    cohort = read(tables / "stage2_all7_cohort_check.csv")
    cohort_text = cohort.astype(str).to_csv(index=False)
    audit.check("cohort_contract", "8646" in cohort_text and "57080" in cohort_text, "expected 8,646 participants and 57,080 valid days")
    sample = read(tables / "sample_alignment.csv")
    pair = read(tables / "stage3_translatability_map.csv")
    audit.check("sample_alignment_contract", len(sample) == 21, f"rows={len(sample)}")
    sample_method_ok = (
        "conversion_method" in sample
        and sample["conversion_method"].eq(
            "unweighted_empirical_quantile_linear_interpolation"
        ).all()
        and sample["use_case"].eq("sample_marginal_position_alignment_only").all()
    )
    audit.check(
        "sample_alignment_method_contract",
        bool(sample_method_ok),
        "explicit linear interpolation and sample-marginal-only use",
    )
    knots = read(tables / "crosswalk_exact_knots.csv")
    metadata = read(tables / "crosswalk_direction_metadata.csv")
    knot_counts = knots.groupby(["source_algorithm", "target_algorithm"]).size().sort_index()
    metadata_counts = metadata.set_index(["source_algorithm", "target_algorithm"])["exact_knot_nodes"].astype(int).sort_index()
    counts_match = knot_counts.index.equals(metadata_counts.index) and np.array_equal(
        knot_counts.to_numpy(int), metadata_counts.to_numpy(int)
    )
    knot_monotone = knots.sort_values(["source_algorithm", "target_algorithm", "knot_index"]).groupby(["source_algorithm", "target_algorithm"]).apply(
        lambda frame: frame["source_input_steps"].is_monotonic_increasing
        and frame["source_input_steps"].is_unique
        and frame["predicted_target_steps"].is_monotonic_increasing,
        include_groups=False,
    ).all()
    audit.check(
        "exact_knot_contract",
        len(knots) == 11416 and len(knot_counts) == 42 and counts_match and bool(knot_monotone),
        f"rows={len(knots)}, directions={len(knot_counts)}, counts_match_metadata={counts_match}",
    )
    exact_hash = hashlib.sha256((tables / "crosswalk_exact_knots.csv").read_bytes()).hexdigest()
    metadata_index = metadata.set_index(["source_algorithm", "target_algorithm"])
    support_ok = True
    flags_ok = True
    for key, part in knots.groupby(["source_algorithm", "target_algorithm"]):
        row = metadata_index.loc[key]
        low = float(row["released_source_p05_steps"])
        high = float(row["released_source_p95_steps"])
        x = part["source_input_steps"].to_numpy(float)
        support_ok = support_ok and bool(x[0] <= low < high <= x[-1])
        expected_flags = (x >= low) & (x <= high)
        flags_ok = flags_ok and np.array_equal(
            bool_series(part["within_released_p05_p95_support"]).to_numpy(),
            expected_flags,
        )
    metadata_ok = (
        len(metadata) == 42
        and metadata["resource_version"].astype(str).eq("1.1.1").all()
        and metadata["out_of_range_policy"].eq("reject_no_formal_conversion").all()
        and metadata["exact_knots_sha256"].eq(exact_hash).all()
        and metadata["exact_knots_file"].eq("crosswalk_exact_knots.csv").all()
        and metadata["strict_converter_file"].eq("analysis/crosswalk_converter.py").all()
        and support_ok
        and flags_ok
    )
    audit.check(
        "direction_metadata_contract",
        bool(metadata_ok),
        f"rows={len(metadata)}, exact_hash={exact_hash}, support_ok={support_ok}, flags_ok={flags_ok}",
    )
    e, h = pair["E"].astype(float), pair["H"].astype(float)
    audit.check("E_display_contract", manuscript_median(e)==.243 and round(float(e.min()),3)==.123 and round(float(e.max()),3)==.402, f"median/range={manuscript_median(e):.3f}/{e.min():.3f}-{e.max():.3f}")
    audit.check("H_display_contract", manuscript_median(h)==.285 and round(float(h.min()),3)==.142 and round(float(h.max()),3)==.546, f"median/range={manuscript_median(h):.3f}/{h.min():.3f}-{h.max():.3f}")

    row_contracts = {
        "common_support_detail": (tables / "stage6_common_support_subgroup_stability.csv", 252),
        "common_support_summary": (tables / "stage6_common_support_summary.csv", 42),
        "continuous_bootstrap": (tables / "stage6_continuous_bootstrap.csv", 84),
        "cross_cycle": (tables / "stage6_cross_cycle_transport_validation.csv", 84),
        "cross_cycle_ci": (tables / "stage6_cross_cycle_transport_ci.csv", 1176),
        "fixed_threshold": (tables / "stage6_fixed_threshold_oof_reclassification.csv", 252),
        "fixed_threshold_ci": (tables / "stage6_fixed_threshold_oof_reclassification_ci.csv", 1764),
        "release_domain_whole": (tables / "release_domain_whole_cohort.csv", 42),
        "release_domain_cross_cycle": (tables / "release_domain_cross_cycle.csv", 84),
    }
    for name, (path, expected) in row_contracts.items():
        observed = len(read(path)); audit.check(f"{name}_contract", observed == expected, f"rows={observed}, expected={expected}")

    release_whole = read(tables / "release_domain_whole_cohort.csv")
    release_cycle = read(tables / "release_domain_cross_cycle.csv")
    release_pair = release_whole.merge(
        pair[["source_algorithm", "target_algorithm", "E"]],
        on=["source_algorithm", "target_algorithm"],
        validate="one_to_one",
    )
    global_e_difference = float(
        np.max(np.abs(release_pair["global_oof_E"].to_numpy(float) - release_pair["E"].to_numpy(float)))
    )
    whole_identity_difference = float(
        np.max(
            np.abs(
                release_whole["release_domain_oof_E"].to_numpy(float)
                - release_whole["global_oof_E"].to_numpy(float)
                - release_whole["delta_E_domain_minus_global"].to_numpy(float)
            )
        )
    )
    cycle_identity_difference = float(
        np.max(
            np.abs(
                release_cycle["transport_release_domain_E"].to_numpy(float)
                - release_cycle["within_test_oof_release_domain_E"].to_numpy(float)
                - release_cycle["transport_penalty_E_on_same_covered_test"].to_numpy(float)
            )
        )
    )
    audit.check(
        "release_domain_global_E_identity",
        global_e_difference <= 1e-12,
        f"max_abs_difference={global_e_difference:.3g}",
    )
    audit.check(
        "release_domain_delta_identities",
        whole_identity_difference <= 1e-12 and cycle_identity_difference <= 1e-12,
        f"whole={whole_identity_difference:.3g}, cross_cycle={cycle_identity_difference:.3g}",
    )
    release_domain_counts_ok = (
        release_whole["n_total"].astype(int).eq(8646).all()
        and release_whole["n_covered"].astype(int).lt(release_whole["n_total"].astype(int)).all()
        and release_whole["coverage_pct"].astype(float).between(85, 95).all()
        and set(release_cycle["n_train"].astype(int)) == {4284, 4362}
        and set(release_cycle["n_test"].astype(int)) == {4284, 4362}
        and release_cycle["coverage_pct"].astype(float).between(85, 95).all()
    )
    audit.check(
        "release_domain_scope_contract",
        bool(release_domain_counts_ok),
        "whole n=8646; cycle n=4284/4362; coverage required within 85%-95%",
    )
    release_domain_rules_ok = (
        release_whole["range_rule"].eq(
            "inclusive source P05-P95 derived from the training fold only"
        ).all()
        and release_whole["fit_rule"].eq(
            "isotonic fit used the complete training fold without trimming"
        ).all()
        and release_cycle["range_rule"].eq(
            "inclusive source P05-P95 derived from the complete training cycle only"
        ).all()
        and release_cycle["fit_rule"].eq(
            "transport isotonic fit used the complete training cycle without trimming"
        ).all()
        and release_cycle["comparison_rule"].eq(
            "transported and within-test OOF predictions evaluated on identical covered test participants"
        ).all()
    )
    audit.check(
        "release_domain_design_contract",
        bool(release_domain_rules_ok),
        "training-only ranges; untrimmed fits; identical covered cross-cycle comparisons",
    )
    for path in [tables / "stage4_crosswalk_uncertainty_ci.csv", tables / "stage6_continuous_bootstrap.csv", tables / "stage6_cross_cycle_transport_ci.csv", tables / "stage6_fixed_threshold_oof_reclassification_ci.csv"]:
        frame = read(path)
        requested = pd.to_numeric(frame.get("bootstrap_requested"), errors="coerce")
        success_columns = [column for column in frame if "bootstrap_success" in column]
        condition = requested.eq(300).all() and all(pd.to_numeric(frame[column], errors="coerce").eq(300).all() for column in success_columns)
        audit.check(f"bootstrap_300_{path.stem}", bool(condition), f"requested={sorted(requested.dropna().unique().tolist())}")

    wear = read(generated / "wear" / "wear_threshold_cohort_summary.csv")
    observed_wear = {int(row.wear_threshold):(int(row.n_subjects),int(row.n_valid_person_days)) for row in wear.itertuples()}
    expected_wear = {600:(9023,60782),720:(8936,59845),960:(8646,57080),1296:(8066,49564)}
    audit.check("wear_cohort_contract", observed_wear == expected_wear, f"observed={observed_wear}")

    common = read(tables / "stage6_common_support_summary.csv")
    common_detail = read(tables / "stage6_common_support_subgroup_stability.csv")
    common_bootstrap = read(tables / "stage6_continuous_bootstrap.csv")
    common_schema_ok = (
        "common_support_empirical_H" not in common
        and not any("empirical" in column.lower() for column in common.columns)
        and len(set(common["source_algorithm"].astype(str) + "->" + common["target_algorithm"].astype(str))) == 42
        and set(common_detail["grid_definition"].astype(str)) == {"original_grid_stage3", "common_support_linear"}
        and common_detail.groupby("grid_definition").size().to_dict() == {"common_support_linear": 126, "original_grid_stage3": 126}
        and common_bootstrap.groupby("support_definition").size().to_dict() == {"common_support_linear": 42, "original_grid_stage3": 42}
        and common_detail["minimum_level_n"].astype(int).eq(200).all()
        and common_detail.loc[common_detail["grid_definition"].eq("common_support_linear"), "grid_n"].astype(int).eq(101).all()
        and np.allclose(
            common_detail.loc[common_detail["grid_definition"].eq("common_support_linear"), "grid_min"].to_numpy(float),
            common_detail.loc[common_detail["grid_definition"].eq("common_support_linear"), "common_support_low"].to_numpy(float),
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            common_detail.loc[common_detail["grid_definition"].eq("common_support_linear"), "grid_max"].to_numpy(float),
            common_detail.loc[common_detail["grid_definition"].eq("common_support_linear"), "common_support_high"].to_numpy(float),
            rtol=0.0,
            atol=1e-10,
        )
    )
    audit.check(
        "single_grid_common_support_schema",
        bool(common_schema_ok),
        "summary=42; detail=126 original + 126 equally-spaced common-support; bootstrap=42 + 42",
    )
    displayed = pair[["source_algorithm", "target_algorithm", "H"]].merge(
        common[["source_algorithm", "target_algorithm", "common_support_linear_H"]],
        on=["source_algorithm", "target_algorithm"],
    )
    exact_change = displayed["common_support_linear_H"].to_numpy(float) - displayed["H"].to_numpy(float)
    quantum = Decimal("0.001")
    shown_change = [
        Decimal(str(float(common_value))).quantize(quantum, rounding=ROUND_HALF_UP)
        - Decimal(str(float(full_value))).quantize(quantum, rounding=ROUND_HALF_UP)
        for common_value, full_value in zip(
            displayed["common_support_linear_H"], displayed["H"]
        )
    ]
    ordered_change = sorted(shown_change)
    midpoint = len(ordered_change) // 2
    shown_median = (
        (ordered_change[midpoint - 1] + ordered_change[midpoint]) / Decimal(2)
    ).quantize(quantum, rounding=ROUND_HALF_UP)
    shown_lower = sum(value < 0 for value in shown_change)
    shown_unchanged = sum(value == 0 for value in shown_change)
    shown_higher = sum(value > 0 for value in shown_change)
    audit.check(
        "figure4_common_support_display_contract",
        shown_lower == 32 and shown_unchanged == 6 and shown_higher == 4 and shown_median == Decimal("-0.024"),
        f"raw_lower={(exact_change < 0).sum()}, displayed_lower={shown_lower}, displayed_unchanged={shown_unchanged}, displayed_higher={shown_higher}, displayed_median={shown_median}",
    )

    publication = generated / "publication_tables"
    main_counts = [len(read(publication / f"main_table{i}.csv")) for i in range(1,4)]
    audit.check("main_table_contract", main_counts == [22,7,9], f"rows={main_counts}")
    supplemental = list(publication.glob("TableS*.csv"))
    audit.check("supplemental_data_layer", len(supplemental) == 20, f"machine-readable files={len(supplemental)}")
    publication_copies = {
        publication / "TableS4_common_support_detail.csv": tables / "stage6_common_support_subgroup_stability.csv",
        publication / "TableS4_common_support_summary.csv": tables / "stage6_common_support_summary.csv",
        publication / "TableS7_continuous_support_bootstrap.csv": tables / "stage6_continuous_bootstrap.csv",
        publication / "TableS8_wear_threshold_pair_metrics.csv": generated / "wear" / "wear_threshold_pair_metrics.csv",
        publication / "TableS8_wear_threshold_comparison_vs_960.csv": generated / "wear" / "wear_threshold_comparison_vs_960.csv",
    }
    publication_copies_ok = all(
        destination.is_file() and source.is_file() and destination.read_bytes() == source.read_bytes()
        for destination, source in publication_copies.items()
    )
    audit.check(
        "common_support_publication_copy_contract",
        publication_copies_ok,
        "Table S4/S7/S8 files are byte-identical copies of their generated aggregate sources",
    )
    table3 = read(publication / "main_table3.csv")
    expected_table3_common = {
        f"{ALGORITHM_LABELS[row.source_algorithm]} → {ALGORITHM_LABELS[row.target_algorithm]}": float(row.common_support_linear_H)
        for row in common.itertuples()
    }
    table3_common_ok = (
        len(table3) == 9
        and "Common_support_max_spread_over_IQR" in table3
        and all(
            row.Source_to_target in expected_table3_common
            and abs(float(row.Common_support_max_spread_over_IQR) - expected_table3_common[row.Source_to_target]) <= 1e-12
            for row in table3.itertuples()
        )
    )
    audit.check(
        "main_table3_common_support_contract",
        table3_common_ok,
        "all nine Table 3 common-support values come from common_support_linear_H",
    )
    figure_paths = [generated / "figures" / f"Figure{i}.png" for i in range(1,5)] + [generated / "figures" / f"FigureS{i}.png" for i in range(1,5)]
    valid_figures = []
    for path in figure_paths:
        with Image.open(path) as image:
            valid_figures.append(path.stat().st_size > 10_000 and image.width >= 1000 and image.height >= 600)
    audit.check("eight_figure_contract", len(figure_paths)==8 and all(valid_figures), f"valid={sum(valid_figures)}/8")
    figure_log = json.loads((generated / "logs" / "tables_figures_run.json").read_text())
    audit.check("figure4_three_panels", figure_log.get("figure4_panels") == ["A","B","C"], str(figure_log.get("figure4_panels")))
    audit.check(
        "figure4_single_grid_method_contract",
        figure_log.get("figure4_common_support_definition") == "common_support_linear_H from the single equally spaced common-support grid",
        str(figure_log.get("figure4_common_support_definition")),
    )
    sample_flow = figure_log.get("sample_flow_audit", {})
    expected_sample_flow = {
        "positive_mec_weight_subjects": 19151,
        "seven_way_subjects": 14693,
        "seven_way_person_days": 130186,
        "troiano_intersection_subjects": 14689,
        "troiano_intersection_person_days": 130179,
        "valid_wear_subjects_before_adult_filter": 13137,
        "final_adult_subjects": 8646,
        "final_valid_person_days": 57080,
        "seven_way_not_in_positive_mec_weight_n": 0,
        "assertion": "PASS",
    }
    audit.check(
        "sample_flow_set_contract",
        sample_flow == expected_sample_flow,
        f"observed={sample_flow}",
    )

    forbidden_extensions = {".docx", ".xpt", ".sav", ".dta", ".sas7bdat"}
    package_files = [path for path in package.rglob("*") if path.is_file()]
    audit.check("no_symlinks", not any(path.is_symlink() for path in package.rglob("*")), "package tree")
    audit.check("no_raw_or_word_files", not any(path.suffix.lower() in forbidden_extensions or path.name.endswith(".csv.xz") for path in package_files), "package inventory")
    text_files = [path for path in package_files if path.suffix.lower() in {".py",".md",".json",".csv",".yml",".yaml",".toml",".cff",".txt"} and path.stat().st_size < 2_000_000]
    absolute_hits = []
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ("/Users/" + "luoxiang") in text:
            absolute_hits.append(str(path.relative_to(package)))
    audit.check("no_private_absolute_paths", not absolute_hits, f"hits={absolute_hits}")
    forbidden_headers = {"seqn","subject_id","fold_id","bootstrap_replicate","paxdaym"}
    header_hits=[]
    for path in generated.rglob("*.csv"):
        columns={str(c).lower() for c in pd.read_csv(path,nrows=0).columns}
        if columns & forbidden_headers: header_hits.append(f"{path.name}:{sorted(columns&forbidden_headers)}")
    audit.check("aggregate_only_outputs", not header_hits, f"hits={header_hits}")

    failures = [check for check in audit.checks if check["status"] == "FAIL"]
    payload = {"status": "RELEASE_CANDIDATE_PASS" if not failures else "FAIL", "check_count": len(audit.checks), "failure_count": len(failures), "checks": audit.checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(payload["status"]); print(f"CHECKS={len(audit.checks)} FAILURES={len(failures)}")
    if failures:
        for failure in failures: print(f"FAIL {failure['name']}: {failure['detail']}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
