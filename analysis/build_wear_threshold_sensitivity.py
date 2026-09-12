#!/usr/bin/env python3
"""Aggregate-only valid-day wear-threshold sensitivity point estimates.

This release runner evaluates 600, 720, and 1296 minutes against the 960-minute
main analysis generated earlier in the same clean run. It never invokes a
bootstrap loop or writes participant, person-day, fold, prediction, or
replicate-level outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_ROOT = SCRIPT_PATH.parent
EXPECTED_THRESHOLDS = (600, 720, 1296)
BASELINE_THRESHOLD = 960
EXPECTED_ALGORITHMS = ("acti", "adept", "oak", "scrf", "scssl", "vs", "vsrev")
EXPECTED_CYCLES = ("2011-2012_G", "2013-2014_H")

RAW_REQUIRED_FILES = (
    "nhanes_1440_troianowear.csv.xz",
    "nhanes_1440_actisteps.csv.xz",
    "nhanes_1440_adeptsteps.csv.xz",
    "nhanes_1440_oaksteps.csv.xz",
    "nhanes_1440_scrfsteps.csv.xz",
    "nhanes_1440_scsslsteps.csv.xz",
    "nhanes_1440_vssteps.csv.xz",
    "nhanes_1440_vsrevsteps.csv.xz",
)

PAIR_FILE = "wear_threshold_pair_metrics.csv"
FIXED_FILE = "wear_threshold_fixed_threshold_metrics.csv"
ATTAINMENT_FILE = "wear_threshold_algorithm_attainment.csv"
TRANSPORT_FILE = "wear_threshold_cross_cycle_transport.csv"
COHORT_FILE = "wear_threshold_cohort_summary.csv"
COMPARISON_FILE = "wear_threshold_comparison_vs_960.csv"
REPORT_FILE = "WEAR_THRESHOLD_SENSITIVITY_REPORT.md"
RUN_FILE = "wear_threshold_sensitivity_run.json"


class Blocked(RuntimeError):
    """A prompt-defined hard stop or an unrecoverable direct error."""


def log(message: str) -> None:
    print(f"[wear-threshold {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate-only point-estimate sensitivity for approved daily wear thresholds."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Output root from the main analysis in this same run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=int, default=list(EXPECTED_THRESHOLDS))
    parser.add_argument("--baseline-threshold", type=int, default=BASELINE_THRESHOLD)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise Blocked(f"Refusing to write an empty required aggregate table: {path.name}")
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def as_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return math.nan


def finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def numeric_or_na(value: float) -> float | str:
    return value if math.isfinite(value) else "NA"


def bool_from_csv(value: object) -> bool:
    return str(value).strip().lower() == "true"


def expected_pairs() -> set[tuple[str, str]]:
    return {(source, target) for source in EXPECTED_ALGORITHMS for target in EXPECTED_ALGORITHMS if source != target}


def ensure_unique(rows: list[dict[str, object]], keys: tuple[str, ...], label: str) -> None:
    observed = [tuple(str(row.get(key, "")) for key in keys) for row in rows]
    if len(observed) != len(set(observed)):
        raise Blocked(f"Duplicate primary key in {label}: {keys}")


def baseline_tables(baseline_dir: Path) -> dict[str, Any]:
    table_root = baseline_dir / "tables"
    stage3 = read_csv(table_root / "stage3_translatability_map.csv")
    common = read_csv(table_root / "stage6_common_support_summary.csv")
    fixed = read_csv(table_root / "stage6_fixed_threshold_oof_reclassification.csv")
    transport = read_csv(table_root / "stage6_cross_cycle_transport_validation.csv")
    if (len(stage3), len(common), len(fixed), len(transport)) != (42, 42, 252, 84):
        raise Blocked("Baseline aggregate table row count mismatch.")
    pair_set = expected_pairs()
    for label, rows in (("stage3", stage3), ("common-support", common)):
        pairs = {(row["source_algorithm"], row["target_algorithm"]) for row in rows}
        if pairs != pair_set or len(rows) != 42:
            raise Blocked(f"Baseline {label} pair key set mismatch.")
    fixed_keys = {
        (
            row["source_algorithm"], row["target_algorithm"], float(row["threshold_target_steps"]), row["analysis_weighting"]
        )
        for row in fixed
    }
    expected_fixed = {
        (source, target, threshold, weighting)
        for source, target in pair_set
        for threshold in (7000.0, 8000.0, 10000.0)
        for weighting in ("unweighted_primary", "WTMEC4YR_weighted_sensitivity")
    }
    if fixed_keys != expected_fixed:
        raise Blocked("Baseline fixed-threshold key set mismatch.")
    transport_keys = {
        (row["source_algorithm"], row["target_algorithm"], row["train_cycle"], row["test_cycle"])
        for row in transport
    }
    expected_transport = {
        (source, target, train, test)
        for source, target in pair_set
        for train, test in ((EXPECTED_CYCLES[0], EXPECTED_CYCLES[1]), (EXPECTED_CYCLES[1], EXPECTED_CYCLES[0]))
    }
    if transport_keys != expected_transport:
        raise Blocked("Baseline cross-cycle transport key set mismatch.")
    return {
        "stage3": {(row["source_algorithm"], row["target_algorithm"]): row for row in stage3},
        "common": {(row["source_algorithm"], row["target_algorithm"]): row for row in common},
        "fixed": {
            (row["source_algorithm"], row["target_algorithm"], float(row["threshold_target_steps"]), row["analysis_weighting"]): row
            for row in fixed
        },
        "transport": {
            (row["source_algorithm"], row["target_algorithm"], row["train_cycle"], row["test_cycle"]): row
            for row in transport
        },
    }


def assert_required_raw_files(raw_dir: Path) -> None:
    raw_physio = raw_dir / "physionet_v1.0.1"
    for name in RAW_REQUIRED_FILES:
        path = raw_physio / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise Blocked(f"Required nonempty compressed input is missing: {path}")


def check_arguments(args: argparse.Namespace) -> None:
    if tuple(args.thresholds) != EXPECTED_THRESHOLDS:
        raise Blocked(f"Approved threshold order is exactly {EXPECTED_THRESHOLDS}; received {tuple(args.thresholds)}.")
    if args.baseline_threshold != BASELINE_THRESHOLD:
        raise Blocked(f"Approved baseline threshold is exactly {BASELINE_THRESHOLD}.")
    if args.output_dir.resolve() == args.baseline_dir.resolve():
        raise Blocked("Wear-sensitivity output must be separate from the main baseline directory.")


def prepare_output_directory(output_dir: Path, resume: bool) -> None:
    if output_dir.exists():
        contents = list(output_dir.iterdir())
        if contents and not resume:
            raise Blocked("Output directory already contains files; refusing to overwrite.")
        if contents and resume:
            known = {PAIR_FILE, FIXED_FILE, ATTAINMENT_FILE, TRANSPORT_FILE, COHORT_FILE, COMPARISON_FILE, REPORT_FILE, RUN_FILE}
            unexpected = [item.name for item in contents if item.name not in known and not item.name.endswith(".tmp")]
            if unexpected:
                raise Blocked(f"Output directory contains unrecognized files: {unexpected}")
    else:
        output_dir.mkdir(parents=True)


def load_pipeline() -> dict[str, Any]:
    sys.dont_write_bytecode = True
    scripts_path = str(ANALYSIS_ROOT)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    mvp = importlib.import_module("build_mvp_exposure_qc")
    stage2 = importlib.import_module("build_stage2_all7_crosswalk_matrix")
    stage2c = importlib.import_module("build_stage2c_all7_subgroup_stability")
    support = importlib.import_module("build_stage6_common_support")
    cycle = importlib.import_module("build_stage6_cycle_threshold")
    if tuple(stage2.ALGORITHMS) != EXPECTED_ALGORITHMS:
        raise Blocked("Frozen seven-algorithm set differs from the approved set.")
    if int(mvp.D9_WEAR_THRESHOLD_MINUTES) != BASELINE_THRESHOLD:
        raise Blocked("Frozen D9 threshold global is not 960 before parameterization.")
    if int(mvp.D9_MIN_VALID_DAYS) != 3:
        raise Blocked("Frozen minimum valid-day count differs from 3.")
    if int(support.FOLDS) != 5 or int(cycle.FOLDS) != 5:
        raise Blocked("Frozen five-fold subject-grouped OOF definition is unavailable.")
    if int(support.STAGE2_RANDOM_SEED) != 20260607 or int(cycle.CROSSWALK_SEED) != 20260607:
        raise Blocked("Frozen crosswalk seed differs from 20260607.")
    return {"mvp": mvp, "stage2": stage2, "stage2c": stage2c, "support": support, "cycle": cycle}


@contextlib.contextmanager
def temporary_wear_threshold(mvp: Any, value: int) -> Iterator[None]:
    original = mvp.D9_WEAR_THRESHOLD_MINUTES
    mvp.D9_WEAR_THRESHOLD_MINUTES = value
    try:
        yield
    finally:
        mvp.D9_WEAR_THRESHOLD_MINUTES = original


def build_parameterized_cohort(pipeline: dict[str, Any], raw_dir: Path, threshold: int) -> Any:
    mvp = pipeline["mvp"]
    stage2c = pipeline["stage2c"]
    with temporary_wear_threshold(mvp, threshold):
        cohort = stage2c.build_daily_cohort(raw_dir)
    if int(mvp.D9_WEAR_THRESHOLD_MINUTES) != BASELINE_THRESHOLD:
        raise Blocked("Threshold restoration failed after parameterized cohort construction.")
    return cohort


def h_reason_for_pair(cohort: Any, sample_idx: np.ndarray, source: str, support: Any, definition: str) -> str:
    x_all = np.asarray(cohort.daily_by_alg[source][sample_idx], dtype=float)
    valid = np.isfinite(x_all)
    reasons: list[str] = []
    minimum = support.MIN_SUBGROUP_CURVE_N if definition == support.ORIGINAL_DEFINITION else support.MIN_COMMON_SUPPORT_LEVEL_N
    for group_type in ("age_group", "sex", "bmi_group"):
        all_groups = support.group_values_for(cohort, sample_idx, group_type)
        groups = all_groups[valid]
        x = x_all[valid]
        levels = support.eligible_levels(groups, group_type, minimum)
        if len(levels) < 2:
            reasons.append(f"{group_type}:LT2_LEVELS_N_GE_{minimum}")
            continue
        if definition == support.ORIGINAL_DEFINITION:
            return "ESTIMATED"
        p05 = [support.quantile(x[mask], 0.05) for _level, mask, _n in levels]
        p95 = [support.quantile(x[mask], 0.95) for _level, mask, _n in levels]
        low, high = max(p05), min(p95)
        if not (support.finite(low) and support.finite(high) and low < high):
            reasons.append(f"{group_type}:NONPOSITIVE_COMMON_SUPPORT")
            continue
        return "ESTIMATED"
    return ";".join(reasons) if reasons else "NOT_ESTIMABLE_UNSPECIFIED"


def pair_metric_row(threshold: int, computation: Any, baseline: dict[str, Any], cohort: Any, all_idx: np.ndarray, support: Any) -> dict[str, Any]:
    key = (computation.source, computation.target)
    stage3 = baseline["stage3"][key]
    common = baseline["common"][key]
    metric_values = {
        "grouped_oof_mae_per_target_iqr": float(computation.mae_per_target_iqr),
        "full_range_h_per_target_iqr": float(computation.spread_by_definition[support.ORIGINAL_DEFINITION]),
        "common_support_linear_h_per_target_iqr": float(computation.spread_by_definition[support.COMMON_LINEAR_DEFINITION]),
    }
    baseline_values = {
        "grouped_oof_mae_per_target_iqr": as_float(stage3["mae_per_target_iqr"]),
        "full_range_h_per_target_iqr": as_float(stage3["max_subgroup_p95_spread_over_target_iqr"]),
        "common_support_linear_h_per_target_iqr": as_float(common["common_support_linear_H"]),
    }
    row: dict[str, Any] = {
        "wear_threshold": threshold,
        "source_algorithm": computation.source,
        "target_algorithm": computation.target,
        "n": computation.n,
        "target_iqr": numeric_or_na(float(computation.target_iqr)),
        **{column: numeric_or_na(value) for column, value in metric_values.items()},
        "full_range_h_driver": computation.driver_by_definition[support.ORIGINAL_DEFINITION] or "NA",
        "common_support_linear_h_driver": computation.driver_by_definition[support.COMMON_LINEAR_DEFINITION] or "NA",
        "full_range_h_reason": "ESTIMATED" if finite(metric_values["full_range_h_per_target_iqr"]) else h_reason_for_pair(cohort, all_idx, computation.source, support, support.ORIGINAL_DEFINITION),
        "common_support_linear_h_reason": "ESTIMATED" if finite(metric_values["common_support_linear_h_per_target_iqr"]) else h_reason_for_pair(cohort, all_idx, computation.source, support, support.COMMON_LINEAR_DEFINITION),
    }
    for column, value in metric_values.items():
        base = baseline_values[column]
        row[f"baseline_960_{column}"] = numeric_or_na(base)
        row[f"delta_vs_960_{column}"] = numeric_or_na(value - base) if math.isfinite(value) and math.isfinite(base) else "NA"
    return row


def enrich_fixed_rows(threshold: int, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    metric_columns = (
        "observed_target_attainment_pct",
        "mapped_source_attainment_pct",
        "attainment_difference_pp",
        "discordant_reclassification_pct",
        "cohen_kappa",
        "positive_agreement",
        "negative_agreement",
    )
    enriched: list[dict[str, Any]] = []
    for source_row in rows:
        row = {"wear_threshold": threshold, **source_row}
        key = (
            str(row["source_algorithm"]),
            str(row["target_algorithm"]),
            float(row["threshold_target_steps"]),
            str(row["analysis_weighting"]),
        )
        base = baseline["fixed"].get(key)
        if base is None:
            raise Blocked(f"Missing complete-key 960 fixed-threshold baseline for {key}")
        row["baseline_960_n_rows"] = int(float(base["n_rows"]))
        row["baseline_960_target_threshold_degenerate"] = base["target_threshold_degenerate"]
        for column in metric_columns:
            current = as_float(row[column])
            prior = as_float(base[column])
            row[f"baseline_960_{column}"] = numeric_or_na(prior)
            row[f"delta_vs_960_{column}"] = numeric_or_na(current - prior) if math.isfinite(current) and math.isfinite(prior) else "NA"
        enriched.append(row)
    return enriched


def algorithm_attainment_rows(threshold: int, fixed_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in EXPECTED_ALGORITHMS:
        for target_steps in (7000.0, 8000.0, 10000.0):
            for weighting in ("unweighted_primary", "WTMEC4YR_weighted_sensitivity"):
                current = [
                    row for row in fixed_rows
                    if row["target_algorithm"] == target
                    and float(row["threshold_target_steps"]) == target_steps
                    and row["analysis_weighting"] == weighting
                ]
                if len(current) != 6:
                    raise Blocked(f"Expected six source directions for attainment key {(target, target_steps, weighting)}")
                values = [as_float(row["observed_target_attainment_pct"]) for row in current]
                if not all(math.isfinite(value) for value in values) or not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
                    raise Blocked(f"Observed target attainment differs across source directions for {(target, target_steps, weighting)}")
                baseline_rows = [
                    baseline["fixed"][(source, target, target_steps, weighting)]
                    for source in EXPECTED_ALGORITHMS if source != target
                ]
                baseline_values = [as_float(row["observed_target_attainment_pct"]) for row in baseline_rows]
                if not np.allclose(baseline_values, baseline_values[0], rtol=0.0, atol=1e-12):
                    raise Blocked(f"Baseline observed target attainment differs across source directions for {(target, target_steps, weighting)}")
                output.append(
                    {
                        "wear_threshold": threshold,
                        "target_algorithm": target,
                        "threshold_target_steps": target_steps,
                        "analysis_weighting": weighting,
                        "observed_target_attainment_pct": values[0],
                        "baseline_960_observed_target_attainment_pct": baseline_values[0],
                        "delta_vs_960_observed_target_attainment_pct": values[0] - baseline_values[0],
                        "n_source_directions_checked": len(current),
                    }
                )
    if len(output) != 42:
        raise Blocked("Algorithm attainment output is not exactly 42 rows.")
    return output


def enrich_transport_rows(threshold: int, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for source_row in rows:
        row = {"wear_threshold": threshold, **source_row}
        key = (row["source_algorithm"], row["target_algorithm"], row["train_cycle"], row["test_cycle"])
        base = baseline["transport"].get(key)
        if base is None:
            raise Blocked(f"Missing complete-key 960 transport baseline for {key}")
        for column in (
            "transport_mae_per_target_iqr",
            "within_test_oof_mae_per_target_iqr",
            "transport_penalty_mae_per_target_iqr",
        ):
            current, prior = as_float(row[column]), as_float(base[column])
            row[f"baseline_960_{column}"] = numeric_or_na(prior)
            row[f"delta_vs_960_{column}"] = numeric_or_na(current - prior) if math.isfinite(current) and math.isfinite(prior) else "NA"
        enriched.append(row)
    return enriched


def cohort_row(threshold: int, cohort: Any, cycles: np.ndarray) -> dict[str, Any]:
    n_g = int(np.sum(cycles == EXPECTED_CYCLES[0]))
    n_h = int(np.sum(cycles == EXPECTED_CYCLES[1]))
    baseline = {"n_subjects": 8646, "n_valid_person_days": 57080, "n_subjects_2011_2012_G": 4284, "n_subjects_2013_2014_H": 4362}
    row: dict[str, Any] = {
        "wear_threshold": threshold,
        "analysis_role": "sensitivity_point_estimate" if threshold in (600, 720) else "post_hoc_strict_stress_test_point_estimate",
        "n_subjects": int(cohort.subjects_n),
        "n_valid_person_days": int(cohort.valid_days_n),
        "n_subjects_2011_2012_G": n_g,
        "n_subjects_2013_2014_H": n_h,
    }
    for name, baseline_value in baseline.items():
        current = int(row[name])
        row[f"baseline_960_{name}"] = baseline_value
        row[f"delta_vs_960_{name}"] = current - baseline_value
        row[f"pct_delta_vs_960_{name}"] = 100.0 * (current - baseline_value) / baseline_value
    return row


def baseline_cohort_row() -> dict[str, Any]:
    return {
        "wear_threshold": BASELINE_THRESHOLD,
        "analysis_role": "existing_frozen_main_analysis_baseline_light_cohort_check_only",
        "n_subjects": 8646,
        "n_valid_person_days": 57080,
        "n_subjects_2011_2012_G": 4284,
        "n_subjects_2013_2014_H": 4362,
        "baseline_960_n_subjects": 8646,
        "delta_vs_960_n_subjects": 0,
        "pct_delta_vs_960_n_subjects": 0.0,
        "baseline_960_n_valid_person_days": 57080,
        "delta_vs_960_n_valid_person_days": 0,
        "pct_delta_vs_960_n_valid_person_days": 0.0,
        "baseline_960_n_subjects_2011_2012_G": 4284,
        "delta_vs_960_n_subjects_2011_2012_G": 0,
        "pct_delta_vs_960_n_subjects_2011_2012_G": 0.0,
        "baseline_960_n_subjects_2013_2014_H": 4362,
        "delta_vs_960_n_subjects_2013_2014_H": 0,
        "pct_delta_vs_960_n_subjects_2013_2014_H": 0.0,
    }


def existing_rows(output_dir: Path, filename: str) -> list[dict[str, Any]]:
    path = output_dir / filename
    return read_csv(path) if path.is_file() else []


def resume_completed_thresholds(output_dir: Path, checkpoints: Path, resume: bool) -> set[int]:
    tables = {
        PAIR_FILE: existing_rows(output_dir, PAIR_FILE),
        FIXED_FILE: existing_rows(output_dir, FIXED_FILE),
        ATTAINMENT_FILE: existing_rows(output_dir, ATTAINMENT_FILE),
        TRANSPORT_FILE: existing_rows(output_dir, TRANSPORT_FILE),
        COHORT_FILE: existing_rows(output_dir, COHORT_FILE),
    }
    any_table = any(tables.values())
    if any_table and not resume:
        raise Blocked("Partial output exists without --resume; refusing to overwrite.")
    if not any_table:
        return set()
    observed_tags: set[int] = set()
    for name, rows in tables.items():
        if name == COHORT_FILE:
            tags = {int(float(row["wear_threshold"])) for row in rows if int(float(row["wear_threshold"])) != BASELINE_THRESHOLD}
        else:
            tags = {int(float(row["wear_threshold"])) for row in rows}
        if not tags.issubset(set(EXPECTED_THRESHOLDS)):
            raise Blocked(f"Unapproved threshold tag in resumable {name}: {sorted(tags)}")
        observed_tags |= tags
    completed: set[int] = set()
    expected_counts = {PAIR_FILE: 42, FIXED_FILE: 252, ATTAINMENT_FILE: 42, TRANSPORT_FILE: 84, COHORT_FILE: 1}
    for threshold in sorted(observed_tags):
        checkpoint = checkpoints / f"threshold_{threshold}.json"
        if not checkpoint.is_file():
            raise Blocked(f"Threshold {threshold} has aggregate rows but no checkpoint; cannot safely resume.")
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS" or int(payload.get("wear_threshold", -1)) != threshold:
            raise Blocked(f"Invalid checkpoint for threshold {threshold}.")
        for name, required in expected_counts.items():
            rows = tables[name]
            count = sum(1 for row in rows if int(float(row["wear_threshold"])) == threshold)
            if count != required:
                raise Blocked(f"Resumable {name} has {count}, not {required}, rows for threshold {threshold}.")
        completed.add(threshold)
    return completed


def write_partial_outputs(output_dir: Path, cohort_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]], attainment_rows: list[dict[str, Any]], transport_rows: list[dict[str, Any]]) -> None:
    atomic_csv(output_dir / COHORT_FILE, cohort_rows)
    atomic_csv(output_dir / PAIR_FILE, pair_rows)
    atomic_csv(output_dir / FIXED_FILE, fixed_rows)
    atomic_csv(output_dir / ATTAINMENT_FILE, attainment_rows)
    atomic_csv(output_dir / TRANSPORT_FILE, transport_rows)


def run_one_threshold(threshold: int, pipeline: dict[str, Any], raw_dir: Path, baseline: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage2 = pipeline["stage2"]
    support = pipeline["support"]
    cycle = pipeline["cycle"]
    log(f"building threshold {threshold} cohort")
    cohort = build_parameterized_cohort(pipeline, raw_dir, threshold)
    try:
        cycles, _strata, _psus, weights, _design_audit = cycle.align_design(cohort, raw_dir)
        if set(np.unique(cycles).tolist()) != set(EXPECTED_CYCLES):
            raise Blocked(f"Unexpected cycle labels at threshold {threshold}.")
        cycle_idx = {label: np.flatnonzero(cycles == label) for label in EXPECTED_CYCLES}
        for label, indices in cycle_idx.items():
            if len(indices) < cycle.FOLDS:
                raise Blocked(f"Threshold {threshold} cycle {label} has too few subjects for frozen five-fold OOF.")
        pairs = list(itertools.permutations(stage2.ALGORITHMS, 2))
        if set(pairs) != expected_pairs() or len(pairs) != 42:
            raise Blocked("Frozen directed-pair definition is incomplete.")
        all_idx = np.arange(cohort.subjects_n)
        log(f"threshold {threshold}: 42 continuous pair point estimates")
        pair_rows: list[dict[str, Any]] = []
        for source, target in pairs:
            computation = support.compute_pair(cohort, all_idx, source, target, support.STAGE2_RANDOM_SEED, keep_detail=False)
            if not math.isfinite(float(computation.mae_per_target_iqr)):
                raise Blocked(f"Threshold {threshold} has non-estimable critical E for {source}->{target}.")
            pair_rows.append(pair_metric_row(threshold, computation, baseline, cohort, all_idx, support))
        if len(pair_rows) != 42:
            raise Blocked(f"Threshold {threshold} does not contain all 42 continuous pairs.")
        log(f"threshold {threshold}: fixed-target grouped OOF point estimates")
        fixed_raw: list[dict[str, Any]] = []
        for source, target in pairs:
            fixed_raw.extend(cycle.threshold_rows_for_pair(cohort, all_idx, weights, source, target, cycle.CROSSWALK_SEED))
        if len(fixed_raw) != 252:
            raise Blocked(f"Threshold {threshold} fixed-target output is not 252 rows.")
        fixed_rows = enrich_fixed_rows(threshold, fixed_raw, baseline)
        attainment_rows = algorithm_attainment_rows(threshold, fixed_rows, baseline)
        log(f"threshold {threshold}: 84 cross-cycle transport point estimates")
        transport_raw: list[dict[str, Any]] = []
        for train_cycle, test_cycle in ((EXPECTED_CYCLES[0], EXPECTED_CYCLES[1]), (EXPECTED_CYCLES[1], EXPECTED_CYCLES[0])):
            for source, target in pairs:
                transport_raw.append(cycle.cross_cycle_row(cohort, cycle_idx[train_cycle], cycle_idx[test_cycle], source, target, train_cycle, test_cycle, cycle.CROSSWALK_SEED))
        if len(transport_raw) != 84:
            raise Blocked(f"Threshold {threshold} transport output is not 84 rows.")
        return cohort_row(threshold, cohort, cycles), pair_rows, fixed_rows, attainment_rows, enrich_transport_rows(threshold, transport_raw, baseline)
    finally:
        del cohort


def quantiles(values: list[float]) -> dict[str, float]:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return {"median": math.nan, "q25": math.nan, "q75": math.nan, "minimum": math.nan, "maximum": math.nan}
    return {
        "median": float(np.median(clean)),
        "q25": float(np.quantile(clean, 0.25)),
        "q75": float(np.quantile(clean, 0.75)),
        "minimum": float(np.min(clean)),
        "maximum": float(np.max(clean)),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(current: list[float], baseline: list[float]) -> float:
    pairs = [(left, right) for left, right in zip(current, baseline) if math.isfinite(left) and math.isfinite(right)]
    if len(pairs) < 2:
        return math.nan
    left, right = np.asarray([item[0] for item in pairs]), np.asarray([item[1] for item in pairs])
    left_rank, right_rank = average_ranks(left), average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def comparison_and_report(cohort_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]], attainment_rows: list[dict[str, Any]], transport_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    comparisons: list[dict[str, Any]] = []
    lines = [
        "# 有效日佩戴时长阈值敏感性分析报告",
        "",
        "- 状态：PASS（仅表示批准的三个门槛点估计、结构验证和受保护文件完整性均通过；不设或推断科学 PASS/FAIL 阈值）。",
        "- 960 分钟：既有冻结主分析基线；本次仅完成参数化队列一致性校验，未重算其 E/H、固定阈值、transport 或 bootstrap。",
        "- 新门槛：600、720 分钟为既有宽松敏感性点；1296 分钟为 post hoc strict stress test，均不替代 960 分钟主分析。",
        "- 未运行 bootstrap 或置信区间重算；所有表仅含聚合结果。",
        "- E、固定阈值 OOF 和跨周期比较保留五折 subject-grouped OOF；五折不用于共同覆盖曲线集成。",
        "- 共同覆盖 H 使用全数据合格子群保序曲线，并在共同覆盖区间内使用 101 个等距评价点。",
        "",
        "## 队列规模",
        "",
        "| valid-day 门槛 | N | 有效人日 | 相对 960 N 变化 | 相对 960 有效人日变化 | 2011–2012 G / 2013–2014 H |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    cohort_by_threshold = {int(row["wear_threshold"]): row for row in cohort_rows}
    for threshold in EXPECTED_THRESHOLDS:
        row = cohort_by_threshold[threshold]
        lines.append(
            f"| {threshold} | {int(float(row['n_subjects'])):,} | {int(float(row['n_valid_person_days'])):,} | "
            f"{float(row['pct_delta_vs_960_n_subjects']):+.2f}% | {float(row['pct_delta_vs_960_n_valid_person_days']):+.2f}% | "
            f"{int(float(row['n_subjects_2011_2012_G'])):,} / {int(float(row['n_subjects_2013_2014_H'])):,} |"
        )

    metric_specs = (
        ("E", "grouped_oof_mae_per_target_iqr", "baseline_960_grouped_oof_mae_per_target_iqr"),
        ("complete-sample P05-P95 H", "full_range_h_per_target_iqr", "baseline_960_full_range_h_per_target_iqr"),
        ("common-support H (101-point equally spaced grid)", "common_support_linear_h_per_target_iqr", "baseline_960_common_support_linear_h_per_target_iqr"),
    )
    lines.extend(["", "## 连续换算（42 个有向 pair）相对 960", "", "| 指标 | 门槛 | 中位数 [IQR] | 范围 | Spearman 排序相关 | 绝对变化中位数 / 最大值 | NA |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label, current_column, baseline_column in metric_specs:
        for threshold in EXPECTED_THRESHOLDS:
            rows = [row for row in pair_rows if int(row["wear_threshold"]) == threshold]
            current = [as_float(row[current_column]) for row in rows]
            prior = [as_float(row[baseline_column]) for row in rows]
            changes = [abs(left - right) for left, right in zip(current, prior) if math.isfinite(left) and math.isfinite(right)]
            stat = quantiles(current)
            change = quantiles(changes)
            na_n = sum(not math.isfinite(value) for value in current)
            comparisons.append({"wear_threshold": threshold, "section": "continuous_summary", "metric": label, "median": numeric_or_na(stat["median"]), "q25": numeric_or_na(stat["q25"]), "q75": numeric_or_na(stat["q75"]), "minimum": numeric_or_na(stat["minimum"]), "maximum": numeric_or_na(stat["maximum"]), "spearman_vs_960": numeric_or_na(spearman(current, prior)), "median_absolute_change_vs_960": numeric_or_na(change["median"]), "maximum_absolute_change_vs_960": numeric_or_na(change["maximum"]), "na_n": na_n})
            lines.append(f"| {label} | {threshold} | {fmt(stat['median'])} [{fmt(stat['q25'])}, {fmt(stat['q75'])}] | {fmt(stat['minimum'])}–{fmt(stat['maximum'])} | {fmt(spearman(current, prior))} | {fmt(change['median'])} / {fmt(change['maximum'])} | {na_n} |")
            changed = []
            for row in rows:
                current_value, baseline_value = as_float(row[current_column]), as_float(row[baseline_column])
                if math.isfinite(current_value) and math.isfinite(baseline_value):
                    changed.append((abs(current_value - baseline_value), row))
            for rank, (absolute_change, row) in enumerate(sorted(changed, key=lambda item: item[0], reverse=True)[:5], start=1):
                comparisons.append({"wear_threshold": threshold, "section": "continuous_largest_absolute_change", "metric": label, "rank": rank, "source_algorithm": row["source_algorithm"], "target_algorithm": row["target_algorithm"], "current": as_float(row[current_column]), "baseline_960": as_float(row[baseline_column]), "absolute_change": absolute_change})

    lines.extend(["", "### 连续指标中相对 960 绝对变化最大的 5 个方向（仅供人工定位）", ""])
    for label, current_column, baseline_column in metric_specs:
        lines.extend([f"#### {label}", "", "| 门槛 | rank | source→target | 当前 | 960 | 绝对变化 |", "|---:|---:|---|---:|---:|---:|"])
        top_rows = [row for row in comparisons if row.get("section") == "continuous_largest_absolute_change" and row.get("metric") == label]
        for row in sorted(top_rows, key=lambda item: (int(item["wear_threshold"]), int(item["rank"]))):
            lines.append(f"| {row['wear_threshold']} | {row['rank']} | {row['source_algorithm']}→{row['target_algorithm']} | {fmt(as_float(row['current']))} | {fmt(as_float(row['baseline_960']))} | {fmt(as_float(row['absolute_change']))} |")
        lines.append("")

    lines.extend(["## 算法原生固定目标达标率", "", "| 门槛 | target steps | weighting | 七算法达标率范围 | 相对 960 最大绝对变化（百分点） |", "|---:|---:|---|---:|---:"])
    for threshold in EXPECTED_THRESHOLDS:
        for target_steps in (7000.0, 8000.0, 10000.0):
            for weighting in ("unweighted_primary", "WTMEC4YR_weighted_sensitivity"):
                rows = [row for row in attainment_rows if int(row["wear_threshold"]) == threshold and float(row["threshold_target_steps"]) == target_steps and row["analysis_weighting"] == weighting]
                values = [as_float(row["observed_target_attainment_pct"]) for row in rows]
                deltas = [abs(as_float(row["delta_vs_960_observed_target_attainment_pct"])) for row in rows]
                comparisons.append({"wear_threshold": threshold, "section": "attainment_summary", "metric": "observed_target_attainment_pct", "target_steps": target_steps, "analysis_weighting": weighting, "minimum": min(values), "maximum": max(values), "maximum_absolute_change_vs_960": max(deltas)})
                lines.append(f"| {threshold} | {int(target_steps):,} | {weighting} | {fmt(min(values), 1)}–{fmt(max(values), 1)}% | {fmt(max(deltas), 1)} |")

    lines.extend(["", "## 固定目标 OOF discordance", "", "仅汇总当前与 960 均为非退化 target 的方向；退化状态改变的方向单列计数，未强行比较不同集合。", "", "| 门槛 | target steps | weighting | 非退化方向 n | discordance 中位数 [范围] (%) | 相对 960 绝对变化中位数 / 最大值（百分点） | 退化状态改变 n |", "|---:|---:|---|---:|---:|---:|---:|"])
    for threshold in EXPECTED_THRESHOLDS:
        for target_steps in (7000.0, 8000.0, 10000.0):
            for weighting in ("unweighted_primary", "WTMEC4YR_weighted_sensitivity"):
                rows = [row for row in fixed_rows if int(row["wear_threshold"]) == threshold and float(row["threshold_target_steps"]) == target_steps and row["analysis_weighting"] == weighting]
                comparable = [row for row in rows if not bool_from_csv(row["target_threshold_degenerate"]) and not bool_from_csv(row["baseline_960_target_threshold_degenerate"])]
                values = [as_float(row["discordant_reclassification_pct"]) for row in comparable]
                deltas = [abs(as_float(row["delta_vs_960_discordant_reclassification_pct"])) for row in comparable]
                stat, change = quantiles(values), quantiles(deltas)
                changed_status = sum(bool_from_csv(row["target_threshold_degenerate"]) != bool_from_csv(row["baseline_960_target_threshold_degenerate"]) for row in rows)
                comparisons.append({"wear_threshold": threshold, "section": "discordance_summary", "metric": "discordant_reclassification_pct", "target_steps": target_steps, "analysis_weighting": weighting, "n_comparable_nondegenerate": len(comparable), "median": numeric_or_na(stat["median"]), "minimum": numeric_or_na(stat["minimum"]), "maximum": numeric_or_na(stat["maximum"]), "median_absolute_change_vs_960": numeric_or_na(change["median"]), "maximum_absolute_change_vs_960": numeric_or_na(change["maximum"]), "degenerate_status_changed_n": changed_status})
                lines.append(f"| {threshold} | {int(target_steps):,} | {weighting} | {len(comparable)} | {fmt(stat['median'], 2)} [{fmt(stat['minimum'], 2)}, {fmt(stat['maximum'], 2)}] | {fmt(change['median'], 2)} / {fmt(change['maximum'], 2)} | {changed_status} |")

    lines.extend(["", "## 跨周期 transport penalty（84 个方向）", "", "| 门槛 | penalty MAE/IQR 中位数 [范围] | 相对 960 绝对变化中位数 / 最大值 | Spearman 排序相关 |", "|---:|---:|---:|---:|"])
    for threshold in EXPECTED_THRESHOLDS:
        rows = [row for row in transport_rows if int(row["wear_threshold"]) == threshold]
        values = [as_float(row["transport_penalty_mae_per_target_iqr"]) for row in rows]
        prior = [as_float(row["baseline_960_transport_penalty_mae_per_target_iqr"]) for row in rows]
        deltas = [abs(left - right) for left, right in zip(values, prior) if math.isfinite(left) and math.isfinite(right)]
        stat, change = quantiles(values), quantiles(deltas)
        rho = spearman(values, prior)
        comparisons.append({"wear_threshold": threshold, "section": "transport_penalty_summary", "metric": "transport_penalty_mae_per_target_iqr", "median": numeric_or_na(stat["median"]), "minimum": numeric_or_na(stat["minimum"]), "maximum": numeric_or_na(stat["maximum"]), "median_absolute_change_vs_960": numeric_or_na(change["median"]), "maximum_absolute_change_vs_960": numeric_or_na(change["maximum"]), "spearman_vs_960": numeric_or_na(rho)})
        lines.append(f"| {threshold} | {fmt(stat['median'])} [{fmt(stat['minimum'])}, {fmt(stat['maximum'])}] | {fmt(change['median'])} / {fmt(change['maximum'])} | {fmt(rho)} |")

    pair_na = {threshold: sum(not finite(row["full_range_h_per_target_iqr"]) or not finite(row["common_support_linear_h_per_target_iqr"]) for row in pair_rows if int(row["wear_threshold"]) == threshold) for threshold in EXPECTED_THRESHOLDS}
    lines.extend(["", "## 可估计性与解释边界", "", f"- 任一 H 指标为 NA 的 pair 数：600={pair_na[600]}，720={pair_na[720]}，1296={pair_na[1296]}；对应机器可读原因位于 pair metrics 表中。", "- 本报告只描述实际点估计的范围、变化和排序一致性；未计算 p 值、置信区间或预设可接受性界限。", "- population attainment 对齐不等同于 individual conversion；结果不指定任何算法为真实步数或金标准。", ""])
    return comparisons, "\n".join(lines)


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def final_validate(output_dir: Path) -> dict[str, Any]:
    cohort_rows = read_csv(output_dir / COHORT_FILE)
    pair_rows = read_csv(output_dir / PAIR_FILE)
    fixed_rows = read_csv(output_dir / FIXED_FILE)
    attainment_rows = read_csv(output_dir / ATTAINMENT_FILE)
    transport_rows = read_csv(output_dir / TRANSPORT_FILE)
    if len(cohort_rows) != 4 or {int(float(row["wear_threshold"])) for row in cohort_rows} != {600, 720, 960, 1296}:
        raise Blocked("Final cohort summary does not contain exactly 3 new thresholds plus 960 baseline.")
    required = ((pair_rows, 126, PAIR_FILE), (fixed_rows, 756, FIXED_FILE), (attainment_rows, 126, ATTAINMENT_FILE), (transport_rows, 252, TRANSPORT_FILE))
    for rows, expected_n, label in required:
        if len(rows) != expected_n or {int(float(row["wear_threshold"])) for row in rows} != set(EXPECTED_THRESHOLDS):
            raise Blocked(f"Final {label} count or threshold set is invalid.")
    ensure_unique(pair_rows, ("wear_threshold", "source_algorithm", "target_algorithm"), PAIR_FILE)
    ensure_unique(fixed_rows, ("wear_threshold", "source_algorithm", "target_algorithm", "threshold_target_steps", "analysis_weighting"), FIXED_FILE)
    ensure_unique(attainment_rows, ("wear_threshold", "target_algorithm", "threshold_target_steps", "analysis_weighting"), ATTAINMENT_FILE)
    ensure_unique(transport_rows, ("wear_threshold", "source_algorithm", "target_algorithm", "train_cycle", "test_cycle"), TRANSPORT_FILE)
    for rows, label in ((pair_rows, PAIR_FILE), (fixed_rows, FIXED_FILE), (attainment_rows, ATTAINMENT_FILE), (transport_rows, TRANSPORT_FILE)):
        algorithms = {str(row.get("source_algorithm", row.get("target_algorithm", ""))) for row in rows}
        target_algorithms = {str(row.get("target_algorithm", "")) for row in rows}
        if target_algorithms != set(EXPECTED_ALGORITHMS) or (algorithms and not algorithms.issubset(set(EXPECTED_ALGORITHMS))):
            raise Blocked(f"Unexpected algorithm set in {label}.")
        if any(row.get("source_algorithm") == row.get("target_algorithm") for row in rows if "source_algorithm" in row):
            raise Blocked(f"Self-pair found in {label}.")
        forbidden = {field.lower() for row in rows for field in row} & {"seqn", "paxdaym", "fold_id", "subject_id"}
        if forbidden:
            raise Blocked(f"Individual-level key field leaked to {label}: {sorted(forbidden)}")
    return {
        "cohort_rows": len(cohort_rows),
        "pair_rows": len(pair_rows),
        "fixed_rows": len(fixed_rows),
        "attainment_rows": len(attainment_rows),
        "transport_rows": len(transport_rows),
        "aggregate_only_output": True,
        "bootstrap_recomputed": False,
    }


def main() -> None:
    args = parse_args()
    check_arguments(args)
    raw_dir = args.raw_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    started = datetime.now(timezone.utc)
    log("release preflight")
    assert_required_raw_files(raw_dir)
    baseline = baseline_tables(baseline_dir)
    prepare_output_directory(args.output_dir, args.resume)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    pipeline = load_pipeline()
    log("960 light cohort consistency check only")
    cohort_960 = build_parameterized_cohort(pipeline, raw_dir, BASELINE_THRESHOLD)
    if (int(cohort_960.subjects_n), int(cohort_960.valid_days_n)) != (8646, 57080):
        raise Blocked(f"960 cohort consistency gate failed: got {cohort_960.subjects_n}/{cohort_960.valid_days_n}, expected 8646/57080.")
    del cohort_960
    completed = resume_completed_thresholds(args.output_dir, checkpoints, args.resume)
    cohort_rows = existing_rows(args.output_dir, COHORT_FILE) or [baseline_cohort_row()]
    pair_rows = existing_rows(args.output_dir, PAIR_FILE)
    fixed_rows = existing_rows(args.output_dir, FIXED_FILE)
    attainment_rows = existing_rows(args.output_dir, ATTAINMENT_FILE)
    transport_rows = existing_rows(args.output_dir, TRANSPORT_FILE)
    for threshold in EXPECTED_THRESHOLDS:
        if threshold in completed:
            log(f"threshold {threshold}: checkpoint verified; not rerunning")
            continue
        cohort_new, pairs_new, fixed_new, attainment_new, transport_new = run_one_threshold(threshold, pipeline, raw_dir, baseline)
        cohort_rows = [row for row in cohort_rows if int(float(row["wear_threshold"])) != threshold] + [cohort_new]
        pair_rows = [row for row in pair_rows if int(float(row["wear_threshold"])) != threshold] + pairs_new
        fixed_rows = [row for row in fixed_rows if int(float(row["wear_threshold"])) != threshold] + fixed_new
        attainment_rows = [row for row in attainment_rows if int(float(row["wear_threshold"])) != threshold] + attainment_new
        transport_rows = [row for row in transport_rows if int(float(row["wear_threshold"])) != threshold] + transport_new
        write_partial_outputs(args.output_dir, cohort_rows, pair_rows, fixed_rows, attainment_rows, transport_rows)
        checkpoint = {
            "status": "PASS",
            "wear_threshold": threshold,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "aggregate_row_counts_for_threshold": {"pair_metrics": 42, "fixed_threshold_metrics": 252, "algorithm_attainment": 42, "cross_cycle_transport": 84},
            "bootstrap_recomputed": False,
            "individual_level_output_written": False,
        }
        atomic_json(checkpoints / f"threshold_{threshold}.json", checkpoint)
        log(f"threshold {threshold}: aggregate outputs and checkpoint written")
    comparisons, report = comparison_and_report(cohort_rows, pair_rows, fixed_rows, attainment_rows, transport_rows)
    atomic_csv(args.output_dir / COMPARISON_FILE, comparisons)
    atomic_text(args.output_dir / REPORT_FILE, report)
    validation = final_validate(args.output_dir)
    run_record = {
        "status": "PASS",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "thresholds_completed": list(EXPECTED_THRESHOLDS),
        "baseline_threshold": BASELINE_THRESHOLD,
        "baseline_light_cohort_check": {"n_subjects": 8646, "n_valid_person_days": 57080},
        "oof_definition": "five-fold subject-grouped OOF retained for E, fixed-threshold, and cross-cycle evaluation; not used for common-support curve construction",
        "common_support_definition": "common_support_linear",
        "common_support_curve_estimator": "full-data eligible subgroup isotonic; same fitted subgroup curve as complete-sample H",
        "common_support_grid": "101-point equally spaced grid within the common-support interval",
        "bootstrap_recomputed": False,
        "manuscript_modified": False,
        "source_project_modified": False,
        "individual_level_output_written": False,
        "raw_inputs_verified_by_parent_runner": True,
        "baseline_generated_in_same_run": True,
        "final_validation": validation,
    }
    atomic_json(args.output_dir / RUN_FILE, run_record)
    log("PASS: all three approved thresholds and final hard validation completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
