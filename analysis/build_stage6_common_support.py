#!/usr/bin/env python3
"""Stage 6 continuous common-support subgroup stability.

This aggregate-only analysis rebuilds the D9-valid all-7 adult cohort in
memory. It evaluates complete-sample H and common-support H with the same
full-resample subgroup isotonic models, changing only the evaluation grid for
the common-support estimate. No participant, person-day, OOF-prediction, or
bootstrap-replicate table is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from build_mvp_crosswalk_stability import (
    FOLDS,
    HEADLINE_METHOD,
    MIN_SUBGROUP_CURVE_N,
    RANDOM_SEED as STAGE2_RANDOM_SEED,
    fit_crosswalk,
)
from build_mvp_exposure_qc import write_csv
from build_stage2_all7_crosswalk_matrix import ALGORITHMS
from build_stage2c_all7_subgroup_stability import (
    DailyCohort,
    build_daily_cohort,
    preferred_group_order,
    safe_divide,
)
from build_stage4_crosswalk_uncertainty_ci import align_design, design_bootstrap_indices


DEFAULT_BOOTSTRAP_REPS = 300
RANDOM_SEED_STAGE6 = 20260715
# Common-support H uses the same eligible subgroups as the original-grid H.
# Keep this name for compatibility with downstream modules.
MIN_COMMON_SUPPORT_LEVEL_N = MIN_SUBGROUP_CURVE_N
GRID_N = 101
CI_LOW_Q = 2.5
CI_HIGH_Q = 97.5
TIER1_MAE_MAX = 0.20
TIER1_SPREAD_MAX = 0.20
TIER2_MAE_MAX = 0.27
TIER2_SPREAD_MAX = 0.35

ORIGINAL_DEFINITION = "original_grid_stage3"
COMMON_LINEAR_DEFINITION = "common_support_linear"
DEFINITIONS = [
    ORIGINAL_DEFINITION,
    COMMON_LINEAR_DEFINITION,
]


@dataclass(frozen=True)
class PairKey:
    source: str
    target: str


@dataclass(frozen=True)
class Stage3Point:
    source: str
    target: str
    mae_per_target_iqr: float
    max_spread_per_target_iqr: float
    max_spread_group_type: str
    tier_final: str
    featured_main_flag: str
    featured_main_reason: str


@dataclass
class PairComputation:
    source: str
    target: str
    n: int
    target_iqr: float
    mae_per_target_iqr: float
    spread_by_definition: dict[str, float]
    spread_raw_by_definition: dict[str, float]
    driver_by_definition: dict[str, str]
    detail_rows: list[dict[str, object]]


def log(message: str) -> None:
    print(f"[stage6-common-support {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_STAGE6)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be >=1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >=1")
    return args


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def finite_percentile(values: list[float], probability: float) -> float:
    clean = np.asarray([value for value in values if finite(float(value))], dtype=float)
    if len(clean) == 0:
        return math.nan
    return float(np.percentile(clean, probability))


def normalize_tier(value: str) -> str:
    stripped = str(value).strip()
    for tier in ("tier_1", "tier_2", "tier_3"):
        if stripped.startswith(tier):
            return tier
    raise ValueError(f"Unrecognized tier value: {value!r}")


def tier_from_metrics(mae_per_iqr: float, spread_per_iqr: float) -> str:
    if not finite(mae_per_iqr) or not finite(spread_per_iqr):
        return "not_estimable"
    if mae_per_iqr <= TIER1_MAE_MAX and spread_per_iqr <= TIER1_SPREAD_MAX:
        return "tier_1"
    if mae_per_iqr <= TIER2_MAE_MAX and spread_per_iqr <= TIER2_SPREAD_MAX:
        return "tier_2"
    return "tier_3"


def load_stage3_points(out_dir: Path) -> dict[PairKey, Stage3Point]:
    path = out_dir / "tables" / "stage3_translatability_map.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Required locked Stage 3 point map not found: {path}. "
            "Pass the project outputs root as --out-dir."
        )
    points: dict[PairKey, Stage3Point] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_algorithm",
            "target_algorithm",
            "mae_per_target_iqr",
            "max_subgroup_p95_spread_over_target_iqr",
            "max_subgroup_p95_spread_group_type",
            "tier_final",
            "featured_main_flag",
            "featured_main_reason",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Stage 3 map is missing required columns: {sorted(missing)}")
        for row in reader:
            key = PairKey(row["source_algorithm"], row["target_algorithm"])
            if key in points:
                raise RuntimeError(f"Duplicate Stage 3 directed pair: {key}")
            points[key] = Stage3Point(
                source=key.source,
                target=key.target,
                mae_per_target_iqr=parse_float(row["mae_per_target_iqr"]),
                max_spread_per_target_iqr=parse_float(
                    row["max_subgroup_p95_spread_over_target_iqr"]
                ),
                max_spread_group_type=row["max_subgroup_p95_spread_group_type"],
                tier_final=normalize_tier(row["tier_final"]),
                featured_main_flag=row["featured_main_flag"],
                featured_main_reason=row["featured_main_reason"],
            )
    expected = {
        PairKey(source, target)
        for source in ALGORITHMS
        for target in ALGORITHMS
        if source != target
    }
    if set(points) != expected:
        raise RuntimeError(
            "Stage 3 directed-pair set mismatch: "
            f"expected={len(expected)}, observed={len(points)}, "
            f"missing={sorted(expected - set(points), key=lambda key: (key.source, key.target))}, "
            f"extra={sorted(set(points) - expected, key=lambda key: (key.source, key.target))}."
        )
    return points


def group_values_for(cohort: DailyCohort, sample_idx: np.ndarray, group_type: str) -> np.ndarray:
    if group_type == "age_group":
        return cohort.age_group[sample_idx]
    if group_type == "sex":
        return cohort.sex[sample_idx]
    if group_type == "bmi_group":
        return cohort.bmi_group[sample_idx]
    raise ValueError(f"Unknown group type: {group_type}")


def grouped_folds(subject_ids: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return row masks with every repeated copy of a subject kept in one fold."""

    ids = np.asarray(subject_ids, dtype=object)
    unique_subjects = np.asarray(sorted(set(ids.tolist())), dtype=object)
    n_splits = min(FOLDS, len(unique_subjects))
    if n_splits < 2:
        return []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_unique, test_unique in splitter.split(unique_subjects):
        train_subjects = set(unique_subjects[train_unique].tolist())
        test_subjects = set(unique_subjects[test_unique].tolist())
        train_mask = np.asarray([subject in train_subjects for subject in ids], dtype=bool)
        test_mask = np.asarray([subject in test_subjects for subject in ids], dtype=bool)
        folds.append((train_mask, test_mask))
    return folds


def grouped_oof_mae_per_iqr(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    seed: int,
) -> float:
    predictions = np.full(len(x), np.nan, dtype=float)
    folds = grouped_folds(subject_ids, seed)
    if not folds:
        return math.nan
    for train_mask, test_mask in folds:
        fitted = fit_crosswalk(HEADLINE_METHOD, x[train_mask], y[train_mask])
        predictions[test_mask] = fitted.predict(x[test_mask])
    if not np.all(np.isfinite(predictions)):
        return math.nan
    target_iqr = quantile(y, 0.75) - quantile(y, 0.25)
    mae = float(np.mean(np.abs(predictions - y)))
    return safe_divide(mae, target_iqr)


def eligible_levels(
    values: np.ndarray,
    group_type: str,
    minimum_n: int,
) -> list[tuple[str, np.ndarray, int]]:
    levels: list[tuple[str, np.ndarray, int]] = []
    for level in preferred_group_order(group_type):
        if level == "missing":
            continue
        mask = values == level
        n = int(np.sum(mask))
        if n >= minimum_n:
            levels.append((level, mask, n))
    return levels


def spread_stats(predictions: list[np.ndarray]) -> tuple[float, float]:
    if len(predictions) < 2:
        return math.nan, math.nan
    matrix = np.vstack(predictions)
    spread = np.max(matrix, axis=0) - np.min(matrix, axis=0)
    return float(np.percentile(spread, 95)), float(np.max(spread))


def detail_row(
    *,
    source: str,
    target: str,
    group_type: str,
    definition: str,
    estimator: str,
    levels: list[str],
    ns: list[int],
    level_p05: list[float],
    level_p95: list[float],
    original_low: float,
    original_high: float,
    common_low: float,
    common_high: float,
    grid_low: float,
    grid_high: float,
    pooled_common_n: int,
    target_iqr: float,
    p95_spread: float,
    max_spread: float,
) -> dict[str, object]:
    if definition == ORIGINAL_DEFINITION:
        grid_role = "complete_sample_reference"
        support_rule = "pooled source P05-P95; may extend beyond a subgroup's own support"
    else:
        grid_role = "common_support_linear"
        support_rule = "intersection of eligible subgroup source P05-P95; 101-point equally spaced grid"
    return {
        "exposure": "daily_steps",
        "source_algorithm": source,
        "target_algorithm": target,
        "method": HEADLINE_METHOD,
        "group_type": group_type,
        "grid_definition": definition,
        "grid_role": grid_role,
        "curve_estimator": estimator,
        "minimum_level_n": MIN_SUBGROUP_CURVE_N,
        "n_groups_compared": len(levels),
        "group_levels": "|".join(levels),
        "group_ns": "|".join(str(n) for n in ns),
        "group_source_p05": "|".join(f"{value:.12g}" for value in level_p05),
        "group_source_p95": "|".join(f"{value:.12g}" for value in level_p95),
        "original_source_p05": original_low,
        "original_source_p95": original_high,
        "common_support_low": common_low,
        "common_support_high": common_high,
        "common_support_width": common_high - common_low if finite(common_low) and finite(common_high) else math.nan,
        "grid_n": GRID_N,
        "grid_min": grid_low,
        "grid_max": grid_high,
        "pooled_observations_within_common_support": pooled_common_n,
        "target_iqr": target_iqr,
        "p95_between_subgroup_spread_predicted_target": p95_spread,
        "p95_between_subgroup_spread_predicted_target_over_target_iqr": safe_divide(p95_spread, target_iqr),
        "max_between_subgroup_spread_predicted_target": max_spread,
        "max_between_subgroup_spread_predicted_target_over_target_iqr": safe_divide(max_spread, target_iqr),
        "support_rule": support_rule,
    }


def compute_pair(
    cohort: DailyCohort,
    sample_idx: np.ndarray,
    source: str,
    target: str,
    seed: int,
    keep_detail: bool,
) -> PairComputation:
    x_all = cohort.daily_by_alg[source][sample_idx]
    y_all = cohort.daily_by_alg[target][sample_idx]
    ids_all = cohort.subject_ids[sample_idx]
    group_arrays = {
        group_type: group_values_for(cohort, sample_idx, group_type)
        for group_type in ("age_group", "sex", "bmi_group")
    }
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    x = np.asarray(x_all[valid], dtype=float)
    y = np.asarray(y_all[valid], dtype=float)
    subject_ids = np.asarray(ids_all[valid], dtype=object)
    groups = {group_type: values[valid] for group_type, values in group_arrays.items()}
    if len(x) < 10:
        return PairComputation(source, target, len(x), math.nan, math.nan, {}, {}, {}, [])

    target_iqr = quantile(y, 0.75) - quantile(y, 0.25)
    mae_per_iqr = grouped_oof_mae_per_iqr(x, y, subject_ids, seed)
    original_low = quantile(x, 0.05)
    original_high = quantile(x, 0.95)
    original_grid = np.linspace(original_low, original_high, GRID_N)

    metric_rows: list[dict[str, object]] = []
    spread_candidates: dict[str, list[tuple[str, float, float]]] = defaultdict(list)

    for group_type, group_values in groups.items():
        eligible_group_levels = eligible_levels(
            group_values,
            group_type,
            MIN_SUBGROUP_CURVE_N,
        )
        original_predictions: list[np.ndarray] = []
        fitted_subgroup_models = []
        for _level, level_mask, _n in eligible_group_levels:
            fitted = fit_crosswalk(HEADLINE_METHOD, x[level_mask], y[level_mask])
            fitted_subgroup_models.append(fitted)
            original_predictions.append(fitted.predict(original_grid))
        original_p95, original_max = spread_stats(original_predictions)

        levels = [level for level, _mask, _n in eligible_group_levels]
        ns = [n for _level, _mask, n in eligible_group_levels]
        level_p05 = [quantile(x[mask], 0.05) for _level, mask, _n in eligible_group_levels]
        level_p95 = [quantile(x[mask], 0.95) for _level, mask, _n in eligible_group_levels]
        common_low = max(level_p05) if level_p05 else math.nan
        common_high = min(level_p95) if level_p95 else math.nan
        common_valid = (
            len(eligible_group_levels) >= 2
            and finite(common_low)
            and finite(common_high)
            and common_low < common_high
        )

        linear_p95 = math.nan
        linear_max = math.nan
        linear_grid_low = math.nan
        linear_grid_high = math.nan
        pooled_common_n = 0
        if common_valid:
            eligible_mask = np.zeros(len(x), dtype=bool)
            for _level, level_mask, _n in eligible_group_levels:
                eligible_mask |= level_mask
            pooled_common_mask = eligible_mask & (x >= common_low) & (x <= common_high)
            pooled_common_x = x[pooled_common_mask]
            pooled_common_n = int(len(pooled_common_x))
            linear_grid = np.linspace(common_low, common_high, GRID_N)
            linear_grid_low = float(np.min(linear_grid))
            linear_grid_high = float(np.max(linear_grid))
            common_predictions_linear: list[np.ndarray] = []
            # The subgroup models above are fitted once on the full resample
            # and reused here; observed density inside a valid intersection
            # does not determine the 101 evaluation coordinates.
            for fitted in fitted_subgroup_models:
                common_predictions_linear.append(fitted.predict(linear_grid))
            linear_p95, linear_max = spread_stats(common_predictions_linear)

        original_level_names = [level for level, _mask, _n in eligible_group_levels]
        original_ns = [n for _level, _mask, n in eligible_group_levels]
        original_level_p05 = [quantile(x[mask], 0.05) for _level, mask, _n in eligible_group_levels]
        original_level_p95 = [quantile(x[mask], 0.95) for _level, mask, _n in eligible_group_levels]

        spread_candidates[ORIGINAL_DEFINITION].append((group_type, original_p95, original_max))
        spread_candidates[COMMON_LINEAR_DEFINITION].append((group_type, linear_p95, linear_max))

        if keep_detail:
            metric_rows.append(
                detail_row(
                    source=source,
                    target=target,
                    group_type=group_type,
                    definition=ORIGINAL_DEFINITION,
                    estimator="full-resample subgroup isotonic; complete-sample reference",
                    levels=original_level_names,
                    ns=original_ns,
                    level_p05=original_level_p05,
                    level_p95=original_level_p95,
                    original_low=original_low,
                    original_high=original_high,
                    common_low=common_low,
                    common_high=common_high,
                    grid_low=original_low,
                    grid_high=original_high,
                    pooled_common_n=pooled_common_n,
                    target_iqr=target_iqr,
                    p95_spread=original_p95,
                    max_spread=original_max,
                )
            )
            metric_rows.append(
                detail_row(
                    source=source,
                    target=target,
                    group_type=group_type,
                    definition=COMMON_LINEAR_DEFINITION,
                    estimator="full-resample subgroup isotonic; same fitted subgroup curve as original-grid H",
                    levels=levels,
                    ns=ns,
                    level_p05=level_p05,
                    level_p95=level_p95,
                    original_low=original_low,
                    original_high=original_high,
                    common_low=common_low,
                    common_high=common_high,
                    grid_low=linear_grid_low,
                    grid_high=linear_grid_high,
                    pooled_common_n=pooled_common_n,
                    target_iqr=target_iqr,
                    p95_spread=linear_p95,
                    max_spread=linear_max,
                )
            )

    spread_by_definition: dict[str, float] = {}
    spread_raw_by_definition: dict[str, float] = {}
    driver_by_definition: dict[str, str] = {}
    for definition in DEFINITIONS:
        candidates = [
            (group_type, p95_value, max_value)
            for group_type, p95_value, max_value in spread_candidates.get(definition, [])
            if finite(p95_value)
        ]
        if not candidates:
            spread_by_definition[definition] = math.nan
            spread_raw_by_definition[definition] = math.nan
            driver_by_definition[definition] = ""
            continue
        driver, p95_value, _max_value = max(candidates, key=lambda item: item[1])
        spread_raw_by_definition[definition] = p95_value
        spread_by_definition[definition] = safe_divide(p95_value, target_iqr)
        driver_by_definition[definition] = driver

    return PairComputation(
        source=source,
        target=target,
        n=len(x),
        target_iqr=target_iqr,
        mae_per_target_iqr=mae_per_iqr,
        spread_by_definition=spread_by_definition,
        spread_raw_by_definition=spread_raw_by_definition,
        driver_by_definition=driver_by_definition,
        detail_rows=metric_rows,
    )


def build_tier_sensitivity_row(
    computation: PairComputation,
    stage3: Stage3Point,
) -> dict[str, object]:
    original_recomputed_spread = computation.spread_by_definition[ORIGINAL_DEFINITION]
    linear_spread = computation.spread_by_definition[COMMON_LINEAR_DEFINITION]
    original_recomputed_tier = tier_from_metrics(
        stage3.mae_per_target_iqr,
        original_recomputed_spread,
    )
    linear_tier = tier_from_metrics(stage3.mae_per_target_iqr, linear_spread)
    return {
        "exposure": "daily_steps",
        "source_algorithm": computation.source,
        "target_algorithm": computation.target,
        "method": HEADLINE_METHOD,
        "n": computation.n,
        "target_iqr": computation.target_iqr,
        "stage3_mae_per_target_iqr_loaded": stage3.mae_per_target_iqr,
        "stage6_grouped_oof_mae_per_target_iqr_recomputed": computation.mae_per_target_iqr,
        "absolute_mae_normalized_diff_vs_stage3": abs(
            computation.mae_per_target_iqr - stage3.mae_per_target_iqr
        ),
        "stage3_original_grid_max_spread_per_target_iqr_loaded": stage3.max_spread_per_target_iqr,
        "stage3_original_grid_driver_loaded": stage3.max_spread_group_type,
        "stage3_tier_final_preserved": stage3.tier_final,
        "stage6_original_grid_max_spread_per_target_iqr_recomputed": original_recomputed_spread,
        "stage6_original_grid_driver_recomputed": computation.driver_by_definition[ORIGINAL_DEFINITION],
        "absolute_original_spread_normalized_diff_vs_stage3": abs(
            original_recomputed_spread - stage3.max_spread_per_target_iqr
        ),
        "stage6_original_grid_tier_recomputed": original_recomputed_tier,
        "common_support_linear_max_spread_per_target_iqr": linear_spread,
        "common_support_linear_driver": computation.driver_by_definition[COMMON_LINEAR_DEFINITION],
        "common_support_linear_tier": linear_tier,
        "common_support_linear_changed_vs_stage3": linear_tier != stage3.tier_final,
        "featured_main_flag_preserved": stage3.featured_main_flag,
        "featured_main_reason_preserved": stage3.featured_main_reason,
        "tier_rule_fixed": (
            "tier_1 if MAE/IQR<=0.20 and spread/IQR<=0.20; "
            "tier_2 if MAE/IQR<=0.27 and spread/IQR<=0.35; else tier_3"
        ),
        "interpretation": (
            "Stage 3 tier is preserved. The common-support tier uses the loaded "
            "Stage 3 MAE/IQR and the support-restricted linear-grid spread."
        ),
    }


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + (z * z) / trials
    center = (probability + (z * z) / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + (z * z) / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def probability_columns(successes: int, trials: int, prefix: str) -> dict[str, object]:
    low, high = wilson_interval(successes, trials)
    return {
        prefix: successes / trials if trials else math.nan,
        f"{prefix}_ci_low": low,
        f"{prefix}_ci_high": high,
    }


def build_bootstrap_rows(
    points: dict[PairKey, Stage3Point],
    point_results: dict[PairKey, PairComputation],
    boot_mae: dict[PairKey, list[float]],
    boot_spread: dict[tuple[PairKey, str], list[float]],
    boot_tiers: dict[tuple[PairKey, str], list[str]],
    requested_reps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(points, key=lambda item: (item.source, item.target)):
        stage3 = points[key]
        point_result = point_results[key]
        for definition in DEFINITIONS:
            tier_values = [
                tier for tier in boot_tiers[(key, definition)]
                if tier in {"tier_1", "tier_2", "tier_3"}
            ]
            trials = len(tier_values)
            counts = Counter(tier_values)
            modal_tier = max(
                ("tier_1", "tier_2", "tier_3"),
                key=lambda tier: (counts[tier], -int(tier[-1])),
            ) if trials else "not_estimable"
            modal_count = counts[modal_tier] if trials else 0
            row: dict[str, object] = {
                "exposure": "daily_steps",
                "source_algorithm": key.source,
                "target_algorithm": key.target,
                "method": HEADLINE_METHOD,
                "stability_definition": definition,
                "stage3_tier_final_preserved": stage3.tier_final,
                "stage3_mae_per_target_iqr_loaded": stage3.mae_per_target_iqr,
                "point_grouped_oof_mae_per_target_iqr": point_result.mae_per_target_iqr,
                "point_max_spread_per_target_iqr": point_result.spread_by_definition[definition],
                "point_spread_driver": point_result.driver_by_definition[definition],
                "point_tier_under_definition": (
                    stage3.tier_final
                    if definition == ORIGINAL_DEFINITION
                    else tier_from_metrics(
                        stage3.mae_per_target_iqr,
                        point_result.spread_by_definition[definition],
                    )
                ),
                "mae_per_target_iqr_ci_low": finite_percentile(boot_mae[key], CI_LOW_Q),
                "mae_per_target_iqr_ci_high": finite_percentile(boot_mae[key], CI_HIGH_Q),
                "max_spread_per_target_iqr_ci_low": finite_percentile(
                    boot_spread[(key, definition)], CI_LOW_Q
                ),
                "max_spread_per_target_iqr_ci_high": finite_percentile(
                    boot_spread[(key, definition)], CI_HIGH_Q
                ),
                "bootstrap_success": trials,
                "bootstrap_requested": requested_reps,
                "modal_tier": modal_tier,
                "modal_tier_probability": modal_count / trials if trials else math.nan,
            }
            modal_low, modal_high = wilson_interval(modal_count, trials)
            row["modal_tier_probability_ci_low"] = modal_low
            row["modal_tier_probability_ci_high"] = modal_high
            row.update(probability_columns(counts["tier_1"], trials, "p_tier_1"))
            row.update(probability_columns(counts["tier_2"], trials, "p_tier_2"))
            row.update(probability_columns(counts["tier_3"], trials, "p_tier_3"))
            row.update(
                probability_columns(
                    counts["tier_1"] + counts["tier_2"],
                    trials,
                    "p_convertible_tier_1_or_2",
                )
            )
            row["ci_method_continuous"] = "percentile PSU-within-stratum bootstrap (2.5th, 97.5th percentiles)"
            row["ci_method_probabilities"] = "Wilson 95% interval for aggregate bootstrap tier proportion"
            row["tier_cutpoints_fixed_within_bootstrap"] = True
            rows.append(row)
    return rows


def format_number(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def build_summary(
    path: Path,
    cohort: DailyCohort,
    tier_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    bootstrap_reps: int,
    original_mae_max_diff: float,
    original_spread_max_diff: float,
) -> None:
    def tier_counts(column: str) -> Counter[str]:
        return Counter(str(row[column]) for row in tier_rows)

    stage3_counts = tier_counts("stage3_tier_final_preserved")
    linear_counts = tier_counts("common_support_linear_tier")
    changed_linear = [
        row for row in tier_rows
        if bool(row["common_support_linear_changed_vs_stage3"])
    ]
    bootstrap_by_definition = {
        definition: [
            row for row in bootstrap_rows
            if row["stability_definition"] == definition
        ]
        for definition in DEFINITIONS
    }

    lines = [
        "# Stage 6 Common-Support and Tier-Uncertainty Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "- Status: continuous support validation/sensitivity add-on; no Stage 3 point tier or featured pair was overwritten.",
        f"- Cohort: {cohort.subjects_n} adults / {cohort.valid_days_n} valid person-days; 42 directed pairs.",
        f"- Bootstrap completed: {bootstrap_reps}/{DEFAULT_BOOTSTRAP_REPS} requested by the final analysis specification.",
        f"- Common support: intersection of every eligible nonmissing subgroup's source P05-P95; the eligible subgroups are the same as for original-grid H (n>={MIN_SUBGROUP_CURVE_N}).",
        "- Common-support curves: full-resample subgroup isotonic; the same fitted subgroup curve is used for original-grid H and common-support H, with a 101-point equally spaced grid inside the common-support interval.",
        "- Original-grid tier bootstrap uses full-resample subgroup isotonic curves on the pooled source P05-P95 grid.",
        "- Bootstrap: PSU sampled with replacement within SDMVSTRA; grouped OOF MAE/IQR and curve spread were recomputed; 0.20/0.27 and 0.20/0.35 cutpoints stayed fixed.",
        "- Guardrails: aggregate outputs only; no SEQN, person-day, minute, participant OOF prediction, or bootstrap-replicate table was written.",
        "",
        "## Point-Estimate Reproduction Audit",
        "",
        f"- Maximum absolute grouped-OOF MAE/IQR difference versus loaded Stage 3: {original_mae_max_diff:.12g}.",
        f"- Maximum absolute original-grid max-spread/IQR difference versus loaded Stage 3: {original_spread_max_diff:.12g}.",
        "- These are audit comparisons only; `tier_final` remains the loaded Stage 3 value.",
        "",
        "## Point Tier Sensitivity",
        "",
        "| definition | tier 1 | tier 2 | tier 3 |",
        "|---|---:|---:|---:|",
        f"| Stage 3 original (preserved) | {stage3_counts['tier_1']} | {stage3_counts['tier_2']} | {stage3_counts['tier_3']} |",
        f"| common support, linear headline | {linear_counts['tier_1']} | {linear_counts['tier_2']} | {linear_counts['tier_3']} |",
        "",
        f"- Pairs changing tier under the common-support linear grid: {len(changed_linear)}/42.",
        "",
        "| source -> target | Stage 3 | common linear | common linear max spread/IQR | driver |",
        "|---|---|---|---|---:|---|",
    ]
    for row in changed_linear:
        lines.append(
            f"| {row['source_algorithm']}->{row['target_algorithm']} | "
            f"{row['stage3_tier_final_preserved']} | {row['common_support_linear_tier']} | "
            f"{format_number(float(row['common_support_linear_max_spread_per_target_iqr']))} | "
            f"{row['common_support_linear_driver']} |"
        )

    lines.extend(
        [
            "",
            "## Bootstrap Tier Stability",
            "",
            "| definition | modal tier 1 | modal tier 2 | modal tier 3 | min modal probability | median P(convertible) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for definition in DEFINITIONS:
        rows = bootstrap_by_definition[definition]
        modal_counts = Counter(str(row["modal_tier"]) for row in rows)
        modal_probabilities = [float(row["modal_tier_probability"]) for row in rows]
        convertible = [float(row["p_convertible_tier_1_or_2"]) for row in rows]
        lines.append(
            f"| {definition} | {modal_counts['tier_1']} | {modal_counts['tier_2']} | "
            f"{modal_counts['tier_3']} | {min(modal_probabilities):.3f} | "
            f"{float(np.median(convertible)):.3f} |"
        )

    if bootstrap_reps != DEFAULT_BOOTSTRAP_REPS:
        lines.extend(
            [
                "",
                "## Smoke-Run Warning",
                "",
                f"- This run used {bootstrap_reps} replicates, not the locked 300. Its tier probabilities and CIs are execution checks only and must not enter the manuscript or reports.",
            ]
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `outputs/tables/stage6_common_support_subgroup_stability.csv`",
            "- `outputs/tables/stage6_common_support_tier_sensitivity.csv`",
            "- `outputs/tables/stage6_tier_bootstrap_stability.csv`",
            "- `outputs/tables/STAGE6_COMMON_SUPPORT_TIER_SUMMARY.md`",
            "- `outputs/logs/stage6_common_support_tier_run.json`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    out_tables = args.out_dir / "tables"
    out_logs = args.out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    log("loading locked Stage 3 point map")
    stage3_points = load_stage3_points(args.out_dir)
    log("rebuilding locked D9-valid all-7 adult daily-steps cohort")
    cohort = build_daily_cohort(args.raw_dir)
    if cohort.subjects_n != 8646 or cohort.valid_days_n != 57080:
        raise RuntimeError(
            f"Stage 6 cohort drift: expected 8646/57080, got "
            f"{cohort.subjects_n}/{cohort.valid_days_n}."
        )
    log("aligning DEMO survey-design variables")
    strata, psu, _weights, design_rows = align_design(cohort, args.raw_dir)

    pairs = [
        PairKey(source, target)
        for source in ALGORITHMS
        for target in ALGORITHMS
        if source != target
    ]
    all_indices = np.arange(cohort.subjects_n)
    detail_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    point_results: dict[PairKey, PairComputation] = {}

    log("computing 42-pair common-support point estimates")
    for pair_number, key in enumerate(pairs, start=1):
        computation = compute_pair(
            cohort,
            all_indices,
            key.source,
            key.target,
            STAGE2_RANDOM_SEED,
            keep_detail=True,
        )
        point_results[key] = computation
        detail_rows.extend(computation.detail_rows)
        tier_rows.append(build_tier_sensitivity_row(computation, stage3_points[key]))
        if pair_number == 1 or pair_number == len(pairs) or pair_number % 7 == 0:
            log(f"point pair {pair_number}/{len(pairs)} complete")

    original_mae_diffs = [
        float(row["absolute_mae_normalized_diff_vs_stage3"])
        for row in tier_rows
    ]
    original_spread_diffs = [
        float(row["absolute_original_spread_normalized_diff_vs_stage3"])
        for row in tier_rows
    ]
    max_mae_diff = max(original_mae_diffs)
    max_spread_diff = max(original_spread_diffs)
    if max_mae_diff > 1e-10 or max_spread_diff > 1e-10:
        raise RuntimeError(
            "Stage 3 point reproduction failed: "
            f"max MAE/IQR diff={max_mae_diff:.12g}, "
            f"max original spread/IQR diff={max_spread_diff:.12g}."
        )
    if Counter(row["stage3_tier_final_preserved"] for row in tier_rows) != Counter(
        {"tier_1": 4, "tier_2": 15, "tier_3": 23}
    ):
        raise RuntimeError("Loaded Stage 3 tier count drift; expected 4/15/23.")

    rng = np.random.default_rng(args.seed)
    boot_mae: dict[PairKey, list[float]] = defaultdict(list)
    boot_spread: dict[tuple[PairKey, str], list[float]] = defaultdict(list)
    boot_tiers: dict[tuple[PairKey, str], list[str]] = defaultdict(list)
    log(f"starting {args.bootstrap_reps} PSU-within-stratum bootstrap replicates")
    for replicate in range(1, args.bootstrap_reps + 1):
        sample_idx = design_bootstrap_indices(strata, psu, rng)
        replicate_seed = args.seed + replicate
        for key in pairs:
            computation = compute_pair(
                cohort,
                sample_idx,
                key.source,
                key.target,
                replicate_seed,
                keep_detail=False,
            )
            if finite(computation.mae_per_target_iqr):
                boot_mae[key].append(computation.mae_per_target_iqr)
            for definition in DEFINITIONS:
                spread = computation.spread_by_definition.get(definition, math.nan)
                if not finite(computation.mae_per_target_iqr) or not finite(spread):
                    continue
                boot_spread[(key, definition)].append(spread)
                boot_tiers[(key, definition)].append(
                    tier_from_metrics(computation.mae_per_target_iqr, spread)
                )
        if (
            replicate == 1
            or replicate == args.bootstrap_reps
            or replicate % args.progress_every == 0
        ):
            log(f"bootstrap replicate {replicate}/{args.bootstrap_reps} complete")

    bootstrap_rows = build_bootstrap_rows(
        stage3_points,
        point_results,
        boot_mae,
        boot_spread,
        boot_tiers,
        args.bootstrap_reps,
    )

    detail_fieldnames = [
        "exposure",
        "source_algorithm",
        "target_algorithm",
        "method",
        "group_type",
        "grid_definition",
        "grid_role",
        "curve_estimator",
        "minimum_level_n",
        "n_groups_compared",
        "group_levels",
        "group_ns",
        "group_source_p05",
        "group_source_p95",
        "original_source_p05",
        "original_source_p95",
        "common_support_low",
        "common_support_high",
        "common_support_width",
        "grid_n",
        "grid_min",
        "grid_max",
        "pooled_observations_within_common_support",
        "target_iqr",
        "p95_between_subgroup_spread_predicted_target",
        "p95_between_subgroup_spread_predicted_target_over_target_iqr",
        "max_between_subgroup_spread_predicted_target",
        "max_between_subgroup_spread_predicted_target_over_target_iqr",
        "support_rule",
    ]
    tier_fieldnames = list(tier_rows[0].keys())
    bootstrap_fieldnames = list(bootstrap_rows[0].keys())
    write_csv(
        out_tables / "stage6_common_support_subgroup_stability.csv",
        detail_rows,
        detail_fieldnames,
    )
    write_csv(
        out_tables / "stage6_common_support_tier_sensitivity.csv",
        tier_rows,
        tier_fieldnames,
    )
    write_csv(
        out_tables / "stage6_tier_bootstrap_stability.csv",
        bootstrap_rows,
        bootstrap_fieldnames,
    )
    build_summary(
        out_tables / "STAGE6_COMMON_SUPPORT_TIER_SUMMARY.md",
        cohort,
        tier_rows,
        bootstrap_rows,
        args.bootstrap_reps,
        max_mae_diff,
        max_spread_diff,
    )

    design_overall = next(row for row in design_rows if row["scope"] == "design_overall")
    finished = datetime.now(timezone.utc)
    run_log = {
        "script": "scripts/python/build_stage6_common_support_tier.py",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "analysis_role": "post-freeze validation/sensitivity; does not overwrite Stage 3",
        "algorithms": ALGORITHMS,
        "directed_pairs": len(pairs),
        "cohort_subjects": cohort.subjects_n,
        "cohort_valid_days": cohort.valid_days_n,
        "design_strata": len(np.unique(strata)),
        "design_stratum_psu_cells": design_overall["n_psu"],
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_full_spec_complete": args.bootstrap_reps == DEFAULT_BOOTSTRAP_REPS,
        "seed": args.seed,
        "progress_every": args.progress_every,
        "folds": FOLDS,
        "stage2_point_seed": STAGE2_RANDOM_SEED,
        "minimum_subgroup_curve_n": MIN_SUBGROUP_CURVE_N,
        "minimum_common_support_level_n": MIN_COMMON_SUPPORT_LEVEL_N,
        "common_support_rule": "intersection of nonmissing eligible subgroup source P05-P95",
        "grid_n": GRID_N,
        "common_support_headline_grid": COMMON_LINEAR_DEFINITION,
        "common_support_curve_estimator": "full-resample subgroup isotonic; same fitted subgroup curve as original-grid H",
        "common_support_grid": "101-point equally spaced grid within the common-support interval",
        "original_grid_curve_estimator": "full-resample subgroup isotonic; complete-sample reference",
        "tier_cutpoints": {
            "tier_1": {"mae_per_target_iqr_max": 0.20, "spread_per_target_iqr_max": 0.20},
            "tier_2": {"mae_per_target_iqr_max": 0.27, "spread_per_target_iqr_max": 0.35},
        },
        "stage3_point_audit": {
            "max_abs_mae_per_target_iqr_diff": max_mae_diff,
            "max_abs_original_grid_spread_per_target_iqr_diff": max_spread_diff,
            "loaded_tier_counts": dict(Counter(row["stage3_tier_final_preserved"] for row in tier_rows)),
        },
        "output_row_counts": {
            "common_support_subgroup_stability": len(detail_rows),
            "common_support_tier_sensitivity": len(tier_rows),
            "tier_bootstrap_stability": len(bootstrap_rows),
        },
        "outputs": [
            "outputs/tables/stage6_common_support_subgroup_stability.csv",
            "outputs/tables/stage6_common_support_tier_sensitivity.csv",
            "outputs/tables/stage6_tier_bootstrap_stability.csv",
            "outputs/tables/STAGE6_COMMON_SUPPORT_TIER_SUMMARY.md",
            "outputs/logs/stage6_common_support_tier_run.json",
        ],
        "guardrails": [
            "Stage 3 tier_final and featured_main are loaded and preserved",
            "fixed tier cutpoints; 0.27 is never reselected within bootstrap",
            "public raw data read in memory only",
            "aggregate outputs only",
            "no SEQN/person-day/minute-level output",
            "no individual OOF prediction output",
            "no bootstrap replicate-level output",
            "no log/manuscript edits by this script",
        ],
    }
    with (out_logs / "stage6_common_support_tier_run.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(run_log, handle, ensure_ascii=False, indent=2)
    log("done")


def load_continuous_points(out_dir: Path) -> dict[PairKey, dict[str, object]]:
    """Load the clean E/H pair table without any acceptability categories."""
    path = out_dir / "tables" / "stage3_translatability_map.csv"
    rows: dict[PairKey, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = PairKey(row["source_algorithm"], row["target_algorithm"])
            rows[key] = {
                "E": parse_float(row["mae_per_target_iqr"]),
                "H": parse_float(row["max_subgroup_p95_spread_over_target_iqr"]),
                "H_driver": row["max_subgroup_p95_spread_group_type"],
            }
    expected = {PairKey(source, target) for source in ALGORITHMS for target in ALGORITHMS if source != target}
    if set(rows) != expected:
        raise RuntimeError("Continuous E/H pair set is incomplete")
    return rows


def main_release() -> None:
    """Release entry point: continuous support diagnostics only, with no grades."""
    args = parse_args()
    started = datetime.now(timezone.utc)
    out_tables = args.out_dir / "tables"
    out_logs = args.out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    points = load_continuous_points(args.out_dir)
    cohort = build_daily_cohort(args.raw_dir)
    if cohort.subjects_n != 8646 or cohort.valid_days_n != 57080:
        raise RuntimeError(f"Common-support cohort drift: {cohort.subjects_n}/{cohort.valid_days_n}")
    strata, psu, _weights, design_rows = align_design(cohort, args.raw_dir)
    pairs = [PairKey(source, target) for source in ALGORITHMS for target in ALGORITHMS if source != target]
    all_indices = np.arange(cohort.subjects_n)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    point_results: dict[PairKey, PairComputation] = {}

    log("computing 42-pair continuous common-support estimates")
    for pair_number, key in enumerate(pairs, start=1):
        result = compute_pair(cohort, all_indices, key.source, key.target, STAGE2_RANDOM_SEED, keep_detail=True)
        point_results[key] = result
        detail_rows.extend(result.detail_rows)
        loaded = points[key]
        summary_rows.append({
            "source_algorithm": key.source,
            "target_algorithm": key.target,
            "n": result.n,
            "target_iqr": result.target_iqr,
            "E": float(loaded["E"]),
            "E_recomputed": result.mae_per_target_iqr,
            "E_absolute_reproduction_difference": abs(result.mae_per_target_iqr - float(loaded["E"])),
            "full_range_H": float(loaded["H"]),
            "full_range_H_driver": loaded["H_driver"],
            "full_range_H_recomputed": result.spread_by_definition[ORIGINAL_DEFINITION],
            "full_range_H_absolute_reproduction_difference": abs(result.spread_by_definition[ORIGINAL_DEFINITION] - float(loaded["H"])),
            "common_support_linear_H": result.spread_by_definition[COMMON_LINEAR_DEFINITION],
            "common_support_linear_driver": result.driver_by_definition[COMMON_LINEAR_DEFINITION],
            "interpretation": "continuous_support_sensitivity_without_acceptability_grade",
        })
        if pair_number == 1 or pair_number % 7 == 0:
            log(f"point pair {pair_number}/{len(pairs)} complete")

    max_e_diff = max(float(row["E_absolute_reproduction_difference"]) for row in summary_rows)
    max_h_diff = max(float(row["full_range_H_absolute_reproduction_difference"]) for row in summary_rows)
    if max_e_diff > 1e-10 or max_h_diff > 1e-10:
        raise RuntimeError(f"Point reproduction failed: E={max_e_diff:.12g}, H={max_h_diff:.12g}")

    rng = np.random.default_rng(args.seed)
    boot_mae: dict[PairKey, list[float]] = defaultdict(list)
    boot_spread: dict[tuple[PairKey, str], list[float]] = defaultdict(list)
    log(f"starting {args.bootstrap_reps} continuous PSU-within-stratum bootstrap replicates")
    for replicate in range(1, args.bootstrap_reps + 1):
        sample_idx = design_bootstrap_indices(strata, psu, rng)
        for key in pairs:
            result = compute_pair(cohort, sample_idx, key.source, key.target, args.seed + replicate, keep_detail=False)
            if finite(result.mae_per_target_iqr):
                boot_mae[key].append(result.mae_per_target_iqr)
            for definition in DEFINITIONS:
                value = result.spread_by_definition.get(definition, math.nan)
                if finite(value):
                    boot_spread[(key, definition)].append(value)
        if replicate == 1 or replicate == args.bootstrap_reps or replicate % args.progress_every == 0:
            log(f"bootstrap replicate {replicate}/{args.bootstrap_reps} complete")

    continuous_rows: list[dict[str, object]] = []
    for key in pairs:
        point = point_results[key]
        for definition in DEFINITIONS:
            e_values = boot_mae[key]
            h_values = boot_spread[(key, definition)]
            continuous_rows.append({
                "source_algorithm": key.source,
                "target_algorithm": key.target,
                "support_definition": definition,
                "point_E": point.mae_per_target_iqr,
                "E_ci_low": finite_percentile(e_values, CI_LOW_Q),
                "E_ci_high": finite_percentile(e_values, CI_HIGH_Q),
                "point_H": point.spread_by_definition[definition],
                "H_driver": point.driver_by_definition[definition],
                "H_ci_low": finite_percentile(h_values, CI_LOW_Q),
                "H_ci_high": finite_percentile(h_values, CI_HIGH_Q),
                "E_bootstrap_success": len(e_values),
                "H_bootstrap_success": len(h_values),
                "bootstrap_requested": args.bootstrap_reps,
                "ci_method": "percentile_PSU_within_stratum_bootstrap",
            })

    write_csv(out_tables / "stage6_common_support_subgroup_stability.csv", detail_rows, list(detail_rows[0]))
    write_csv(out_tables / "stage6_common_support_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(out_tables / "stage6_continuous_bootstrap.csv", continuous_rows, list(continuous_rows[0]))
    design_overall = next(row for row in design_rows if row["scope"] == "design_overall")
    run_log = {
        "script": "analysis/build_stage6_common_support.py",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_subjects": cohort.subjects_n,
        "cohort_valid_days": cohort.valid_days_n,
        "directed_pairs": len(pairs),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_full_spec_complete": args.bootstrap_reps == DEFAULT_BOOTSTRAP_REPS,
        "design_strata": len(np.unique(strata)),
        "design_stratum_psu_cells": design_overall["n_psu"],
        "minimum_subgroup_curve_n": MIN_SUBGROUP_CURVE_N,
        "minimum_common_support_level_n": MIN_COMMON_SUPPORT_LEVEL_N,
        "common_support_rule": "intersection of nonmissing eligible subgroup source P05-P95 using the same eligible subgroups as original-grid H",
        "grid_n": GRID_N,
        "common_support_headline_grid": COMMON_LINEAR_DEFINITION,
        "common_support_curve_estimator": "full-resample subgroup isotonic; same fitted subgroup curve as original-grid H",
        "common_support_grid": "101-point equally spaced grid within the common-support interval",
        "original_grid_curve_estimator": "full-resample subgroup isotonic; complete-sample reference",
        "point_reproduction": {"max_abs_E_difference": max_e_diff, "max_abs_H_difference": max_h_diff},
        "output_row_counts": {"detail": len(detail_rows), "summary": len(summary_rows), "continuous_bootstrap": len(continuous_rows)},
        "acceptability_grades_generated": False,
        "aggregate_only": True,
    }
    with (out_logs / "stage6_common_support_run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_log, handle, ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main_release()
