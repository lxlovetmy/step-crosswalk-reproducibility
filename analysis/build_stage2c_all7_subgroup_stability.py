#!/usr/bin/env python3
"""Stage 2c: all-7 directed-pair crosswalk subgroup stability.

Exposure-only. Rebuilds the locked all-7 D9-valid adult daily-steps cohort in
memory, then extends the MVP subgroup stability machinery to all 42 directed
source->target pairs. Outputs are aggregate tables and a run log only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from build_mvp_crosswalk_stability import (
    ALL_METHODS,
    FOLDS,
    HEADLINE_METHOD,
    MIN_SUBGROUP_CURVE_N,
    RANDOM_SEED,
    age_group,
    bmi_group,
    fit_crosswalk,
    read_demo_bmi,
    sex_group,
)
from build_mvp_exposure_qc import (
    TROIANO_FILE,
    collect_d9_valid_days,
    collect_wear_masks,
    process_step_algorithm,
    read_adult_seqns_from_demo,
    read_step_index,
    write_csv,
)
from build_stage2_all7_crosswalk_matrix import (
    ALGORITHMS,
    ALL_STEP_FILES,
    DAILY_KEY,
    aligned_array,
)
from gate0_d9_wear_audit import locate_file, pct, read_day_minmax, summary_stats


MVP_REFERENCE_P95_SPREAD = {
    "age_group": 2952.7,
    "sex": 1807.8,
    "bmi_group": 1193.2,
}


@dataclass
class DailyCohort:
    subjects_n: int
    valid_days_n: int
    subject_ids: np.ndarray
    seven_way_subject_ids: frozenset[str]
    sample_flow_counts: dict[str, int]
    daily_by_alg: dict[str, np.ndarray]
    age_group: np.ndarray
    sex: np.ndarray
    bmi_group: np.ndarray
    denominator_rows: list[dict[str, object]]


@dataclass
class PairData:
    source_algorithm: str
    target_algorithm: str
    x: np.ndarray
    y: np.ndarray
    age_group: np.ndarray
    sex: np.ndarray
    bmi_group: np.ndarray
    target_iqr: float


def log(message: str) -> None:
    print(f"[stage2c {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def is_finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def target_iqr(values: np.ndarray) -> float:
    stats = summary_stats(values.tolist())
    return float(stats["p75"]) - float(stats["p25"])


def safe_divide(value: float, denominator: float) -> float:
    if not is_finite(value) or not is_finite(denominator) or denominator <= 0:
        return math.nan
    return value / denominator


def preferred_group_order(group_type: str) -> list[str]:
    if group_type == "age_group":
        return ["20-39", "40-59", "60+", "missing"]
    if group_type == "sex":
        return ["male", "female", "missing"]
    if group_type == "bmi_group":
        return ["<25", "25-<30", ">=30", "missing"]
    raise ValueError(f"Unknown group type: {group_type}")


def group_values(data: PairData, group_type: str) -> np.ndarray:
    if group_type == "age_group":
        return data.age_group
    if group_type == "sex":
        return data.sex
    if group_type == "bmi_group":
        return data.bmi_group
    raise ValueError(f"Unknown group type: {group_type}")


def method_role(method: str) -> str:
    return "headline" if method == HEADLINE_METHOD else "fullsample_robustness"


def cross_validated_predictions(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    predictions = np.full(len(x), np.nan, dtype=float)
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in splitter.split(x):
        fitted = fit_crosswalk(method, x[train_idx], y[train_idx])
        predictions[test_idx] = fitted.predict(x[test_idx])
    return predictions


def residual_summary_row(
    data: PairData,
    method: str,
    scope_type: str,
    scope_level: str,
    target: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    residual = predicted - target
    abs_error = np.abs(residual)
    residual_stats = summary_stats(residual.tolist())
    abs_stats = summary_stats(abs_error.tolist())
    mae = float(np.mean(abs_error)) if len(abs_error) else math.nan
    rmse = math.sqrt(float(np.mean(residual ** 2))) if len(residual) else math.nan
    return {
        "exposure": "daily_steps",
        "source_algorithm": data.source_algorithm,
        "target_algorithm": data.target_algorithm,
        "method": method,
        "method_role": method_role(method),
        "scope_type": scope_type,
        "scope_level": scope_level,
        "n": len(target),
        "mean_error_pred_minus_target": residual_stats["mean"],
        "median_error_pred_minus_target": residual_stats["p50"],
        "error_sd": residual_stats["sd"],
        "error_p05": residual_stats["p05"],
        "error_p95": residual_stats["p95"],
        "mae": mae,
        "rmse": rmse,
        "absolute_error_p50": abs_stats["p50"],
        "absolute_error_p90": abs_stats["p90"],
        "absolute_error_p95": abs_stats["p95"],
        "target_iqr": data.target_iqr,
        "mae_per_target_iqr": safe_divide(mae, data.target_iqr),
        "note": "Full-sample source->target 5-fold OOF residual; aggregate row only.",
    }


def build_daily_cohort(raw_dir: Path) -> DailyCohort:
    physio_dir = raw_dir / "physionet_v1.0.1"
    troiano_path = locate_file(physio_dir, TROIANO_FILE)
    step_paths = {alg: locate_file(physio_dir, filename) for alg, filename in ALL_STEP_FILES.items()}

    log("reading adult DEMO filter and subgroup covariates")
    adult_seqns = read_adult_seqns_from_demo(raw_dir)
    covariates = read_demo_bmi(raw_dir)

    log("reading all 7 step file indices")
    step_indices = {alg: read_step_index(path) for alg, path in step_paths.items()}
    per_alg_pd = {alg: len(index) for alg, index in step_indices.items()}
    per_alg_subjects = {alg: len({seqn for seqn, _day in index}) for alg, index in step_indices.items()}
    seven_way = set.intersection(*step_indices.values())
    seven_way_subjects = {seqn for seqn, _day in seven_way}

    minmax_by_seqn = read_day_minmax(troiano_path)
    troiano_subjects = set(minmax_by_seqn)
    _valid_days_by_subject, final_subjects, final_valid_days, alignment_rows = collect_d9_valid_days(
        troiano_path=troiano_path,
        step_common_keys=seven_way,
        minmax_by_seqn=minmax_by_seqn,
        adult_seqns=adult_seqns,
    )
    align_map = {row["metric"]: row["value"] for row in alignment_rows}
    log(f"cohort rebuilt: {len(final_subjects)} adults, {len(final_valid_days)} valid person-days")

    wear_masks = collect_wear_masks(troiano_path, final_valid_days)
    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]] = {}
    aggregate_rows: list[dict[str, object]] = []
    for alg in ALGORITHMS:
        log(f"processing daily steps: {alg}")
        metrics, aggregate = process_step_algorithm(alg, step_paths[alg], wear_masks)
        metrics_by_algorithm[alg] = metrics
        aggregate_rows.append(
            {
                "algorithm": alg,
                "subjects_processed": aggregate["subjects_processed"],
                "valid_days_processed": aggregate["valid_days_processed"],
            }
        )

    subjects_sorted = sorted(final_subjects)
    daily_by_alg = {
        alg: aligned_array(metrics_by_algorithm[alg], subjects_sorted, DAILY_KEY)
        for alg in ALGORITHMS
    }
    age_groups: list[str] = []
    sex_groups: list[str] = []
    bmi_groups: list[str] = []
    bmi_missing_n = 0
    for seqn in subjects_sorted:
        covar = covariates.get(seqn, {"age": math.nan, "sex_code": math.nan, "bmi": math.nan})
        bmi_value = float(covar["bmi"])
        if math.isnan(bmi_value):
            bmi_missing_n += 1
        age_groups.append(age_group(float(covar["age"])))
        sex_groups.append(sex_group(float(covar["sex_code"])))
        bmi_groups.append(bmi_group(bmi_value))

    denominator_rows = [
        {
            "scope": "cohort_check",
            "group_type": "overall",
            "group_level": "per_algorithm_step_person_days",
            "n_subjects": "",
            "n_person_days": sorted(set(per_alg_pd.values())),
            "pct_subjects": "",
            "bmx_bmi_missing_n": "",
            "note": "Each of 7 step files; should be identical.",
        },
        {
            "scope": "cohort_check",
            "group_type": "overall",
            "group_level": "per_algorithm_step_subjects",
            "n_subjects": sorted(set(per_alg_subjects.values())),
            "n_person_days": "",
            "pct_subjects": "",
            "bmx_bmi_missing_n": "",
            "note": "Each of 7 step files; should be identical.",
        },
        {
            "scope": "cohort_check",
            "group_type": "overall",
            "group_level": "seven_way_intersection",
            "n_subjects": len(seven_way_subjects),
            "n_person_days": len(seven_way),
            "pct_subjects": "",
            "bmx_bmi_missing_n": "",
            "note": "Person-days present in all 7 step files before Troiano/D9/adult filters.",
        },
        {
            "scope": "cohort_check",
            "group_type": "overall",
            "group_level": "troiano_intersect_seven_way",
            "n_subjects": align_map.get("troiano_intersect_mvp_step_subjects", ""),
            "n_person_days": align_map.get("troiano_intersect_mvp_step_person_days", ""),
            "pct_subjects": "",
            "bmx_bmi_missing_n": "",
            "note": "Troiano rows intersecting all-7 step person-days; collect_d9_valid_days labels this internally as mvp_step.",
        },
        {
            "scope": "cohort_check",
            "group_type": "overall",
            "group_level": "d9_valid_adult_final",
            "n_subjects": len(final_subjects),
            "n_person_days": len(final_valid_days),
            "pct_subjects": 1.0,
            "bmx_bmi_missing_n": bmi_missing_n,
            "note": "Frozen Stage 2c denominator: all-7 step intersection, Troiano D9 main rule, adults >=20.",
        },
    ]
    for row in aggregate_rows:
        denominator_rows.append(
            {
                "scope": "algorithm_processing",
                "group_type": "algorithm",
                "group_level": row["algorithm"],
                "n_subjects": row["subjects_processed"],
                "n_person_days": row["valid_days_processed"],
                "pct_subjects": "",
                "bmx_bmi_missing_n": "",
                "note": "Subjects/person-days processed for daily steps after shared D9 valid-day mask.",
            }
        )
    subgroup_arrays = {
        "age_group": np.asarray(age_groups, dtype=object),
        "sex": np.asarray(sex_groups, dtype=object),
        "bmi_group": np.asarray(bmi_groups, dtype=object),
    }
    for group_type, values in subgroup_arrays.items():
        for level in preferred_group_order(group_type):
            n = int(np.sum(values == level))
            denominator_rows.append(
                {
                    "scope": "subgroup_count",
                    "group_type": group_type,
                    "group_level": level,
                    "n_subjects": n,
                    "n_person_days": "",
                    "pct_subjects": pct(n, len(final_subjects)),
                    "bmx_bmi_missing_n": bmi_missing_n if group_type == "bmi_group" and level == "missing" else "",
                    "note": "Missing rows are counted for residual summaries; missing strata are not fitted as subgroup curves.",
                }
            )

    return DailyCohort(
        subjects_n=len(final_subjects),
        valid_days_n=len(final_valid_days),
        subject_ids=np.asarray(subjects_sorted, dtype=object),
        seven_way_subject_ids=frozenset(seven_way_subjects),
        sample_flow_counts={
            "seven_way_subjects": len(seven_way_subjects),
            "seven_way_person_days": len(seven_way),
            "troiano_intersection_subjects": len(seven_way_subjects & troiano_subjects),
            "troiano_intersection_person_days": int(
                align_map.get("troiano_intersect_mvp_step_person_days", 0)
            ),
            "valid_wear_subjects_before_adult_filter": int(
                align_map.get("d9_valid_subjects_before_adult_filter", 0)
            ),
            "final_adult_subjects": len(final_subjects),
            "final_valid_person_days": len(final_valid_days),
        },
        daily_by_alg=daily_by_alg,
        age_group=np.asarray(age_groups, dtype=object),
        sex=np.asarray(sex_groups, dtype=object),
        bmi_group=np.asarray(bmi_groups, dtype=object),
        denominator_rows=denominator_rows,
    )


def pair_data(cohort: DailyCohort, source: str, target: str) -> PairData:
    x = cohort.daily_by_alg[source]
    y = cohort.daily_by_alg[target]
    mask = ~np.isnan(x) & ~np.isnan(y)
    return PairData(
        source_algorithm=source,
        target_algorithm=target,
        x=x[mask],
        y=y[mask],
        age_group=cohort.age_group[mask],
        sex=cohort.sex[mask],
        bmi_group=cohort.bmi_group[mask],
        target_iqr=target_iqr(y[mask]),
    )


def build_fullsample_oof_residual_rows(data: PairData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        predicted = cross_validated_predictions(data.x, data.y, method)
        rows.append(residual_summary_row(data, method, "overall", "all", data.y, predicted))
        for group_type in ["age_group", "sex", "bmi_group"]:
            groups = group_values(data, group_type)
            for level in preferred_group_order(group_type):
                mask = groups == level
                if not np.any(mask):
                    continue
                rows.append(residual_summary_row(data, method, group_type, level, data.y[mask], predicted[mask]))
    return rows


def build_curve_rows_for_pair(
    data: PairData,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    divergence_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []

    x_grid = np.linspace(float(np.quantile(data.x, 0.05)), float(np.quantile(data.x, 0.95)), 101)
    full_model = fit_crosswalk(HEADLINE_METHOD, data.x, data.y)
    full_pred = full_model.predict(x_grid)

    for group_type in ["age_group", "sex", "bmi_group"]:
        groups = group_values(data, group_type)
        subgroup_predictions: list[np.ndarray] = []
        subgroup_levels: list[str] = []
        subgroup_ns: list[int] = []
        for level in preferred_group_order(group_type):
            if level == "missing":
                continue
            mask = groups == level
            n_group = int(np.sum(mask))
            if n_group < MIN_SUBGROUP_CURVE_N:
                continue
            subgroup_model = fit_crosswalk(HEADLINE_METHOD, data.x[mask], data.y[mask])
            subgroup_pred = subgroup_model.predict(x_grid)
            subgroup_predictions.append(subgroup_pred)
            subgroup_levels.append(level)
            subgroup_ns.append(n_group)

            diff = subgroup_pred - full_pred
            abs_diff = np.abs(diff)
            diff_stats = summary_stats(diff.tolist())
            abs_stats = summary_stats(abs_diff.tolist())
            mean_diff = float(diff_stats["mean"])
            median_diff = float(diff_stats["p50"])
            max_abs = float(np.max(abs_diff))
            p95_abs = float(abs_stats["p95"])
            divergence_rows.append(
                {
                    "exposure": "daily_steps",
                    "source_algorithm": data.source_algorithm,
                    "target_algorithm": data.target_algorithm,
                    "method": HEADLINE_METHOD,
                    "method_role": "headline_subgroup_curve",
                    "group_type": group_type,
                    "group_level": level,
                    "n_group": n_group,
                    "grid_n": len(x_grid),
                    "source_grid_min": float(np.min(x_grid)),
                    "source_grid_max": float(np.max(x_grid)),
                    "target_iqr": data.target_iqr,
                    "mean_subgroup_minus_full_predicted_target": mean_diff,
                    "mean_subgroup_minus_full_predicted_target_over_target_iqr": safe_divide(mean_diff, data.target_iqr),
                    "median_subgroup_minus_full_predicted_target": median_diff,
                    "median_subgroup_minus_full_predicted_target_over_target_iqr": safe_divide(median_diff, data.target_iqr),
                    "max_abs_subgroup_minus_full_predicted_target": max_abs,
                    "max_abs_subgroup_minus_full_predicted_target_over_target_iqr": safe_divide(max_abs, data.target_iqr),
                    "p95_abs_subgroup_minus_full_predicted_target": p95_abs,
                    "p95_abs_subgroup_minus_full_predicted_target_over_target_iqr": safe_divide(p95_abs, data.target_iqr),
                    "note": "Subgroup-specific isotonic curve compared with full-sample isotonic curve on an aggregate source grid.",
                }
            )
            for x_value, full_value, subgroup_value in zip(x_grid, full_pred, subgroup_pred):
                diff_value = float(subgroup_value - full_value)
                grid_rows.append(
                    {
                        "exposure": "daily_steps",
                        "source_algorithm": data.source_algorithm,
                        "target_algorithm": data.target_algorithm,
                        "method": HEADLINE_METHOD,
                        "group_type": group_type,
                        "group_level": level,
                        "source_input": float(x_value),
                        "predicted_target_full_sample": float(full_value),
                        "predicted_target_subgroup": float(subgroup_value),
                        "subgroup_minus_full": diff_value,
                        "subgroup_minus_full_over_target_iqr": safe_divide(diff_value, data.target_iqr),
                        "target_iqr": data.target_iqr,
                    }
                )

        if len(subgroup_predictions) >= 2:
            matrix = np.vstack(subgroup_predictions)
            spread = np.max(matrix, axis=0) - np.min(matrix, axis=0)
            spread_stats = summary_stats(spread.tolist())
            mean_spread = float(spread_stats["mean"])
            p50_spread = float(spread_stats["p50"])
            p95_spread = float(spread_stats["p95"])
            max_spread = float(spread_stats["max"])
            spread_rows.append(
                {
                    "exposure": "daily_steps",
                    "source_algorithm": data.source_algorithm,
                    "target_algorithm": data.target_algorithm,
                    "method": HEADLINE_METHOD,
                    "method_role": "headline_subgroup_curve",
                    "group_type": group_type,
                    "n_groups_compared": len(subgroup_levels),
                    "group_levels": "|".join(subgroup_levels),
                    "group_ns": "|".join(str(n) for n in subgroup_ns),
                    "grid_n": len(x_grid),
                    "source_grid_min": float(np.min(x_grid)),
                    "source_grid_max": float(np.max(x_grid)),
                    "target_iqr": data.target_iqr,
                    "mean_between_subgroup_spread_predicted_target": mean_spread,
                    "mean_between_subgroup_spread_predicted_target_over_target_iqr": safe_divide(mean_spread, data.target_iqr),
                    "p50_between_subgroup_spread_predicted_target": p50_spread,
                    "p50_between_subgroup_spread_predicted_target_over_target_iqr": safe_divide(p50_spread, data.target_iqr),
                    "p95_between_subgroup_spread_predicted_target": p95_spread,
                    "p95_between_subgroup_spread_predicted_target_over_target_iqr": safe_divide(p95_spread, data.target_iqr),
                    "max_between_subgroup_spread_predicted_target": max_spread,
                    "max_between_subgroup_spread_predicted_target_over_target_iqr": safe_divide(max_spread, data.target_iqr),
                    "note": "At each source-grid point, spread = max subgroup isotonic curve - min subgroup isotonic curve.",
                }
            )

    return divergence_rows, spread_rows, grid_rows


def build_summary(
    path: Path,
    cohort: DailyCohort,
    residual_rows: list[dict[str, object]],
    divergence_rows: list[dict[str, object]],
    spread_rows: list[dict[str, object]],
) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    cohort_row = next(row for row in cohort.denominator_rows if row["group_level"] == "d9_valid_adult_final")
    top_spreads = sorted(
        spread_rows,
        key=lambda row: float(row["p95_between_subgroup_spread_predicted_target_over_target_iqr"]),
        reverse=True,
    )
    top_divergence = sorted(
        divergence_rows,
        key=lambda row: float(row["p95_abs_subgroup_minus_full_predicted_target_over_target_iqr"]),
        reverse=True,
    )
    robustness_rows = [row for row in residual_rows if row["scope_type"] == "overall"]
    robustness_by_method = {
        method: [row for row in robustness_rows if row["method"] == method]
        for method in ALL_METHODS
    }
    adept_oak_spreads = {
        row["group_type"]: row
        for row in spread_rows
        if row["source_algorithm"] == "adept"
        and row["target_algorithm"] == "oak"
        and row["method"] == HEADLINE_METHOD
    }

    lines = [
        "# Stage 2c All-7 Subgroup Crosswalk Stability Summary",
        "",
        f"- Generated: {generated}",
        "- Scope: exposure-layer only; all 42 directed daily-steps source->target pairs across 7 algorithms.",
        "- Cohort: all-7 step intersection + Troiano D9 main valid-day rule + adults >=20; no HbA1c, waist, DIQ, RXQ, or weights linked.",
        f"- Denominator check: {cohort.subjects_n} adults / {cohort.valid_days_n} valid person-days. Required frozen denominator: 8,646 / 57,080.",
        "- Methods: full-sample 5-fold OOF residual summaries for isotonic headline plus quantile and monotone-spline robustness; subgroup curves use isotonic headline only.",
        "- Subgroups: age 20-39 / 40-59 / 60+ / missing; sex male / female / missing; BMI <25 / 25-<30 / >=30 / missing. Missing strata are counted and included in OOF residual summaries; missing strata are not independently fitted as subgroup curves, mirroring the MVP script.",
        "- Cross-pair headline sorting below uses target-IQR normalized columns; raw target steps/day columns are retained in the CSV tables for scale-specific inspection.",
        "- Guardrails: aggregate tables only; no SEQN/person-day/minute/person-level OOF prediction or bootstrap replicate output; docs/SAP.md unchanged by this script.",
        "",
        "## Cohort Check",
        "",
        "| metric | n_subjects | n_person_days | note |",
        "|---|---:|---:|---|",
        f"| d9_valid_adult_final | {cohort_row['n_subjects']} | {cohort_row['n_person_days']} | {cohort_row['note']} |",
        "",
        "## Full-Sample OOF Robustness (Overall Rows)",
        "",
        "| method | pairs | median MAE/target-IQR | p95 MAE/target-IQR | max MAE/target-IQR |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ALL_METHODS:
        rows = robustness_by_method[method]
        values = [float(row["mae_per_target_iqr"]) for row in rows]
        stats = summary_stats(values)
        lines.append(
            f"| {method} | {len(rows)} | {fmt(float(stats['p50']), 3)} | {fmt(float(stats['p95']), 3)} | {fmt(float(stats['max']), 3)} |"
        )

    lines.extend(
        [
            "",
            "## Largest Between-Subgroup Curve Spread (Normalized)",
            "",
            "| rank | source -> target | subgroup | p95 spread / target-IQR | p95 spread raw target steps/day | groups |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(top_spreads[:12], start=1):
        lines.append(
            "| {rank} | {source}->{target} | {group} | {norm} | {raw} | {groups} |".format(
                rank=rank,
                source=row["source_algorithm"],
                target=row["target_algorithm"],
                group=row["group_type"],
                norm=fmt(float(row["p95_between_subgroup_spread_predicted_target_over_target_iqr"]), 3),
                raw=fmt(float(row["p95_between_subgroup_spread_predicted_target"]), 1),
                groups=str(row["group_levels"]).replace("|", ", "),
            )
        )
    lines.extend(
        [
            "",
            "## Largest Subgroup-vs-Full Curve Divergence (Normalized)",
            "",
            "| rank | source -> target | subgroup | level | p95 abs divergence / target-IQR | p95 abs divergence raw target steps/day |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_divergence[:12], start=1):
        lines.append(
            "| {rank} | {source}->{target} | {group} | {level} | {norm} | {raw} |".format(
                rank=rank,
                source=row["source_algorithm"],
                target=row["target_algorithm"],
                group=row["group_type"],
                level=row["group_level"],
                norm=fmt(float(row["p95_abs_subgroup_minus_full_predicted_target_over_target_iqr"]), 3),
                raw=fmt(float(row["p95_abs_subgroup_minus_full_predicted_target"]), 1),
            )
        )
    lines.extend(
        [
            "",
            "## Internal MVP Reproduction Check: adept -> oak",
            "",
            "| subgroup | Stage 2c p95 spread (oak steps/day) | MVP reference p95 spread | difference | Stage 2c p95 / target-IQR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group_type in ["age_group", "sex", "bmi_group"]:
        row = adept_oak_spreads.get(group_type)
        if row is None:
            lines.append(f"| {group_type} | NA | {fmt(MVP_REFERENCE_P95_SPREAD[group_type], 1)} | NA | NA |")
            continue
        stage2c_value = float(row["p95_between_subgroup_spread_predicted_target"])
        reference = MVP_REFERENCE_P95_SPREAD[group_type]
        lines.append(
            "| {group} | {current} | {reference} | {diff} | {norm} |".format(
                group=group_type,
                current=fmt(stage2c_value, 1),
                reference=fmt(reference, 1),
                diff=fmt(stage2c_value - reference, 1),
                norm=fmt(float(row["p95_between_subgroup_spread_predicted_target_over_target_iqr"]), 3),
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `stage2c_all7_subgroup_counts.csv`: denominator and subgroup counts, including missing rows.",
            "- `stage2c_all7_subgroup_fullsample_oof_residuals.csv`: overall and subgroup residual summaries for isotonic/quantile/monotone-spline full-sample OOF mappings.",
            "- `stage2c_all7_subgroup_curve_divergence.csv`: isotonic subgroup-specific curve vs full-sample curve divergence, with raw and target-IQR normalized columns.",
            "- `stage2c_all7_subgroup_curve_spread.csv`: isotonic between-subgroup curve spread, with raw and target-IQR normalized columns.",
            "- `stage2c_all7_subgroup_curve_grid_daily_headline.csv`: aggregate curve grid coordinates for daily-steps isotonic headline only.",
            "",
            "## Interpretation Guardrail",
            "",
            "- This is not an outcome analysis and does not estimate metabolic associations. Cross-pair stability should be read together with Stage 2a full-sample residuals and the fixed-threshold/AUC results already produced in Stage 2b.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_logs = args.out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    cohort = build_daily_cohort(args.raw_dir)
    if cohort.subjects_n != 8646 or cohort.valid_days_n != 57080:
        raise RuntimeError(
            f"Stage 2c cohort drift: expected 8646/57080, got {cohort.subjects_n}/{cohort.valid_days_n}."
        )

    residual_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    pairs = list(itertools.permutations(ALGORITHMS, 2))
    log(f"running {len(pairs)} directed pairs")
    for idx, (source, target) in enumerate(pairs, start=1):
        log(f"pair {idx:02d}/{len(pairs)}: {source}->{target}")
        data = pair_data(cohort, source, target)
        residual_rows.extend(build_fullsample_oof_residual_rows(data))
        pair_divergence, pair_spread, pair_grid = build_curve_rows_for_pair(data)
        divergence_rows.extend(pair_divergence)
        spread_rows.extend(pair_spread)
        grid_rows.extend(pair_grid)

    write_csv(
        out_tables / "stage2c_all7_subgroup_counts.csv",
        cohort.denominator_rows,
        ["scope", "group_type", "group_level", "n_subjects", "n_person_days", "pct_subjects", "bmx_bmi_missing_n", "note"],
    )
    write_csv(
        out_tables / "stage2c_all7_subgroup_fullsample_oof_residuals.csv",
        residual_rows,
        [
            "exposure", "source_algorithm", "target_algorithm", "method", "method_role",
            "scope_type", "scope_level", "n", "mean_error_pred_minus_target",
            "median_error_pred_minus_target", "error_sd", "error_p05", "error_p95",
            "mae", "rmse", "absolute_error_p50", "absolute_error_p90",
            "absolute_error_p95", "target_iqr", "mae_per_target_iqr", "note",
        ],
    )
    write_csv(
        out_tables / "stage2c_all7_subgroup_curve_divergence.csv",
        divergence_rows,
        [
            "exposure", "source_algorithm", "target_algorithm", "method", "method_role",
            "group_type", "group_level", "n_group", "grid_n", "source_grid_min",
            "source_grid_max", "target_iqr", "mean_subgroup_minus_full_predicted_target",
            "mean_subgroup_minus_full_predicted_target_over_target_iqr",
            "median_subgroup_minus_full_predicted_target",
            "median_subgroup_minus_full_predicted_target_over_target_iqr",
            "max_abs_subgroup_minus_full_predicted_target",
            "max_abs_subgroup_minus_full_predicted_target_over_target_iqr",
            "p95_abs_subgroup_minus_full_predicted_target",
            "p95_abs_subgroup_minus_full_predicted_target_over_target_iqr",
            "note",
        ],
    )
    write_csv(
        out_tables / "stage2c_all7_subgroup_curve_spread.csv",
        spread_rows,
        [
            "exposure", "source_algorithm", "target_algorithm", "method", "method_role",
            "group_type", "n_groups_compared", "group_levels", "group_ns", "grid_n",
            "source_grid_min", "source_grid_max", "target_iqr",
            "mean_between_subgroup_spread_predicted_target",
            "mean_between_subgroup_spread_predicted_target_over_target_iqr",
            "p50_between_subgroup_spread_predicted_target",
            "p50_between_subgroup_spread_predicted_target_over_target_iqr",
            "p95_between_subgroup_spread_predicted_target",
            "p95_between_subgroup_spread_predicted_target_over_target_iqr",
            "max_between_subgroup_spread_predicted_target",
            "max_between_subgroup_spread_predicted_target_over_target_iqr",
            "note",
        ],
    )
    write_csv(
        out_tables / "stage2c_all7_subgroup_curve_grid_daily_headline.csv",
        grid_rows,
        [
            "exposure", "source_algorithm", "target_algorithm", "method", "group_type",
            "group_level", "source_input", "predicted_target_full_sample",
            "predicted_target_subgroup", "subgroup_minus_full",
            "subgroup_minus_full_over_target_iqr", "target_iqr",
        ],
    )
    build_summary(
        out_tables / "STAGE2C_ALL7_SUBGROUP_STABILITY_SUMMARY.md",
        cohort,
        residual_rows,
        divergence_rows,
        spread_rows,
    )

    finished = datetime.now(timezone.utc)
    run_log = {
        "script": "scripts/python/build_stage2c_all7_subgroup_stability.py",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "algorithms": ALGORITHMS,
        "directed_pairs": len(pairs),
        "exposure": "daily_steps",
        "headline_method": HEADLINE_METHOD,
        "fullsample_robustness_methods": [method for method in ALL_METHODS if method != HEADLINE_METHOD],
        "folds": FOLDS,
        "random_seed": RANDOM_SEED,
        "min_subgroup_curve_n": MIN_SUBGROUP_CURVE_N,
        "denominator_subjects": cohort.subjects_n,
        "denominator_valid_days": cohort.valid_days_n,
        "target_iqr_normalized_columns": True,
        "outputs": [
            "outputs/tables/stage2c_all7_subgroup_counts.csv",
            "outputs/tables/stage2c_all7_subgroup_fullsample_oof_residuals.csv",
            "outputs/tables/stage2c_all7_subgroup_curve_divergence.csv",
            "outputs/tables/stage2c_all7_subgroup_curve_spread.csv",
            "outputs/tables/stage2c_all7_subgroup_curve_grid_daily_headline.csv",
            "outputs/tables/STAGE2C_ALL7_SUBGROUP_STABILITY_SUMMARY.md",
            "outputs/logs/stage2c_all7_subgroup_stability_run.json",
        ],
        "guardrails": [
            "exposure-layer only",
            "daily steps only; no cadence subgrouping",
            "no HbA1c/waist/DIQ/RXQ linkage",
            "no weights or survey design",
            "aggregate tables only",
            "no SEQN/person-day/minute/person-level OOF prediction output",
            "no bootstrap replicate output",
            "no SAP edits or new method lock",
        ],
    }
    with (out_logs / "stage2c_all7_subgroup_stability_run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_log, handle, ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main()
