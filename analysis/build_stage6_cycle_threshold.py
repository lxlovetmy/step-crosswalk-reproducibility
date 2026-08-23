#!/usr/bin/env python3
"""Stage 6: cross-cycle transport and fixed-threshold OOF validation.

This post-freeze validation add-on implements two analyses approved in SAP
v1.5 while preserving every earlier Stage 2/3 result:

1. Cross-cycle transport of each of the 42 directed isotonic crosswalks
   (NHANES 2011-2012 G -> 2013-2014 H and H -> G), with a within-test-cycle
   grouped five-fold OOF comparator and PSU-within-stratum bootstrap CIs.
2. Whole-cohort grouped five-fold OOF fixed-threshold classification for all
   42 directed pairs at 7,000/8,000/10,000 target steps/day, unweighted under
   SAP D7 with WTMEC4YR-weighted sensitivity and design-bootstrap CIs.

Only aggregate point and CI tables, a Markdown summary, and a JSON run log are
written. No participant identifier, participant-level prediction, person-day,
minute-level, or bootstrap-replicate table is exported.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from build_mvp_crosswalk_stability import (
    FIXED_STEP_THRESHOLDS,
    FOLDS,
    HEADLINE_METHOD,
    RANDOM_SEED as CROSSWALK_SEED,
    fit_crosswalk,
)
from build_mvp_exposure_qc import write_csv
from build_stage2_all7_crosswalk_matrix import ALGORITHMS
from build_stage2c_all7_subgroup_stability import DailyCohort, build_daily_cohort


DEFAULT_BOOTSTRAP_REPS = 300
DEFAULT_SEED = 20260715
CI_LOW_Q = 2.5
CI_HIGH_Q = 97.5
DEGENERATE_POS_RATE = 0.02

CYCLE_G = "2011-2012_G"
CYCLE_H = "2013-2014_H"
CYCLES = (CYCLE_G, CYCLE_H)

TRANSPORT_POINT_FILE = "stage6_cross_cycle_transport_validation.csv"
TRANSPORT_CI_FILE = "stage6_cross_cycle_transport_ci.csv"
THRESHOLD_POINT_FILE = "stage6_fixed_threshold_oof_reclassification.csv"
THRESHOLD_CI_FILE = "stage6_fixed_threshold_oof_reclassification_ci.csv"
SUMMARY_FILE = "STAGE6_CROSS_CYCLE_THRESHOLD_SUMMARY.md"
RUN_LOG_FILE = "stage6_cycle_threshold_run.json"

TRANSPORT_METRICS = (
    "test_target_iqr",
    "transport_bias_pred_minus_target",
    "transport_mae",
    "transport_rmse",
    "transport_abs_error_p95",
    "transport_mae_per_target_iqr",
    "within_test_oof_bias_pred_minus_target",
    "within_test_oof_mae",
    "within_test_oof_rmse",
    "within_test_oof_abs_error_p95",
    "within_test_oof_mae_per_target_iqr",
    "transport_penalty_mae",
    "transport_penalty_mae_per_target_iqr",
    "transport_to_within_test_oof_mae_ratio",
)

THRESHOLD_METRIC_UNITS = {
    "observed_target_attainment_pct": "percent",
    "mapped_source_attainment_pct": "percent",
    "attainment_difference_pp": "percentage_points",
    "discordant_reclassification_pct": "percent",
    "cohen_kappa": "coefficient",
    "positive_agreement": "proportion",
    "negative_agreement": "proportion",
}


def log(message: str) -> None:
    print(f"[stage6-cycle-threshold {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6 cross-cycle transport and fixed-threshold OOF validation."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be >= 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    return args


def parse_float_or_nan(value: str | None) -> float:
    if value is None:
        return math.nan
    value = value.strip()
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_demo_design_cycle(raw_dir: Path) -> dict[str, dict[str, object]]:
    """Read cycle/design/weight fields from DEMO without writing row-level data."""

    r_raw_dir = json.dumps(str(raw_dir))
    code = f"""
suppressPackageStartupMessages(library(haven))
normalize_seqn <- function(x) as.character(as.integer(as.numeric(x)))
read_cycle <- function(cycle_dir, suffix, cycle_label) {{
  demo_path <- file.path({r_raw_dir}, cycle_dir, paste0("DEMO_", suffix, ".XPT"))
  if (!file.exists(demo_path)) stop("missing DEMO file: ", demo_path)
  demo <- read_xpt(demo_path)[, c("SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC2YR")]
  demo$cycle <- cycle_label
  demo
}}
out <- rbind(
  read_cycle("nhanes_2011_2012", "G", "{CYCLE_G}"),
  read_cycle("nhanes_2013_2014", "H", "{CYCLE_H}")
)
out$SEQN <- normalize_seqn(out$SEQN)
out$SDMVSTRA <- as.numeric(out$SDMVSTRA)
out$SDMVPSU <- as.numeric(out$SDMVPSU)
out$WTMEC4YR <- as.numeric(out$WTMEC2YR) / 2
write.csv(
  out[, c("SEQN", "cycle", "SDMVSTRA", "SDMVPSU", "WTMEC4YR")],
  row.names = FALSE,
  na = ""
)
"""
    completed = subprocess.run(
        ["Rscript", "--vanilla", "-e", code],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[str, dict[str, object]] = {}
    for row in csv.DictReader(io.StringIO(completed.stdout)):
        subject = row.get("SEQN", "").strip()
        if not subject:
            continue
        rows[subject] = {
            "cycle": row.get("cycle", "").strip(),
            "stratum": parse_float_or_nan(row.get("SDMVSTRA")),
            "psu": parse_float_or_nan(row.get("SDMVPSU")),
            "weight": parse_float_or_nan(row.get("WTMEC4YR")),
        }
    return rows


def align_design(
    cohort: DailyCohort,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    design = read_demo_design_cycle(raw_dir)
    cycles: list[str] = []
    strata: list[float] = []
    psus: list[float] = []
    weights: list[float] = []
    for subject in cohort.subject_ids:
        row = design.get(str(subject))
        if row is None:
            raise RuntimeError("A D9 cohort participant is absent from DEMO design data.")
        cycles.append(str(row["cycle"]))
        strata.append(float(row["stratum"]))
        psus.append(float(row["psu"]))
        weights.append(float(row["weight"]))

    cycle_arr = np.asarray(cycles, dtype=object)
    strata_arr = np.asarray(strata, dtype=float)
    psu_arr = np.asarray(psus, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    valid = (
        np.isin(cycle_arr, np.asarray(CYCLES, dtype=object))
        & np.isfinite(strata_arr)
        & np.isfinite(psu_arr)
        & np.isfinite(weight_arr)
        & (weight_arr > 0)
    )
    if not np.all(valid):
        raise RuntimeError(f"Invalid cycle/design/weight data for {int(np.sum(~valid))} cohort participants.")

    audit_rows: list[dict[str, object]] = []
    for cycle in CYCLES:
        cycle_mask = cycle_arr == cycle
        audit_rows.append(
            {
                "scope": "cycle",
                "cycle": cycle,
                "n_subjects": int(np.sum(cycle_mask)),
                "n_strata": int(len(np.unique(strata_arr[cycle_mask]))),
                "n_stratum_psu_cells": int(
                    len(set(zip(strata_arr[cycle_mask].tolist(), psu_arr[cycle_mask].tolist())))
                ),
                "weighted_n_wtmec4yr": float(np.sum(weight_arr[cycle_mask])),
            }
        )
    audit_rows.append(
        {
            "scope": "overall",
            "cycle": "all",
            "n_subjects": int(len(cycle_arr)),
            "n_strata": int(len(np.unique(strata_arr))),
            "n_stratum_psu_cells": int(len(set(zip(strata_arr.tolist(), psu_arr.tolist())))),
            "weighted_n_wtmec4yr": float(np.sum(weight_arr)),
        }
    )
    return cycle_arr, strata_arr, psu_arr, weight_arr, audit_rows


def design_bootstrap_local_indices(
    strata: np.ndarray,
    psu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return local indices after PSU resampling within every stratum."""

    sampled_blocks: list[np.ndarray] = []
    for stratum in sorted(np.unique(strata).tolist()):
        stratum_mask = strata == stratum
        stratum_psus = np.asarray(sorted(np.unique(psu[stratum_mask]).tolist()), dtype=float)
        if len(stratum_psus) < 1:
            continue
        selected = rng.choice(stratum_psus, size=len(stratum_psus), replace=True)
        for selected_psu in selected:
            sampled_blocks.append(np.flatnonzero(stratum_mask & (psu == selected_psu)))
    if not sampled_blocks:
        raise RuntimeError("Design bootstrap produced no observations.")
    sampled = np.concatenate(sampled_blocks)
    rng.shuffle(sampled)
    return sampled


def grouped_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Five-fold OOF prediction with every copy of a subject kept in one fold."""

    prediction = np.full(len(x), np.nan, dtype=float)
    unique_subjects = np.asarray(sorted(set(subject_ids.tolist())), dtype=object)
    n_splits = min(FOLDS, len(unique_subjects))
    if n_splits < 2:
        return fit_crosswalk(HEADLINE_METHOD, x, y).predict(x)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_unique, test_unique in splitter.split(unique_subjects):
        train_subjects = set(unique_subjects[train_unique].tolist())
        test_subjects = set(unique_subjects[test_unique].tolist())
        train_mask = np.fromiter((item in train_subjects for item in subject_ids), dtype=bool, count=len(x))
        test_mask = np.fromiter((item in test_subjects for item in subject_ids), dtype=bool, count=len(x))
        fitted = fit_crosswalk(HEADLINE_METHOD, x[train_mask], y[train_mask])
        prediction[test_mask] = fitted.predict(x[test_mask])
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Grouped OOF prediction contains non-finite values.")
    return prediction


def target_iqr(y: np.ndarray) -> float:
    return float(np.percentile(y, 75) - np.percentile(y, 25))


def safe_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    absolute = np.abs(residual)
    iqr = target_iqr(target)
    mae = float(np.mean(absolute))
    return {
        "bias_pred_minus_target": float(np.mean(residual)),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "abs_error_p95": float(np.percentile(absolute, 95)),
        "target_iqr": iqr,
        "mae_per_target_iqr": safe_divide(mae, iqr),
    }


def finite_percentile(values: list[float], q: float) -> float:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return math.nan
    return float(np.percentile(clean, q))


def cross_cycle_row(
    cohort: DailyCohort,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    source: str,
    target: str,
    train_cycle: str,
    test_cycle: str,
    oof_seed: int,
) -> dict[str, object]:
    x_train_all = cohort.daily_by_alg[source][train_idx]
    y_train_all = cohort.daily_by_alg[target][train_idx]
    x_test_all = cohort.daily_by_alg[source][test_idx]
    y_test_all = cohort.daily_by_alg[target][test_idx]
    train_mask = np.isfinite(x_train_all) & np.isfinite(y_train_all)
    test_mask = np.isfinite(x_test_all) & np.isfinite(y_test_all)
    x_train = x_train_all[train_mask]
    y_train = y_train_all[train_mask]
    x_test = x_test_all[test_mask]
    y_test = y_test_all[test_mask]
    test_ids = cohort.subject_ids[test_idx][test_mask]

    locked_model = fit_crosswalk(HEADLINE_METHOD, x_train, y_train)
    transported = locked_model.predict(x_test)
    within_test_oof = grouped_oof_predictions(x_test, y_test, test_ids, oof_seed)
    transport = error_metrics(transported, y_test)
    comparator = error_metrics(within_test_oof, y_test)
    penalty_mae = transport["mae"] - comparator["mae"]
    test_iqr = transport["target_iqr"]

    return {
        "exposure": "daily_steps",
        "method": HEADLINE_METHOD,
        "source_algorithm": source,
        "target_algorithm": target,
        "train_cycle": train_cycle,
        "test_cycle": test_cycle,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_target_iqr": test_iqr,
        "transport_bias_pred_minus_target": transport["bias_pred_minus_target"],
        "transport_mae": transport["mae"],
        "transport_rmse": transport["rmse"],
        "transport_abs_error_p95": transport["abs_error_p95"],
        "transport_mae_per_target_iqr": transport["mae_per_target_iqr"],
        "within_test_oof_bias_pred_minus_target": comparator["bias_pred_minus_target"],
        "within_test_oof_mae": comparator["mae"],
        "within_test_oof_rmse": comparator["rmse"],
        "within_test_oof_abs_error_p95": comparator["abs_error_p95"],
        "within_test_oof_mae_per_target_iqr": comparator["mae_per_target_iqr"],
        "transport_penalty_mae": penalty_mae,
        "transport_penalty_mae_per_target_iqr": safe_divide(penalty_mae, test_iqr),
        "transport_to_within_test_oof_mae_ratio": safe_divide(transport["mae"], comparator["mae"]),
        "note": (
            "The source-to-target isotonic crosswalk was fitted only in the train cycle and "
            "locked before test-cycle evaluation; comparator is grouped five-fold OOF within "
            "the test cycle and is not used to tune the transported model."
        ),
    }


def weighted_classification_metrics(
    mapped_positive: np.ndarray,
    target_positive: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    if not (len(mapped_positive) == len(target_positive) == len(weights)):
        raise ValueError("Classification arrays must have equal lengths.")
    valid = np.isfinite(weights) & (weights > 0)
    mapped = mapped_positive[valid].astype(bool)
    target = target_positive[valid].astype(bool)
    w = weights[valid].astype(float)
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("Classification metric denominator is zero.")

    both_positive = float(np.sum(w[mapped & target]))
    mapped_only = float(np.sum(w[mapped & ~target]))
    target_only = float(np.sum(w[~mapped & target]))
    both_negative = float(np.sum(w[~mapped & ~target]))
    mapped_rate = (both_positive + mapped_only) / total
    target_rate = (both_positive + target_only) / total
    discordance = (mapped_only + target_only) / total
    observed_agreement = (both_positive + both_negative) / total
    expected_agreement = mapped_rate * target_rate + (1 - mapped_rate) * (1 - target_rate)
    kappa_denominator = 1 - expected_agreement
    cohen_kappa = (
        (observed_agreement - expected_agreement) / kappa_denominator
        if not math.isclose(kappa_denominator, 0.0)
        else math.nan
    )
    positive_agreement = safe_divide(2 * both_positive, 2 * both_positive + mapped_only + target_only)
    negative_agreement = safe_divide(2 * both_negative, 2 * both_negative + mapped_only + target_only)
    return {
        "both_positive": both_positive,
        "mapped_positive_target_negative": mapped_only,
        "mapped_negative_target_positive": target_only,
        "both_negative": both_negative,
        "denominator_measure": total,
        "observed_target_attainment_pct": 100 * target_rate,
        "mapped_source_attainment_pct": 100 * mapped_rate,
        "attainment_difference_pp": 100 * (mapped_rate - target_rate),
        "discordant_reclassification_pct": 100 * discordance,
        "cohen_kappa": cohen_kappa,
        "positive_agreement": positive_agreement,
        "negative_agreement": negative_agreement,
        "target_threshold_degenerate": bool(
            target_rate < DEGENERATE_POS_RATE or target_rate > (1 - DEGENERATE_POS_RATE)
        ),
    }


def threshold_rows_for_pair(
    cohort: DailyCohort,
    sample_idx: np.ndarray,
    survey_weights: np.ndarray,
    source: str,
    target: str,
    oof_seed: int,
) -> list[dict[str, object]]:
    x_all = cohort.daily_by_alg[source][sample_idx]
    y_all = cohort.daily_by_alg[target][sample_idx]
    ids_all = cohort.subject_ids[sample_idx]
    weights_all = survey_weights[sample_idx]
    mask = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(weights_all) & (weights_all > 0)
    x = x_all[mask]
    y = y_all[mask]
    ids = ids_all[mask]
    weights = weights_all[mask]
    prediction = grouped_oof_predictions(x, y, ids, oof_seed)

    rows: list[dict[str, object]] = []
    for threshold in FIXED_STEP_THRESHOLDS:
        mapped_positive = prediction >= threshold
        target_positive = y >= threshold
        for weighting, metric_weights, unit in (
            ("unweighted_primary", np.ones(len(y), dtype=float), "subjects"),
            ("WTMEC4YR_weighted_sensitivity", weights, "WTMEC4YR"),
        ):
            metrics = weighted_classification_metrics(mapped_positive, target_positive, metric_weights)
            rows.append(
                {
                    "exposure": "daily_steps",
                    "method": HEADLINE_METHOD,
                    "validation": "whole_cohort_subject_grouped_5fold_oof",
                    "source_algorithm": source,
                    "target_algorithm": target,
                    "threshold_target_steps": float(threshold),
                    "analysis_weighting": weighting,
                    "n_rows": int(len(y)),
                    "measure_unit": unit,
                    **metrics,
                    "note": (
                        "Mapped source is the grouped five-fold OOF isotonic prediction on the "
                        "target algorithm scale. Positive/negative agreement are symmetric: "
                        "2a/(2a+b+c) and 2d/(2d+b+c). Cohen kappa uses the observed agreement "
                        "minus agreement expected from the mapped/target marginals, divided by "
                        "one minus expected agreement; every cell and marginal is WTMEC4YR-"
                        "weighted in the weighted sensitivity. Attainment difference is mapped "
                        "minus observed target percentage points. WTMEC4YR changes evaluation "
                        "weights only; the D7 headline crosswalk fit remains unweighted."
                    ),
                }
            )
    return rows


def transport_key(row: dict[str, object], metric: str) -> tuple[str, str, str, str, str]:
    return (
        str(row["source_algorithm"]),
        str(row["target_algorithm"]),
        str(row["train_cycle"]),
        str(row["test_cycle"]),
        metric,
    )


def threshold_key(row: dict[str, object], metric: str) -> tuple[str, str, float, str, str]:
    return (
        str(row["source_algorithm"]),
        str(row["target_algorithm"]),
        float(row["threshold_target_steps"]),
        str(row["analysis_weighting"]),
        metric,
    )


def build_transport_ci_rows(
    point_rows: list[dict[str, object]],
    bootstrap_values: dict[tuple[str, str, str, str, str], list[float]],
    requested: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in point_rows:
        for metric in TRANSPORT_METRICS:
            key = transport_key(point, metric)
            values = bootstrap_values.get(key, [])
            rows.append(
                {
                    "source_algorithm": point["source_algorithm"],
                    "target_algorithm": point["target_algorithm"],
                    "train_cycle": point["train_cycle"],
                    "test_cycle": point["test_cycle"],
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci_low": finite_percentile(values, CI_LOW_Q),
                    "ci_high": finite_percentile(values, CI_HIGH_Q),
                    "bootstrap_success": len([value for value in values if math.isfinite(value)]),
                    "bootstrap_requested": requested,
                    "ci_method": (
                        "Percentile CI from independent PSU-within-stratum resampling of train "
                        "and test cycles; train crosswalk refitted in every replicate with no "
                        "test-cycle tuning."
                    ),
                }
            )
    return rows


def build_threshold_ci_rows(
    point_rows: list[dict[str, object]],
    bootstrap_values: dict[tuple[str, str, float, str, str], list[float]],
    requested: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in point_rows:
        for metric, unit in THRESHOLD_METRIC_UNITS.items():
            key = threshold_key(point, metric)
            values = bootstrap_values.get(key, [])
            rows.append(
                {
                    "source_algorithm": point["source_algorithm"],
                    "target_algorithm": point["target_algorithm"],
                    "threshold_target_steps": point["threshold_target_steps"],
                    "analysis_weighting": point["analysis_weighting"],
                    "metric": metric,
                    "metric_unit": unit,
                    "point_estimate": point[metric],
                    "ci_low": finite_percentile(values, CI_LOW_Q),
                    "ci_high": finite_percentile(values, CI_HIGH_Q),
                    "bootstrap_success": len([value for value in values if math.isfinite(value)]),
                    "bootstrap_requested": requested,
                    "ci_method": (
                        "Percentile CI from PSU-within-stratum design bootstrap; grouped OOF "
                        "isotonic mapping and the classification metric were recomputed in each "
                        "replicate."
                    ),
                }
            )
    return rows


def load_stage2_reclassification_audit(
    out_dir: Path,
    threshold_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Audit unweighted point estimates against the earlier post-crosswalk table."""

    path = out_dir / "tables" / "stage2_all7_reclassification_kappa.csv"
    if not path.exists():
        return {"available": False, "reason": f"not found: {path}"}
    current = {
        (
            row["source_algorithm"],
            row["target_algorithm"],
            float(row["threshold_target_steps"]),
        ): row
        for row in threshold_rows
        if row["analysis_weighting"] == "unweighted_primary"
    }
    differences: list[float] = []
    compared = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for prior in csv.DictReader(handle):
            if prior.get("stage") != "after_crosswalk_oof_isotonic":
                continue
            key = (
                prior["source_algorithm"],
                prior["target_algorithm"],
                float(prior["threshold_target_steps"]),
            )
            now = current.get(key)
            if now is None:
                continue
            comparisons = (
                (float(prior["source_or_converted_positive_pct"]) * 100, float(now["mapped_source_attainment_pct"])),
                (float(prior["target_positive_pct"]) * 100, float(now["observed_target_attainment_pct"])),
                (float(prior["discordant_pct"]) * 100, float(now["discordant_reclassification_pct"])),
                (float(prior["kappa"]), float(now["cohen_kappa"])),
            )
            differences.extend(abs(old - new) for old, new in comparisons if math.isfinite(old) and math.isfinite(new))
            compared += 1
    return {
        "available": True,
        "rows_compared": compared,
        "max_absolute_difference_after_scale_alignment": max(differences) if differences else math.nan,
        "note": "Percent columns were aligned to a 0-100 scale before comparison; kappa remained on its native scale.",
    }


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_summary(
    path: Path,
    cohort: DailyCohort,
    design_audit: list[dict[str, object]],
    transport_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    transport_ci_rows: list[dict[str, object]],
    threshold_ci_rows: list[dict[str, object]],
    bootstrap_reps: int,
    stage2_audit: dict[str, object],
) -> None:
    cycle_rows = [row for row in design_audit if row["scope"] == "cycle"]
    largest_penalties = sorted(
        transport_rows,
        key=lambda row: float(row["transport_penalty_mae_per_target_iqr"]),
        reverse=True,
    )
    primary_threshold = [
        row for row in threshold_rows if row["analysis_weighting"] == "unweighted_primary"
    ]
    nondegenerate_threshold = [
        row for row in primary_threshold if not bool(row["target_threshold_degenerate"])
    ]
    best_threshold = sorted(
        nondegenerate_threshold,
        key=lambda row: float(row["discordant_reclassification_pct"]),
    )
    worst_threshold = sorted(
        nondegenerate_threshold,
        key=lambda row: float(row["discordant_reclassification_pct"]),
        reverse=True,
    )
    min_transport_success = min(int(row["bootstrap_success"]) for row in transport_ci_rows)
    min_threshold_success = min(int(row["bootstrap_success"]) for row in threshold_ci_rows)

    lines = [
        "# Stage 6 Cross-Cycle Transport and Fixed-Threshold OOF Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Frozen D9 all-7 adult cohort: {cohort.subjects_n:,} participants / {cohort.valid_days_n:,} valid person-days.",
        f"- Bootstrap: {bootstrap_reps} PSU-within-stratum replicates; G and H cycles resampled independently.",
        "- Cross-cycle models are trained in one cycle, locked, and evaluated in the other; the within-test-cycle OOF result is a comparator, not a tuning step.",
        "- Threshold results use the same whole-cohort grouped five-fold OOF prediction for unweighted primary and WTMEC4YR-weighted sensitivity evaluation.",
        "- Weighted Cohen kappa uses WTMEC4YR-weighted observed agreement and weighted mapped/target marginals; positive and negative agreement are symmetric 2a/(2a+b+c) and 2d/(2d+b+c), not sensitivity and specificity.",
        "- Guardrail: aggregate outputs only; no individual predictions or bootstrap-replicate table is written.",
        "",
        "## Cycle Design Audit",
        "",
        "| cycle | participants | strata | stratum-PSU cells | WTMEC4YR weighted N |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cycle_rows:
        lines.append(
            f"| {row['cycle']} | {int(row['n_subjects']):,} | {int(row['n_strata'])} | "
            f"{int(row['n_stratum_psu_cells'])} | {float(row['weighted_n_wtmec4yr']):,.1f} |"
        )

    lines.extend(
        [
            "",
            "## Largest Cross-Cycle MAE Penalties",
            "",
            "Penalty = transported MAE minus the within-test-cycle grouped OOF MAE on the same test-cycle target scale.",
            "",
            "| rank | direction | pair | transported MAE/IQR | within-test OOF MAE/IQR | penalty MAE/IQR |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(largest_penalties[:10], start=1):
        lines.append(
            f"| {rank} | {row['train_cycle']} -> {row['test_cycle']} | "
            f"{row['source_algorithm']}->{row['target_algorithm']} | "
            f"{fmt(row['transport_mae_per_target_iqr'])} | "
            f"{fmt(row['within_test_oof_mae_per_target_iqr'])} | "
            f"{fmt(row['transport_penalty_mae_per_target_iqr'])} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-Threshold OOF Classification Examples (Unweighted Primary)",
            "",
            "Examples exclude rows whose observed target attainment is <2% or >98%, because low discordance and kappa can be uninformative at a degenerate threshold.",
            "",
            "| end | pair | target threshold | mapped attainment % | observed attainment % | difference pp | discordance % | kappa |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, selected in (("lowest discordance", best_threshold[:5]), ("highest discordance", worst_threshold[:5])):
        for row in selected:
            lines.append(
                f"| {label} | {row['source_algorithm']}->{row['target_algorithm']} | "
                f"{float(row['threshold_target_steps']):.0f} | "
                f"{fmt(row['mapped_source_attainment_pct'], 1)} | "
                f"{fmt(row['observed_target_attainment_pct'], 1)} | "
                f"{fmt(row['attainment_difference_pp'], 1)} | "
                f"{fmt(row['discordant_reclassification_pct'], 1)} | "
                f"{fmt(row['cohen_kappa'])} |"
            )

    lines.extend(
        [
            "",
            "## Quality Control",
            "",
            f"- Cross-cycle CI rows: {len(transport_ci_rows):,}; minimum successful replicates: {min_transport_success}/{bootstrap_reps}.",
            f"- Threshold CI rows: {len(threshold_ci_rows):,}; minimum successful replicates: {min_threshold_success}/{bootstrap_reps}.",
            f"- Earlier Stage 2 post-crosswalk point audit: `{json.dumps(stage2_audit, ensure_ascii=False)}`.",
            "",
            "## Outputs",
            "",
            f"- `outputs/tables/{TRANSPORT_POINT_FILE}`",
            f"- `outputs/tables/{TRANSPORT_CI_FILE}`",
            f"- `outputs/tables/{THRESHOLD_POINT_FILE}`",
            f"- `outputs/tables/{THRESHOLD_CI_FILE}`",
            f"- `outputs/tables/{SUMMARY_FILE}`",
            f"- `outputs/logs/{RUN_LOG_FILE}`",
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

    log("rebuilding the frozen D9 all-7 adult cohort")
    cohort = build_daily_cohort(args.raw_dir)
    if cohort.subjects_n != 8646 or cohort.valid_days_n != 57080:
        raise RuntimeError(
            f"Stage 6 cohort drift: expected 8646/57080, got {cohort.subjects_n}/{cohort.valid_days_n}."
        )
    log("aligning DEMO cycle, stratum, PSU, and WTMEC4YR")
    cycles, strata, psus, survey_weights, design_audit = align_design(cohort, args.raw_dir)

    pairs = list(itertools.permutations(ALGORITHMS, 2))
    cycle_global = {cycle: np.flatnonzero(cycles == cycle) for cycle in CYCLES}
    for cycle in CYCLES:
        if len(cycle_global[cycle]) < FOLDS:
            raise RuntimeError(f"Cycle {cycle} has too few participants for {FOLDS}-fold OOF.")

    log("computing 84 cross-cycle point rows")
    transport_points: list[dict[str, object]] = []
    directions = ((CYCLE_G, CYCLE_H), (CYCLE_H, CYCLE_G))
    for train_cycle, test_cycle in directions:
        for source, target in pairs:
            transport_points.append(
                cross_cycle_row(
                    cohort,
                    cycle_global[train_cycle],
                    cycle_global[test_cycle],
                    source,
                    target,
                    train_cycle,
                    test_cycle,
                    CROSSWALK_SEED,
                )
            )

    log("computing 252 fixed-threshold x weighting point rows")
    all_indices = np.arange(cohort.subjects_n)
    threshold_points: list[dict[str, object]] = []
    for source, target in pairs:
        threshold_points.extend(
            threshold_rows_for_pair(
                cohort,
                all_indices,
                survey_weights,
                source,
                target,
                CROSSWALK_SEED,
            )
        )

    rng = np.random.default_rng(args.seed)
    transport_bootstrap: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    threshold_bootstrap: dict[tuple[str, str, float, str, str], list[float]] = defaultdict(list)
    log(f"starting {args.bootstrap_reps} design-bootstrap replicates")
    for replicate in range(1, args.bootstrap_reps + 1):
        sampled_global: dict[str, np.ndarray] = {}
        for cycle in CYCLES:
            base_global = cycle_global[cycle]
            local = design_bootstrap_local_indices(strata[base_global], psus[base_global], rng)
            sampled_global[cycle] = base_global[local]

        for train_cycle, test_cycle in directions:
            for source, target in pairs:
                row = cross_cycle_row(
                    cohort,
                    sampled_global[train_cycle],
                    sampled_global[test_cycle],
                    source,
                    target,
                    train_cycle,
                    test_cycle,
                    args.seed + replicate,
                )
                for metric in TRANSPORT_METRICS:
                    value = float(row[metric])
                    if math.isfinite(value):
                        transport_bootstrap[transport_key(row, metric)].append(value)

        bootstrap_all = np.concatenate((sampled_global[CYCLE_G], sampled_global[CYCLE_H]))
        rng.shuffle(bootstrap_all)
        for source, target in pairs:
            rows = threshold_rows_for_pair(
                cohort,
                bootstrap_all,
                survey_weights,
                source,
                target,
                args.seed + replicate,
            )
            for row in rows:
                for metric in THRESHOLD_METRIC_UNITS:
                    value = float(row[metric])
                    if math.isfinite(value):
                        threshold_bootstrap[threshold_key(row, metric)].append(value)

        if (
            replicate == 1
            or replicate == args.bootstrap_reps
            or replicate % args.progress_every == 0
        ):
            log(f"bootstrap replicate {replicate}/{args.bootstrap_reps} complete")

    transport_ci = build_transport_ci_rows(
        transport_points, transport_bootstrap, args.bootstrap_reps
    )
    threshold_ci = build_threshold_ci_rows(
        threshold_points, threshold_bootstrap, args.bootstrap_reps
    )
    stage2_audit = load_stage2_reclassification_audit(args.out_dir, threshold_points)

    transport_fields = [
        "exposure",
        "method",
        "source_algorithm",
        "target_algorithm",
        "train_cycle",
        "test_cycle",
        "n_train",
        "n_test",
        *TRANSPORT_METRICS,
        "note",
    ]
    write_csv(out_tables / TRANSPORT_POINT_FILE, transport_points, transport_fields)
    write_csv(
        out_tables / TRANSPORT_CI_FILE,
        transport_ci,
        [
            "source_algorithm",
            "target_algorithm",
            "train_cycle",
            "test_cycle",
            "metric",
            "point_estimate",
            "ci_low",
            "ci_high",
            "bootstrap_success",
            "bootstrap_requested",
            "ci_method",
        ],
    )
    threshold_fields = [
        "exposure",
        "method",
        "validation",
        "source_algorithm",
        "target_algorithm",
        "threshold_target_steps",
        "analysis_weighting",
        "n_rows",
        "measure_unit",
        "both_positive",
        "mapped_positive_target_negative",
        "mapped_negative_target_positive",
        "both_negative",
        "denominator_measure",
        *THRESHOLD_METRIC_UNITS.keys(),
        "target_threshold_degenerate",
        "note",
    ]
    write_csv(out_tables / THRESHOLD_POINT_FILE, threshold_points, threshold_fields)
    write_csv(
        out_tables / THRESHOLD_CI_FILE,
        threshold_ci,
        [
            "source_algorithm",
            "target_algorithm",
            "threshold_target_steps",
            "analysis_weighting",
            "metric",
            "metric_unit",
            "point_estimate",
            "ci_low",
            "ci_high",
            "bootstrap_success",
            "bootstrap_requested",
            "ci_method",
        ],
    )
    write_summary(
        out_tables / SUMMARY_FILE,
        cohort,
        design_audit,
        transport_points,
        threshold_points,
        transport_ci,
        threshold_ci,
        args.bootstrap_reps,
        stage2_audit,
    )

    finished = datetime.now(timezone.utc)
    run_log = {
        "script": "scripts/python/build_stage6_cycle_threshold.py",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "progress_every": args.progress_every,
        "cohort_subjects": cohort.subjects_n,
        "cohort_valid_days": cohort.valid_days_n,
        "algorithms": ALGORITHMS,
        "directed_pairs": len(pairs),
        "cycles": CYCLES,
        "folds": FOLDS,
        "point_oof_seed": CROSSWALK_SEED,
        "design_audit": design_audit,
        "cross_cycle_point_rows": len(transport_points),
        "cross_cycle_ci_rows": len(transport_ci),
        "fixed_threshold_point_rows": len(threshold_points),
        "fixed_threshold_ci_rows": len(threshold_ci),
        "stage2_point_reproduction_audit": stage2_audit,
        "transport_definition": (
            "Fit unweighted isotonic crosswalk in the train cycle only; lock and test in the "
            "other cycle; compare with grouped five-fold OOF fitted within that test cycle."
        ),
        "threshold_definition": (
            "Whole-cohort grouped five-fold OOF mapped target classification at target-native "
            "7000/8000/10000 steps/day; unweighted primary and WTMEC4YR weighted sensitivity."
        ),
        "bootstrap_definition": (
            "Independent PSU-within-stratum resampling of G and H; refit mappings and recompute "
            "metrics in every replicate; percentile 95% intervals; no test tuning."
        ),
        "outputs": [
            f"outputs/tables/{TRANSPORT_POINT_FILE}",
            f"outputs/tables/{TRANSPORT_CI_FILE}",
            f"outputs/tables/{THRESHOLD_POINT_FILE}",
            f"outputs/tables/{THRESHOLD_CI_FILE}",
            f"outputs/tables/{SUMMARY_FILE}",
            f"outputs/logs/{RUN_LOG_FILE}",
        ],
        "guardrails": [
            "aggregate outputs only",
            "no individual prediction output",
            "no person-day or minute-level output",
            "no bootstrap replicate-level output",
            "no test-cycle tuning in transport validation",
            "does not overwrite Stage 2/3 point estimates or Stage 3 tier_final",
        ],
    }
    with (out_logs / RUN_LOG_FILE).open("w", encoding="utf-8") as handle:
        json.dump(run_log, handle, ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main()
