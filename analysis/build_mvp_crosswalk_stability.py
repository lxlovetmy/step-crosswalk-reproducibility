#!/usr/bin/env python3
"""Build MVP adept -> oak crosswalk and aggregate stability diagnostics.

Scope:
- Rebuild D9-valid adult exposures in memory using the already locked MVP
  exposure rules.
- Headline crosswalk method: isotonic regression.
- Robustness crosswalk methods: empirical quantile mapping and monotone PCHIP
  spline calibration.
- Validation: 5-fold out-of-fold predictions; no same-data fit/evaluate loop.
- Outputs are aggregate tables, SVG figures, and a run log only.
- No SEQN/person-day/minute-level outputs, no HbA1c/waist linkage, no models.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from build_mvp_exposure_qc import (
    STEP_FILES,
    TROIANO_FILE,
    collect_d9_valid_days,
    collect_wear_masks,
    process_step_algorithm,
    read_adult_seqns_from_demo,
    read_step_index,
)
from gate0_d9_wear_audit import locate_file, pct, read_day_minmax, summary_stats


HEADLINE_METHOD = "isotonic"
ROBUSTNESS_METHODS = ["quantile", "monotone_spline"]
ALL_METHODS = [HEADLINE_METHOD, *ROBUSTNESS_METHODS]
FOLDS = 5
RANDOM_SEED = 20260607
FIXED_STEP_THRESHOLDS = [7000.0, 8000.0, 10000.0]
# The manuscript and release configuration define eligible nonmissing
# subgroups as those containing at least 200 participants.
MIN_SUBGROUP_CURVE_N = 200


class CrosswalkModel(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass
class ExposureData:
    label: str
    x: np.ndarray
    y: np.ndarray
    age_group: np.ndarray
    sex: np.ndarray
    bmi_group: np.ndarray


@dataclass
class FittedCrosswalk:
    method: str
    model: CrosswalkModel
    x_min: float
    x_max: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x), dtype=float)


class QuantileMap:
    def __init__(self) -> None:
        self.x_values: np.ndarray | None = None
        self.x_probs: np.ndarray | None = None
        self.y_values: np.ndarray | None = None
        self.y_probs: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuantileMap":
        order_x = np.argsort(x, kind="mergesort")
        sorted_x = np.asarray(x[order_x], dtype=float)
        n = len(sorted_x)
        probs = (np.arange(n, dtype=float) + 0.5) / n
        unique_x, inverse = np.unique(sorted_x, return_inverse=True)
        prob_sum = np.bincount(inverse, weights=probs)
        prob_count = np.bincount(inverse)
        self.x_values = unique_x
        self.x_probs = prob_sum / prob_count

        sorted_y = np.sort(np.asarray(y, dtype=float))
        self.y_values = sorted_y
        self.y_probs = probs
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_values is None or self.x_probs is None or self.y_values is None or self.y_probs is None:
            raise RuntimeError("QuantileMap is not fitted.")
        p = np.interp(x, self.x_values, self.x_probs, left=self.x_probs[0], right=self.x_probs[-1])
        return np.interp(p, self.y_probs, self.y_values, left=self.y_values[0], right=self.y_values[-1])


class MonotoneSplineMap:
    def __init__(self) -> None:
        self.interpolator: PchipInterpolator | None = None
        self.x_knots: np.ndarray | None = None
        self.y_knots: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MonotoneSplineMap":
        probs = np.array(
            [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
             0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
             0.95, 0.99, 1.0],
            dtype=float,
        )
        x_knots = np.quantile(x, probs)
        y_knots = np.quantile(y, probs)
        unique_x: list[float] = []
        unique_y: list[float] = []
        for x_value, y_value in zip(x_knots, y_knots):
            if unique_x and math.isclose(float(x_value), unique_x[-1], rel_tol=0.0, abs_tol=1e-9):
                unique_y[-1] = max(unique_y[-1], float(y_value))
            else:
                unique_x.append(float(x_value))
                unique_y.append(float(y_value))
        if len(unique_x) < 2:
            unique_x = [float(np.min(x)) - 0.5, float(np.max(x)) + 0.5]
            unique_y = [float(np.mean(y)), float(np.mean(y))]
        self.x_knots = np.asarray(unique_x, dtype=float)
        self.y_knots = np.maximum.accumulate(np.asarray(unique_y, dtype=float))
        self.interpolator = PchipInterpolator(self.x_knots, self.y_knots, extrapolate=False)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.interpolator is None or self.x_knots is None or self.y_knots is None:
            raise RuntimeError("MonotoneSplineMap is not fitted.")
        clipped = np.clip(x, self.x_knots[0], self.x_knots[-1])
        predicted = np.asarray(self.interpolator(clipped), dtype=float)
        return np.maximum(predicted, 0.0)


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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_demo_bmi(raw_dir: Path) -> dict[str, dict[str, float]]:
    """Read adult age/sex/BMI into memory without writing person-level output."""

    r_raw_dir = json.dumps(str(raw_dir))
    code = f"""
suppressPackageStartupMessages(library(haven))
normalize_seqn <- function(x) as.character(as.integer(as.numeric(x)))
read_cycle <- function(cycle_dir, suffix) {{
  demo_path <- file.path({r_raw_dir}, cycle_dir, paste0("DEMO_", suffix, ".XPT"))
  bmx_path <- file.path({r_raw_dir}, cycle_dir, paste0("BMX_", suffix, ".XPT"))
  if (!file.exists(demo_path)) stop("missing DEMO file: ", demo_path)
  if (!file.exists(bmx_path)) stop("missing BMX file: ", bmx_path)
  demo <- read_xpt(demo_path)[, c("SEQN", "RIDAGEYR", "RIAGENDR")]
  bmx <- read_xpt(bmx_path)[, c("SEQN", "BMXBMI")]
  merged <- merge(demo, bmx, by = "SEQN", all.x = TRUE, sort = FALSE)
  merged
}}
out <- rbind(
  read_cycle("nhanes_2011_2012", "G"),
  read_cycle("nhanes_2013_2014", "H")
)
out <- out[!is.na(out$RIDAGEYR) & out$RIDAGEYR >= 20, ]
out$SEQN <- normalize_seqn(out$SEQN)
write.csv(out, row.names = FALSE, na = "")
"""
    completed = subprocess.run(
        ["Rscript", "--vanilla", "-e", code],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    covariates: dict[str, dict[str, float]] = {}
    for row in reader:
        seqn = row.get("SEQN", "").strip()
        if not seqn:
            continue
        covariates[seqn] = {
            "age": parse_float_or_nan(row.get("RIDAGEYR", "")),
            "sex_code": parse_float_or_nan(row.get("RIAGENDR", "")),
            "bmi": parse_float_or_nan(row.get("BMXBMI", "")),
        }
    return covariates


def parse_float_or_nan(value: str | None) -> float:
    if value is None:
        return math.nan
    stripped = value.strip()
    if not stripped:
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return math.nan


def age_group(age: float) -> str:
    if math.isnan(age):
        return "missing"
    if age < 40:
        return "20-39"
    if age < 60:
        return "40-59"
    return "60+"


def sex_group(sex_code: float) -> str:
    if sex_code == 1:
        return "male"
    if sex_code == 2:
        return "female"
    return "missing"


def bmi_group(bmi: float) -> str:
    if math.isnan(bmi):
        return "missing"
    if bmi < 25:
        return "<25"
    if bmi < 30:
        return "25-<30"
    return ">=30"


def rebuild_subject_records(raw_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    physio_dir = raw_dir / "physionet_v1.0.1"
    troiano_path = locate_file(physio_dir, TROIANO_FILE)
    step_paths = {
        algorithm: locate_file(physio_dir, filename)
        for algorithm, filename in STEP_FILES.items()
    }
    adult_seqns = read_adult_seqns_from_demo(raw_dir)
    covariates = read_demo_bmi(raw_dir)
    step_indices = {algorithm: read_step_index(path) for algorithm, path in step_paths.items()}
    step_common_keys = step_indices["adept"] & step_indices["oak"]
    minmax_by_seqn = read_day_minmax(troiano_path)
    _valid_days_by_subject, final_subjects, final_valid_days, alignment_rows = collect_d9_valid_days(
        troiano_path=troiano_path,
        step_common_keys=step_common_keys,
        minmax_by_seqn=minmax_by_seqn,
        adult_seqns=adult_seqns,
    )
    wear_masks = collect_wear_masks(troiano_path, final_valid_days)
    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]] = {}
    aggregate_by_algorithm: dict[str, dict[str, object]] = {}
    for algorithm, path in step_paths.items():
        metrics, aggregate = process_step_algorithm(algorithm, path, wear_masks)
        metrics_by_algorithm[algorithm] = metrics
        aggregate_by_algorithm[algorithm] = aggregate

    records: list[dict[str, object]] = []
    for seqn in final_subjects:
        covar = covariates.get(seqn, {"age": math.nan, "sex_code": math.nan, "bmi": math.nan})
        records.append(
            {
                "adept_daily": metrics_by_algorithm["adept"].get(seqn, {}).get("daily_steps_whole_day_sum_mean", math.nan),
                "oak_daily": metrics_by_algorithm["oak"].get(seqn, {}).get("daily_steps_whole_day_sum_mean", math.nan),
                "adept_cadence": metrics_by_algorithm["adept"].get(seqn, {}).get("peak30_cadence_mean_secondary", math.nan),
                "oak_cadence": metrics_by_algorithm["oak"].get(seqn, {}).get("peak30_cadence_mean_secondary", math.nan),
                "age_group": age_group(float(covar["age"])),
                "sex": sex_group(float(covar["sex_code"])),
                "bmi_group": bmi_group(float(covar["bmi"])),
                "has_bmi": not math.isnan(float(covar["bmi"])),
            }
        )
    rebuild_log = {
        "alignment_rows": alignment_rows,
        "denominator_subjects": len(final_subjects),
        "denominator_valid_days": len(final_valid_days),
        "algorithm_aggregates": aggregate_by_algorithm,
    }
    return records, rebuild_log


def exposure_data(records: list[dict[str, object]], label: str) -> ExposureData:
    if label == "daily_steps":
        x_name, y_name = "adept_daily", "oak_daily"
    elif label == "peak30_cadence":
        x_name, y_name = "adept_cadence", "oak_cadence"
    else:
        raise ValueError(f"Unknown exposure label: {label}")

    rows = [
        record for record in records
        if is_finite(float(record[x_name])) and is_finite(float(record[y_name]))
    ]
    return ExposureData(
        label=label,
        x=np.asarray([float(record[x_name]) for record in rows], dtype=float),
        y=np.asarray([float(record[y_name]) for record in rows], dtype=float),
        age_group=np.asarray([str(record["age_group"]) for record in rows], dtype=object),
        sex=np.asarray([str(record["sex"]) for record in rows], dtype=object),
        bmi_group=np.asarray([str(record["bmi_group"]) for record in rows], dtype=object),
    )


def is_finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def fit_crosswalk(method: str, x: np.ndarray, y: np.ndarray) -> FittedCrosswalk:
    if len(x) < 2:
        raise ValueError("Need at least two observations to fit crosswalk.")
    if method == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0)
        model.fit(x, y)
    elif method == "quantile":
        model = QuantileMap().fit(x, y)
    elif method == "monotone_spline":
        model = MonotoneSplineMap().fit(x, y)
    else:
        raise ValueError(f"Unknown crosswalk method: {method}")
    return FittedCrosswalk(method=method, model=model, x_min=float(np.min(x)), x_max=float(np.max(x)))


def cross_validated_predictions(data: ExposureData, method: str) -> np.ndarray:
    predictions = np.full(len(data.x), np.nan, dtype=float)
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in splitter.split(data.x):
        fitted = fit_crosswalk(method, data.x[train_idx], data.y[train_idx])
        predictions[test_idx] = fitted.predict(data.x[test_idx])
    return predictions


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return math.nan
    observed = float(np.mean(a == b))
    p_a = float(np.mean(a))
    p_b = float(np.mean(b))
    expected = p_a * p_b + (1.0 - p_a) * (1.0 - p_b)
    if math.isclose(1.0 - expected, 0.0):
        return math.nan
    return (observed - expected) / (1.0 - expected)


def residual_row(
    exposure: str,
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
    target_stats = summary_stats(target.tolist())
    rmse = math.sqrt(float(np.mean(residual ** 2))) if len(residual) else math.nan
    target_iqr = float(target_stats["p75"]) - float(target_stats["p25"]) if len(target) else math.nan
    return {
        "exposure": exposure,
        "method": method,
        "method_role": "headline" if method == HEADLINE_METHOD else "robustness",
        "scope_type": scope_type,
        "scope_level": scope_level,
        "n": len(target),
        "mean_error_pred_minus_oak": residual_stats["mean"],
        "median_error_pred_minus_oak": residual_stats["p50"],
        "error_sd": residual_stats["sd"],
        "error_p05": residual_stats["p05"],
        "error_p95": residual_stats["p95"],
        "mae": float(np.mean(abs_error)) if len(abs_error) else math.nan,
        "rmse": rmse,
        "absolute_error_p50": abs_stats["p50"],
        "absolute_error_p90": abs_stats["p90"],
        "absolute_error_p95": abs_stats["p95"],
        "target_oak_iqr": target_iqr,
        "mae_per_oak_iqr": float(np.mean(abs_error)) / target_iqr if target_iqr and target_iqr > 0 else math.nan,
    }


def build_residual_rows(data_by_exposure: dict[str, ExposureData]) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    oof_predictions: dict[tuple[str, str], np.ndarray] = {}
    for exposure, data in data_by_exposure.items():
        for method in ALL_METHODS:
            predicted = cross_validated_predictions(data, method)
            oof_predictions[(exposure, method)] = predicted
            rows.append(residual_row(exposure, method, "overall", "all", data.y, predicted))
    return rows, oof_predictions


def build_reclassification_rows(
    data: ExposureData,
    oof_predictions: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in FIXED_STEP_THRESHOLDS:
        oak_class = data.y >= threshold
        adept_raw_class = data.x >= threshold
        rows.append(classification_row(
            method="raw_same_numeric",
            method_role="pre_crosswalk_trivial_reference",
            stage="before_crosswalk",
            threshold=threshold,
            converted_or_raw=adept_raw_class,
            oak_class=oak_class,
            note="Same numeric fixed threshold on raw adept and raw oak; expected to be trivial because adept level is far lower.",
        ))
        for method in ALL_METHODS:
            predicted = oof_predictions[("daily_steps", method)]
            converted_class = predicted >= threshold
            rows.append(classification_row(
                method=method,
                method_role="headline" if method == HEADLINE_METHOD else "robustness",
                stage="after_crosswalk_oof",
                threshold=threshold,
                converted_or_raw=converted_class,
                oak_class=oak_class,
                note="Adept converted to oak scale with out-of-fold predictions, then classified at the fixed oak threshold.",
            ))
    return rows


def classification_row(
    method: str,
    method_role: str,
    stage: str,
    threshold: float,
    converted_or_raw: np.ndarray,
    oak_class: np.ndarray,
    note: str,
) -> dict[str, object]:
    discordant = converted_or_raw != oak_class
    n = len(oak_class)
    return {
        "exposure": "daily_steps",
        "threshold_oak_steps": threshold,
        "stage": stage,
        "method": method,
        "method_role": method_role,
        "n": n,
        "adept_or_converted_positive_n": int(np.sum(converted_or_raw)),
        "adept_or_converted_positive_pct": pct(int(np.sum(converted_or_raw)), n),
        "oak_positive_n": int(np.sum(oak_class)),
        "oak_positive_pct": pct(int(np.sum(oak_class)), n),
        "discordant_n": int(np.sum(discordant)),
        "discordant_pct": pct(int(np.sum(discordant)), n),
        "kappa": kappa(converted_or_raw, oak_class),
        "note": note,
    }


def group_arrays(data: ExposureData, group_type: str) -> np.ndarray:
    if group_type == "age_group":
        return data.age_group
    if group_type == "sex":
        return data.sex
    if group_type == "bmi_group":
        return data.bmi_group
    raise ValueError(f"Unknown group type: {group_type}")


def preferred_group_order(group_type: str) -> list[str]:
    if group_type == "age_group":
        return ["20-39", "40-59", "60+", "missing"]
    if group_type == "sex":
        return ["male", "female", "missing"]
    if group_type == "bmi_group":
        return ["<25", "25-<30", ">=30", "missing"]
    return []


def build_subgroup_count_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_n = len(records)
    bmi_missing_n = sum(1 for record in records if str(record["bmi_group"]) == "missing")
    rows.append({
        "group_type": "overall",
        "group_level": "all",
        "n_subjects": total_n,
        "pct_subjects": 1.0,
        "bmx_bmi_missing_n": bmi_missing_n,
        "note": "D9-valid adults in MVP crosswalk denominator.",
    })
    for group_type in ["age_group", "sex", "bmi_group"]:
        for level in preferred_group_order(group_type):
            n = sum(1 for record in records if str(record[group_type if group_type != "age_group" else "age_group"]) == level)
            if n == 0 and level != "missing":
                continue
            rows.append({
                "group_type": group_type,
                "group_level": level,
                "n_subjects": n,
                "pct_subjects": pct(n, total_n),
                "bmx_bmi_missing_n": bmi_missing_n if group_type == "bmi_group" and level == "missing" else "",
                "note": "BMI categories use <25, 25-<30, >=30; missing is counted but not treated as a standard BMI stratum." if group_type == "bmi_group" else "",
            })
    return rows


def build_subgroup_residual_rows(
    data_by_exposure: dict[str, ExposureData],
    oof_predictions: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for exposure, data in data_by_exposure.items():
        for method in ALL_METHODS:
            predicted = oof_predictions[(exposure, method)]
            for group_type in ["age_group", "sex", "bmi_group"]:
                groups = group_arrays(data, group_type)
                for level in preferred_group_order(group_type):
                    mask = groups == level
                    if not np.any(mask):
                        continue
                    rows.append(residual_row(exposure, method, group_type, level, data.y[mask], predicted[mask]))
    return rows


def build_crosswalk_table(data_by_exposure: dict[str, ExposureData]) -> tuple[list[dict[str, object]], dict[tuple[str, str], FittedCrosswalk]]:
    rows: list[dict[str, object]] = []
    fitted_models: dict[tuple[str, str], FittedCrosswalk] = {}
    for exposure, data in data_by_exposure.items():
        grid = np.unique(np.concatenate([
            np.linspace(float(np.quantile(data.x, 0.01)), float(np.quantile(data.x, 0.99)), 101),
            np.quantile(data.x, [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]),
        ]))
        grid = np.asarray(sorted(float(value) for value in grid if is_finite(float(value))), dtype=float)
        for method in ALL_METHODS:
            fitted = fit_crosswalk(method, data.x, data.y)
            fitted_models[(exposure, method)] = fitted
            predicted = fitted.predict(grid)
            for x_value, y_value in zip(grid, predicted):
                rows.append({
                    "exposure": exposure,
                    "method": method,
                    "method_role": "headline" if method == HEADLINE_METHOD else "robustness",
                    "adept_input": x_value,
                    "predicted_oak": y_value,
                    "grid_note": "Aggregate grid from observed adept quantiles plus p01-p99 sequence; no person-level rows.",
                })
    return rows, fitted_models


def build_threshold_equivalents(fitted_models: dict[tuple[str, str], FittedCrosswalk]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        fitted = fitted_models[("daily_steps", method)]
        x_grid = np.linspace(fitted.x_min, fitted.x_max, 4001)
        y_grid = fitted.predict(x_grid)
        for threshold in FIXED_STEP_THRESHOLDS:
            equivalent, status = inverse_monotone_threshold(x_grid, y_grid, threshold)
            rows.append({
                "exposure": "daily_steps",
                "method": method,
                "method_role": "headline" if method == HEADLINE_METHOD else "robustness",
                "oak_threshold_steps": threshold,
                "adept_equivalent_steps": equivalent,
                "inverse_status": status,
                "note": "Adept input where fitted adept->oak mapping first reaches the fixed oak threshold.",
            })
    return rows


def inverse_monotone_threshold(x_grid: np.ndarray, y_grid: np.ndarray, threshold: float) -> tuple[float, str]:
    if threshold <= float(np.nanmin(y_grid)):
        return float(x_grid[0]), "threshold_below_or_at_mapping_min"
    if threshold > float(np.nanmax(y_grid)):
        return float(x_grid[-1]), "threshold_above_mapping_max"
    hits = np.where(y_grid >= threshold)[0]
    if len(hits) == 0:
        return math.nan, "no_crossing"
    idx = int(hits[0])
    if idx == 0:
        return float(x_grid[0]), "crossing_at_grid_min"
    y0, y1 = float(y_grid[idx - 1]), float(y_grid[idx])
    x0, x1 = float(x_grid[idx - 1]), float(x_grid[idx])
    if math.isclose(y1, y0):
        return x1, "flat_crossing"
    fraction = (threshold - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0), "interpolated_crossing"


def build_curve_divergence_rows(
    data_by_exposure: dict[str, ExposureData],
    fitted_models: dict[tuple[str, str], FittedCrosswalk],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    divergence_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for exposure, data in data_by_exposure.items():
        x_grid = np.linspace(float(np.quantile(data.x, 0.05)), float(np.quantile(data.x, 0.95)), 101)
        for method in ALL_METHODS:
            full_pred = fitted_models[(exposure, method)].predict(x_grid)
            for group_type in ["age_group", "sex", "bmi_group"]:
                groups = group_arrays(data, group_type)
                subgroup_predictions: list[np.ndarray] = []
                subgroup_levels: list[str] = []
                for level in preferred_group_order(group_type):
                    if level == "missing":
                        continue
                    mask = groups == level
                    n_group = int(np.sum(mask))
                    if n_group < MIN_SUBGROUP_CURVE_N:
                        continue
                    subgroup_model = fit_crosswalk(method, data.x[mask], data.y[mask])
                    subgroup_pred = subgroup_model.predict(x_grid)
                    subgroup_predictions.append(subgroup_pred)
                    subgroup_levels.append(level)
                    diff = subgroup_pred - full_pred
                    abs_diff = np.abs(diff)
                    diff_stats = summary_stats(diff.tolist())
                    abs_stats = summary_stats(abs_diff.tolist())
                    divergence_rows.append({
                        "exposure": exposure,
                        "method": method,
                        "method_role": "headline" if method == HEADLINE_METHOD else "robustness",
                        "group_type": group_type,
                        "group_level": level,
                        "n_group": n_group,
                        "grid_n": len(x_grid),
                        "adept_grid_min": float(np.min(x_grid)),
                        "adept_grid_max": float(np.max(x_grid)),
                        "mean_subgroup_minus_full_predicted_oak": diff_stats["mean"],
                        "median_subgroup_minus_full_predicted_oak": diff_stats["p50"],
                        "max_abs_subgroup_minus_full_predicted_oak": float(np.max(abs_diff)),
                        "p95_abs_subgroup_minus_full_predicted_oak": abs_stats["p95"],
                        "note": "Subgroup-specific curve compared against the full-sample fitted curve on an aggregate adept grid.",
                    })
                    if exposure == "daily_steps" and method == HEADLINE_METHOD:
                        for x_value, full_value, subgroup_value in zip(x_grid, full_pred, subgroup_pred):
                            grid_rows.append({
                                "exposure": exposure,
                                "method": method,
                                "group_type": group_type,
                                "group_level": level,
                                "adept_input": float(x_value),
                                "predicted_oak_full_sample": float(full_value),
                                "predicted_oak_subgroup": float(subgroup_value),
                                "subgroup_minus_full": float(subgroup_value - full_value),
                            })
                if len(subgroup_predictions) >= 2:
                    matrix = np.vstack(subgroup_predictions)
                    spread = np.max(matrix, axis=0) - np.min(matrix, axis=0)
                    spread_stats = summary_stats(spread.tolist())
                    spread_rows.append({
                        "exposure": exposure,
                        "method": method,
                        "method_role": "headline" if method == HEADLINE_METHOD else "robustness",
                        "group_type": group_type,
                        "n_groups_compared": len(subgroup_levels),
                        "group_levels": "|".join(subgroup_levels),
                        "grid_n": len(x_grid),
                        "adept_grid_min": float(np.min(x_grid)),
                        "adept_grid_max": float(np.max(x_grid)),
                        "mean_between_subgroup_spread_predicted_oak": spread_stats["mean"],
                        "p50_between_subgroup_spread_predicted_oak": spread_stats["p50"],
                        "p95_between_subgroup_spread_predicted_oak": spread_stats["p95"],
                        "max_between_subgroup_spread_predicted_oak": spread_stats["max"],
                        "note": "At each adept-grid point, spread = max subgroup curve - min subgroup curve.",
                    })
    return divergence_rows, spread_rows, grid_rows


def build_go_kill_rows(
    residual_rows: list[dict[str, object]],
    reclass_rows: list[dict[str, object]],
    spread_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    headline_residual = next(
        row for row in residual_rows
        if row["exposure"] == "daily_steps" and row["method"] == HEADLINE_METHOD and row["scope_type"] == "overall"
    )
    rows.append({
        "domain": "daily_steps_headline_oof_residual",
        "readout": "calibration_residual_size",
        "value": headline_residual["mae"],
        "unit": "oak-scale steps/day MAE",
        "go_kill_relevance": "Lower residual supports clean crosswalk; large residual supports Go as residual inconsistency.",
        "locked_decision": "no",
    })
    for threshold in FIXED_STEP_THRESHOLDS:
        raw_row = next(
            row for row in reclass_rows
            if row["threshold_oak_steps"] == threshold and row["stage"] == "before_crosswalk"
        )
        converted_row = next(
            row for row in reclass_rows
            if row["threshold_oak_steps"] == threshold and row["stage"] == "after_crosswalk_oof" and row["method"] == HEADLINE_METHOD
        )
        rows.append({
            "domain": f"fixed_threshold_{int(threshold)}",
            "readout": "raw_vs_after_crosswalk_discordance",
            "value": f"raw_discordant_pct={fmt(float(raw_row['discordant_pct']) * 100, 2)}; after_crosswalk_discordant_pct={fmt(float(converted_row['discordant_pct']) * 100, 2)}; after_crosswalk_kappa={fmt(float(converted_row['kappa']), 3)}",
            "unit": "percent and kappa",
            "go_kill_relevance": "Key readout is after-crosswalk recovery; raw discordance is a trivial level-shift reference.",
            "locked_decision": "no",
        })
    daily_spreads = [
        row for row in spread_rows
        if row["exposure"] == "daily_steps" and row["method"] == HEADLINE_METHOD
    ]
    for row in daily_spreads:
        rows.append({
            "domain": f"subgroup_curve_spread_{row['group_type']}",
            "readout": "subgroup_curve_split",
            "value": f"p95_spread={fmt(float(row['p95_between_subgroup_spread_predicted_oak']), 1)}; max_spread={fmt(float(row['max_between_subgroup_spread_predicted_oak']), 1)}",
            "unit": "oak-scale steps/day",
            "go_kill_relevance": "Large between-subgroup curve spread is a Go signal under SAP §10; clean overlap supports Kill/downscope.",
            "locked_decision": "no",
        })
    return rows


def write_figures(
    out_figures: Path,
    data_by_exposure: dict[str, ExposureData],
    fitted_models: dict[tuple[str, str], FittedCrosswalk],
    oof_predictions: dict[tuple[str, str], np.ndarray],
    curve_grid_rows: list[dict[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_figures.parent / "logs" / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_figures.mkdir(parents=True, exist_ok=True)
    daily = data_by_exposure["daily_steps"]
    x_grid = np.linspace(float(np.quantile(daily.x, 0.01)), float(np.quantile(daily.x, 0.99)), 300)

    plt.figure(figsize=(7.8, 5.2))
    bin_edges = np.quantile(daily.x, np.linspace(0, 1, 21))
    bin_centers: list[float] = []
    bin_medians: list[float] = []
    for idx in range(len(bin_edges) - 1):
        if idx == len(bin_edges) - 2:
            mask = (daily.x >= bin_edges[idx]) & (daily.x <= bin_edges[idx + 1])
        else:
            mask = (daily.x >= bin_edges[idx]) & (daily.x < bin_edges[idx + 1])
        if np.any(mask):
            bin_centers.append(float(np.median(daily.x[mask])))
            bin_medians.append(float(np.median(daily.y[mask])))
    plt.plot(x_grid, fitted_models[("daily_steps", HEADLINE_METHOD)].predict(x_grid), color="#1f77b4", linewidth=2.2, label="isotonic")
    plt.scatter(bin_centers, bin_medians, color="#333333", s=22, label="20-bin median")
    plt.xlabel("Adept daily steps")
    plt.ylabel("Predicted oak daily steps")
    plt.title("Adept to oak daily steps crosswalk")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_figures / "mvp_crosswalk_daily_steps_headline_curve.svg")
    plt.close()

    residual = oof_predictions[("daily_steps", HEADLINE_METHOD)] - daily.y
    plt.figure(figsize=(7.8, 5.2))
    plt.hist(residual, bins=60, color="#4c78a8", alpha=0.86)
    plt.axvline(0, color="#222222", linewidth=1)
    plt.xlabel("Out-of-fold predicted oak - observed oak")
    plt.ylabel("Subjects")
    plt.title("Daily steps headline crosswalk residuals")
    plt.tight_layout()
    plt.savefig(out_figures / "mvp_crosswalk_daily_steps_oof_residual_hist.svg")
    plt.close()

    for group_type in ["age_group", "sex", "bmi_group"]:
        rows = [row for row in curve_grid_rows if row["group_type"] == group_type]
        if not rows:
            continue
        plt.figure(figsize=(7.8, 5.2))
        x_values = sorted({float(row["adept_input"]) for row in rows})
        full_by_x = {
            float(row["adept_input"]): float(row["predicted_oak_full_sample"])
            for row in rows
        }
        plt.plot(x_values, [full_by_x[x] for x in x_values], color="#111111", linewidth=2.4, label="full sample")
        for level in preferred_group_order(group_type):
            level_rows = [row for row in rows if row["group_level"] == level]
            if not level_rows:
                continue
            level_rows = sorted(level_rows, key=lambda row: float(row["adept_input"]))
            plt.plot(
                [float(row["adept_input"]) for row in level_rows],
                [float(row["predicted_oak_subgroup"]) for row in level_rows],
                linewidth=1.7,
                label=level,
            )
        plt.xlabel("Adept daily steps")
        plt.ylabel("Predicted oak daily steps")
        plt.title(f"Daily steps subgroup crosswalk curves: {group_type}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_figures / f"mvp_crosswalk_daily_steps_subgroup_curves_{group_type}.svg")
        plt.close()


def write_summary(
    path: Path,
    rebuild_log: dict[str, object],
    residual_rows: list[dict[str, object]],
    reclass_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    subgroup_count_rows: list[dict[str, object]],
    spread_rows: list[dict[str, object]],
    go_kill_rows: list[dict[str, object]],
) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    denom_subjects = rebuild_log["denominator_subjects"]
    denom_days = rebuild_log["denominator_valid_days"]
    headline_daily = next(
        row for row in residual_rows
        if row["exposure"] == "daily_steps" and row["method"] == HEADLINE_METHOD and row["scope_type"] == "overall"
    )
    headline_cadence = next(
        row for row in residual_rows
        if row["exposure"] == "peak30_cadence" and row["method"] == HEADLINE_METHOD and row["scope_type"] == "overall"
    )
    bmi_count = next(row for row in subgroup_count_rows if row["group_type"] == "overall")
    lines = [
        "# MVP Crosswalk Stability Summary",
        "",
        f"- Generated: {generated}",
        "- Scope: adept -> oak crosswalk, residual/stability diagnostics, and fixed-threshold agreement only.",
        "- Headline method: isotonic regression; robustness methods: quantile mapping and monotone spline calibration.",
        f"- Validation: {FOLDS}-fold out-of-fold predictions for residuals and post-crosswalk reclassification.",
        f"- Denominator: {denom_subjects} D9-valid adults, {denom_days} valid person-days rebuilt in memory.",
        f"- BMI cutpoints: <25, 25-<30, >=30; BMXBMI missing n={bmi_count['bmx_bmi_missing_n']}.",
        "- Guardrail: no SEQN/person-day/minute-level table exported; no HbA1c/waist linkage and no exposure-outcome models.",
        "",
        "## Overall OOF Residuals",
        "",
        "| exposure | method | role | n | bias | MAE | RMSE | abs p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in residual_rows:
        if row["scope_type"] != "overall":
            continue
        lines.append(
            "| {exposure} | {method} | {role} | {n} | {bias} | {mae} | {rmse} | {p95} |".format(
                exposure=row["exposure"],
                method=row["method"],
                role=row["method_role"],
                n=row["n"],
                bias=fmt(float(row["mean_error_pred_minus_oak"]), 2),
                mae=fmt(float(row["mae"]), 2),
                rmse=fmt(float(row["rmse"]), 2),
                p95=fmt(float(row["absolute_error_p95"]), 2),
            )
        )
    lines.extend([
        "",
        "## Fixed Threshold Agreement",
        "",
        "| threshold | stage | method | discordant % | kappa | adept/converted positive % | oak positive % |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for threshold in FIXED_STEP_THRESHOLDS:
        selected = [
            row for row in reclass_rows
            if row["threshold_oak_steps"] == threshold
            and (row["stage"] == "before_crosswalk" or row["method"] == HEADLINE_METHOD)
        ]
        for row in selected:
            lines.append(
                "| {thr} | {stage} | {method} | {discord} | {kappa} | {apct} | {opct} |".format(
                    thr=int(threshold),
                    stage=row["stage"],
                    method=row["method"],
                    discord=fmt(float(row["discordant_pct"]) * 100, 2),
                    kappa=fmt(float(row["kappa"]), 3),
                    apct=fmt(float(row["adept_or_converted_positive_pct"]) * 100, 2),
                    opct=fmt(float(row["oak_positive_pct"]) * 100, 2),
                )
            )
    lines.extend([
        "",
        "## Fixed Threshold Equivalents",
        "",
        "| oak threshold | method | adept equivalent | status |",
        "|---:|---|---:|---|",
    ])
    for row in threshold_rows:
        if row["method"] != HEADLINE_METHOD:
            continue
        lines.append(
            f"| {int(float(row['oak_threshold_steps']))} | {row['method']} | {fmt(float(row['adept_equivalent_steps']), 1)} | {row['inverse_status']} |"
        )
    lines.extend([
        "",
        "## Subgroup Curve Spread",
        "",
        "| group type | method | p95 spread | max spread | groups |",
        "|---|---|---:|---:|---|",
    ])
    for row in spread_rows:
        if row["exposure"] != "daily_steps" or row["method"] != HEADLINE_METHOD:
            continue
        lines.append(
            "| {group_type} | {method} | {p95} | {max_spread} | {groups} |".format(
                group_type=row["group_type"],
                method=row["method"],
                p95=fmt(float(row["p95_between_subgroup_spread_predicted_oak"]), 1),
                max_spread=fmt(float(row["max_between_subgroup_spread_predicted_oak"]), 1),
                groups=row["group_levels"],
            )
        )
    lines.extend([
        "",
        "## Go/Kill Readout",
        "",
        "- This is a descriptive readout for audit; it does not lock a new method decision.",
        f"- Headline daily residual: MAE {fmt(float(headline_daily['mae']), 1)} oak-scale steps/day, RMSE {fmt(float(headline_daily['rmse']), 1)}.",
        f"- Secondary cadence residual: MAE {fmt(float(headline_cadence['mae']), 2)} oak-scale steps/min, RMSE {fmt(float(headline_cadence['rmse']), 2)}.",
        "- Raw fixed-threshold discordance is retained only as the expected level-shift reference; the post-crosswalk rows are the interpretable rows.",
    ])
    for row in go_kill_rows:
        lines.append(f"- {row['domain']}: {row['value']} ({row['go_kill_relevance']})")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_tables = out_dir / "tables"
    out_figures = out_dir / "figures"
    out_logs = out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    records, rebuild_log = rebuild_subject_records(raw_dir)
    data_by_exposure = {
        "daily_steps": exposure_data(records, "daily_steps"),
        "peak30_cadence": exposure_data(records, "peak30_cadence"),
    }
    residual_rows, oof_predictions = build_residual_rows(data_by_exposure)
    subgroup_residual_rows = build_subgroup_residual_rows(data_by_exposure, oof_predictions)
    reclassification_rows = build_reclassification_rows(data_by_exposure["daily_steps"], oof_predictions)
    crosswalk_rows, fitted_models = build_crosswalk_table(data_by_exposure)
    threshold_rows = build_threshold_equivalents(fitted_models)
    divergence_rows, spread_rows, curve_grid_rows = build_curve_divergence_rows(data_by_exposure, fitted_models)
    subgroup_count_rows = build_subgroup_count_rows(records)
    go_kill_rows = build_go_kill_rows(residual_rows, reclassification_rows, spread_rows)

    write_csv(
        out_tables / "mvp_crosswalk_oof_residuals.csv",
        residual_rows,
        [
            "exposure", "method", "method_role", "scope_type", "scope_level", "n",
            "mean_error_pred_minus_oak", "median_error_pred_minus_oak", "error_sd",
            "error_p05", "error_p95", "mae", "rmse", "absolute_error_p50",
            "absolute_error_p90", "absolute_error_p95", "target_oak_iqr", "mae_per_oak_iqr",
        ],
    )
    write_csv(
        out_tables / "mvp_crosswalk_subgroup_fullsample_oof_residuals.csv",
        subgroup_residual_rows,
        [
            "exposure", "method", "method_role", "scope_type", "scope_level", "n",
            "mean_error_pred_minus_oak", "median_error_pred_minus_oak", "error_sd",
            "error_p05", "error_p95", "mae", "rmse", "absolute_error_p50",
            "absolute_error_p90", "absolute_error_p95", "target_oak_iqr", "mae_per_oak_iqr",
        ],
    )
    write_csv(
        out_tables / "mvp_crosswalk_reclassification_kappa.csv",
        reclassification_rows,
        [
            "exposure", "threshold_oak_steps", "stage", "method", "method_role", "n",
            "adept_or_converted_positive_n", "adept_or_converted_positive_pct",
            "oak_positive_n", "oak_positive_pct", "discordant_n", "discordant_pct",
            "kappa", "note",
        ],
    )
    write_csv(
        out_tables / "mvp_crosswalk_table.csv",
        crosswalk_rows,
        ["exposure", "method", "method_role", "adept_input", "predicted_oak", "grid_note"],
    )
    write_csv(
        out_tables / "mvp_crosswalk_fixed_threshold_equivalents.csv",
        threshold_rows,
        ["exposure", "method", "method_role", "oak_threshold_steps", "adept_equivalent_steps", "inverse_status", "note"],
    )
    write_csv(
        out_tables / "mvp_crosswalk_subgroup_counts.csv",
        subgroup_count_rows,
        ["group_type", "group_level", "n_subjects", "pct_subjects", "bmx_bmi_missing_n", "note"],
    )
    write_csv(
        out_tables / "mvp_crosswalk_subgroup_curve_divergence.csv",
        divergence_rows,
        [
            "exposure", "method", "method_role", "group_type", "group_level", "n_group", "grid_n",
            "adept_grid_min", "adept_grid_max", "mean_subgroup_minus_full_predicted_oak",
            "median_subgroup_minus_full_predicted_oak", "max_abs_subgroup_minus_full_predicted_oak",
            "p95_abs_subgroup_minus_full_predicted_oak", "note",
        ],
    )
    write_csv(
        out_tables / "mvp_crosswalk_subgroup_curve_spread.csv",
        spread_rows,
        [
            "exposure", "method", "method_role", "group_type", "n_groups_compared", "group_levels",
            "grid_n", "adept_grid_min", "adept_grid_max",
            "mean_between_subgroup_spread_predicted_oak", "p50_between_subgroup_spread_predicted_oak",
            "p95_between_subgroup_spread_predicted_oak", "max_between_subgroup_spread_predicted_oak", "note",
        ],
    )
    write_csv(
        out_tables / "mvp_crosswalk_subgroup_curve_grid_daily_headline.csv",
        curve_grid_rows,
        [
            "exposure", "method", "group_type", "group_level", "adept_input",
            "predicted_oak_full_sample", "predicted_oak_subgroup", "subgroup_minus_full",
        ],
    )
    write_csv(
        out_tables / "mvp_crosswalk_go_kill_readout.csv",
        go_kill_rows,
        ["domain", "readout", "value", "unit", "go_kill_relevance", "locked_decision"],
    )

    write_figures(out_figures, data_by_exposure, fitted_models, oof_predictions, curve_grid_rows)
    write_summary(
        out_tables / "MVP_CROSSWALK_STABILITY_SUMMARY.md",
        rebuild_log,
        residual_rows,
        reclassification_rows,
        threshold_rows,
        subgroup_count_rows,
        spread_rows,
        go_kill_rows,
    )

    finished = datetime.now(timezone.utc)
    log = {
        "script": "scripts/python/build_mvp_crosswalk_stability.py",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "headline_method": HEADLINE_METHOD,
        "robustness_methods": ROBUSTNESS_METHODS,
        "folds": FOLDS,
        "random_seed": RANDOM_SEED,
        "denominator_subjects": rebuild_log["denominator_subjects"],
        "denominator_valid_days": rebuild_log["denominator_valid_days"],
        "outputs": [
            "outputs/tables/MVP_CROSSWALK_STABILITY_SUMMARY.md",
            "outputs/tables/mvp_crosswalk_oof_residuals.csv",
            "outputs/tables/mvp_crosswalk_subgroup_fullsample_oof_residuals.csv",
            "outputs/tables/mvp_crosswalk_reclassification_kappa.csv",
            "outputs/tables/mvp_crosswalk_table.csv",
            "outputs/tables/mvp_crosswalk_fixed_threshold_equivalents.csv",
            "outputs/tables/mvp_crosswalk_subgroup_counts.csv",
            "outputs/tables/mvp_crosswalk_subgroup_curve_divergence.csv",
            "outputs/tables/mvp_crosswalk_subgroup_curve_spread.csv",
            "outputs/tables/mvp_crosswalk_subgroup_curve_grid_daily_headline.csv",
            "outputs/tables/mvp_crosswalk_go_kill_readout.csv",
            "outputs/figures/mvp_crosswalk_daily_steps_headline_curve.svg",
            "outputs/figures/mvp_crosswalk_daily_steps_oof_residual_hist.svg",
            "outputs/figures/mvp_crosswalk_daily_steps_subgroup_curves_age_group.svg",
            "outputs/figures/mvp_crosswalk_daily_steps_subgroup_curves_sex.svg",
            "outputs/figures/mvp_crosswalk_daily_steps_subgroup_curves_bmi_group.svg",
            "outputs/logs/mvp_crosswalk_stability_run.json",
        ],
        "guardrails": [
            "in-memory subject-level exposure reconstruction only",
            "aggregate tables and no-ID SVG figures only",
            "no SEQN/person-day/minute-level outputs",
            "no HbA1c or waist linkage",
            "no exposure-outcome models",
            "no new SAP method lock",
        ],
    }
    with (out_logs / "mvp_crosswalk_stability_run.json").open("w", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
