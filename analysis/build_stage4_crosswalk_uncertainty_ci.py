#!/usr/bin/env python3
"""Stage 4: bootstrap uncertainty intervals for crosswalk diagnostics.

This is a reporting-compliance add-on for MSSE. It does not change the locked
Stage 2/3 point estimates, tier cutpoints, featured pairs, or SAP decisions.

Outputs are aggregate CI tables only. No SEQN/person-day/minute rows,
individual OOF predictions, or bootstrap replicate-level tables are written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from build_mvp_crosswalk_stability import (
    FOLDS,
    HEADLINE_METHOD,
    MIN_SUBGROUP_CURVE_N,
    RANDOM_SEED as STAGE2_CROSSWALK_SEED,
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
from gate0_d9_wear_audit import summary_stats


RANDOM_SEED_STAGE4 = 20260609
DEFAULT_BOOTSTRAP_REPS = 300
CI_LOW_Q = 2.5
CI_HIGH_Q = 97.5


@dataclass(frozen=True)
class MetricKey:
    metric_family: str
    source_algorithm: str
    target_algorithm: str
    method: str
    group_type: str
    group_level: str
    metric: str


def log(message: str) -> None:
    print(f"[stage4-b4 {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_STAGE4)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def parse_float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_design_info(raw_dir: Path) -> dict[str, dict[str, float]]:
    """Read DEMO design fields into memory without writing person-level output."""

    r_raw_dir = json.dumps(str(raw_dir))
    code = f"""
suppressPackageStartupMessages(library(haven))
normalize_seqn <- function(x) as.character(as.integer(as.numeric(x)))
read_cycle <- function(cycle_dir, suffix) {{
  demo_path <- file.path({r_raw_dir}, cycle_dir, paste0("DEMO_", suffix, ".XPT"))
  if (!file.exists(demo_path)) stop("missing DEMO file: ", demo_path)
  demo <- read_xpt(demo_path)[, c("SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC2YR")]
  demo
}}
out <- rbind(
  read_cycle("nhanes_2011_2012", "G"),
  read_cycle("nhanes_2013_2014", "H")
)
out$SEQN <- normalize_seqn(out$SEQN)
out$WTMEC4YR <- out$WTMEC2YR / 2
write.csv(out[, c("SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC4YR")], row.names = FALSE, na = "")
"""
    completed = subprocess.run(
        ["Rscript", "--vanilla", "-e", code],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    design: dict[str, dict[str, float]] = {}
    for row in reader:
        seqn = row.get("SEQN", "").strip()
        if not seqn:
            continue
        design[seqn] = {
            "stratum": parse_float_or_nan(row.get("SDMVSTRA")),
            "psu": parse_float_or_nan(row.get("SDMVPSU")),
            "weight": parse_float_or_nan(row.get("WTMEC4YR")),
        }
    return design


def align_design(cohort: DailyCohort, raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    design = read_design_info(raw_dir)
    strata: list[float] = []
    psu: list[float] = []
    weights: list[float] = []
    missing = 0
    for seqn in cohort.subject_ids:
        row = design.get(str(seqn))
        if row is None:
            missing += 1
            strata.append(math.nan)
            psu.append(math.nan)
            weights.append(math.nan)
            continue
        strata.append(float(row["stratum"]))
        psu.append(float(row["psu"]))
        weights.append(float(row["weight"]))
    strata_arr = np.asarray(strata, dtype=float)
    psu_arr = np.asarray(psu, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    valid = ~np.isnan(strata_arr) & ~np.isnan(psu_arr) & ~np.isnan(weight_arr) & (weight_arr > 0)
    if not np.all(valid):
        raise RuntimeError(f"Missing/invalid DEMO design data for {int(np.sum(~valid))} cohort subjects.")

    rows: list[dict[str, object]] = []
    for stratum in sorted(np.unique(strata_arr).tolist()):
        mask = strata_arr == stratum
        rows.append(
            {
                "scope": "design_stratum",
                "stratum": int(stratum),
                "n_subjects": int(np.sum(mask)),
                "n_psu": int(len(np.unique(psu_arr[mask]))),
                "weighted_n_wtmec4yr": float(np.sum(weight_arr[mask])),
                "note": "D9-valid adult all-7 crosswalk cohort subjects in this DEMO stratum.",
            }
        )
    rows.append(
        {
            "scope": "design_overall",
            "stratum": "all",
            "n_subjects": int(len(strata_arr)),
            "n_psu": int(len(set(zip(strata_arr.tolist(), psu_arr.tolist())))),
            "weighted_n_wtmec4yr": float(np.sum(weight_arr)),
            "note": f"All cohort subjects; invalid design rows={missing}. WTMEC4YR retained for design audit; crosswalk point estimates remain unweighted under SAP D7.",
        }
    )
    return strata_arr, psu_arr, weight_arr, rows


def design_bootstrap_indices(strata: np.ndarray, psu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample PSUs with replacement within each stratum."""

    indices: list[np.ndarray] = []
    for stratum in sorted(np.unique(strata).tolist()):
        stratum_mask = strata == stratum
        psus = np.asarray(sorted(np.unique(psu[stratum_mask]).tolist()), dtype=float)
        sampled_psus = rng.choice(psus, size=len(psus), replace=True)
        for sampled_psu in sampled_psus:
            indices.append(np.flatnonzero(stratum_mask & (psu == sampled_psu)))
    if not indices:
        raise RuntimeError("No bootstrap indices sampled.")
    out = np.concatenate(indices)
    rng.shuffle(out)
    return out


def target_iqr(values: np.ndarray) -> float:
    stats = summary_stats(values.tolist())
    return float(stats["p75"]) - float(stats["p25"])


def grouped_cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(x), np.nan, dtype=float)
    unique_subjects = np.asarray(sorted(set(subject_ids.tolist())), dtype=object)
    n_splits = min(FOLDS, len(unique_subjects))
    if n_splits < 2:
        fitted = fit_crosswalk(HEADLINE_METHOD, x, y)
        return fitted.predict(x)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_unique_idx, test_unique_idx in splitter.split(unique_subjects):
        train_subjects = set(unique_subjects[train_unique_idx].tolist())
        test_subjects = set(unique_subjects[test_unique_idx].tolist())
        train_mask = np.asarray([subject in train_subjects for subject in subject_ids], dtype=bool)
        test_mask = np.asarray([subject in test_subjects for subject in subject_ids], dtype=bool)
        fitted = fit_crosswalk(HEADLINE_METHOD, x[train_mask], y[train_mask])
        predictions[test_mask] = fitted.predict(x[test_mask])
    return predictions


def metric_row(
    key: MetricKey,
    value: float,
    n: int,
    target_iqr_value: float,
    note: str,
) -> dict[str, object]:
    return {
        "metric_family": key.metric_family,
        "source_algorithm": key.source_algorithm,
        "target_algorithm": key.target_algorithm,
        "method": key.method,
        "group_type": key.group_type,
        "group_level": key.group_level,
        "metric": key.metric,
        "value": float(value),
        "n": int(n),
        "target_iqr": float(target_iqr_value),
        "note": note,
    }


def add_metric(
    rows: list[dict[str, object]],
    source: str,
    target: str,
    family: str,
    group_type: str,
    group_level: str,
    metric: str,
    value: float,
    n: int,
    target_iqr_value: float,
    note: str,
) -> None:
    if math.isnan(float(value)) or math.isinf(float(value)):
        return
    key = MetricKey(family, source, target, HEADLINE_METHOD, group_type, group_level, metric)
    rows.append(metric_row(key, float(value), n, target_iqr_value, note))


def compute_pair_metrics(
    cohort: DailyCohort,
    sample_idx: np.ndarray,
    source: str,
    target: str,
    seed: int,
) -> list[dict[str, object]]:
    x_all = cohort.daily_by_alg[source][sample_idx]
    y_all = cohort.daily_by_alg[target][sample_idx]
    subject_ids = cohort.subject_ids[sample_idx]
    age_groups = cohort.age_group[sample_idx]
    sex_groups = cohort.sex[sample_idx]
    bmi_groups = cohort.bmi_group[sample_idx]

    mask = ~np.isnan(x_all) & ~np.isnan(y_all)
    x = x_all[mask]
    y = y_all[mask]
    ids = subject_ids[mask]
    groups_by_type = {
        "age_group": age_groups[mask],
        "sex": sex_groups[mask],
        "bmi_group": bmi_groups[mask],
    }
    n = len(y)
    if n < 10:
        return []
    iqr = target_iqr(y)
    rows: list[dict[str, object]] = []

    predicted = grouped_cross_validated_predictions(x, y, ids, seed)
    residual = predicted - y
    abs_error = np.abs(residual)
    mae = float(np.mean(abs_error))
    rmse = math.sqrt(float(np.mean(residual**2)))
    abs_p95 = float(np.percentile(abs_error, 95))
    add_metric(rows, source, target, "fullsample_oof_residual", "overall", "all", "mae", mae, n, iqr, "5-fold grouped OOF isotonic residual on target scale.")
    add_metric(rows, source, target, "fullsample_oof_residual", "overall", "all", "rmse", rmse, n, iqr, "5-fold grouped OOF isotonic residual on target scale.")
    add_metric(rows, source, target, "fullsample_oof_residual", "overall", "all", "abs_error_p95", abs_p95, n, iqr, "5-fold grouped OOF isotonic residual on target scale.")
    add_metric(rows, source, target, "fullsample_oof_residual", "overall", "all", "mae_per_target_iqr", safe_divide(mae, iqr), n, iqr, "MAE normalized by target algorithm IQR.")

    x_grid = np.linspace(float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95)), 101)
    full_model = fit_crosswalk(HEADLINE_METHOD, x, y)
    full_pred = full_model.predict(x_grid)

    for group_type, group_values in groups_by_type.items():
        subgroup_predictions: list[np.ndarray] = []
        subgroup_levels: list[str] = []
        subgroup_ns: list[int] = []
        for level in preferred_group_order(group_type):
            if level == "missing":
                continue
            level_mask = group_values == level
            n_group = int(np.sum(level_mask))
            if n_group < MIN_SUBGROUP_CURVE_N:
                continue
            subgroup_model = fit_crosswalk(HEADLINE_METHOD, x[level_mask], y[level_mask])
            subgroup_pred = subgroup_model.predict(x_grid)
            subgroup_predictions.append(subgroup_pred)
            subgroup_levels.append(level)
            subgroup_ns.append(n_group)

            abs_diff = np.abs(subgroup_pred - full_pred)
            p95_abs = float(np.percentile(abs_diff, 95))
            max_abs = float(np.max(abs_diff))
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_divergence",
                group_type,
                level,
                "p95_abs_subgroup_minus_full_predicted_target",
                p95_abs,
                n_group,
                iqr,
                "Subgroup-specific isotonic curve compared with full-sample curve.",
            )
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_divergence",
                group_type,
                level,
                "p95_abs_subgroup_minus_full_predicted_target_over_target_iqr",
                safe_divide(p95_abs, iqr),
                n_group,
                iqr,
                "Subgroup-specific isotonic curve divergence normalized by target IQR.",
            )
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_divergence",
                group_type,
                level,
                "max_abs_subgroup_minus_full_predicted_target",
                max_abs,
                n_group,
                iqr,
                "Subgroup-specific isotonic curve compared with full-sample curve.",
            )
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_divergence",
                group_type,
                level,
                "max_abs_subgroup_minus_full_predicted_target_over_target_iqr",
                safe_divide(max_abs, iqr),
                n_group,
                iqr,
                "Subgroup-specific isotonic curve divergence normalized by target IQR.",
            )

        if len(subgroup_predictions) >= 2:
            matrix = np.vstack(subgroup_predictions)
            spread = np.max(matrix, axis=0) - np.min(matrix, axis=0)
            p95_spread = float(np.percentile(spread, 95))
            max_spread = float(np.max(spread))
            group_level = "|".join(subgroup_levels)
            n_for_row = int(sum(subgroup_ns))
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_spread",
                group_type,
                group_level,
                "p95_between_subgroup_spread_predicted_target",
                p95_spread,
                n_for_row,
                iqr,
                "At each source-grid point, spread = max subgroup curve - min subgroup curve.",
            )
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_spread",
                group_type,
                group_level,
                "p95_between_subgroup_spread_predicted_target_over_target_iqr",
                safe_divide(p95_spread, iqr),
                n_for_row,
                iqr,
                "Between-subgroup curve spread normalized by target IQR.",
            )
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_spread",
                group_type,
                group_level,
                "max_between_subgroup_spread_predicted_target",
                max_spread,
                n_for_row,
                iqr,
                "At each source-grid point, spread = max subgroup curve - min subgroup curve.",
            )
            add_metric(
                rows,
                source,
                target,
                "subgroup_curve_spread",
                group_type,
                group_level,
                "max_between_subgroup_spread_predicted_target_over_target_iqr",
                safe_divide(max_spread, iqr),
                n_for_row,
                iqr,
                "Between-subgroup curve spread normalized by target IQR.",
            )

    return rows


def key_from_row(row: dict[str, object]) -> MetricKey:
    return MetricKey(
        str(row["metric_family"]),
        str(row["source_algorithm"]),
        str(row["target_algorithm"]),
        str(row["method"]),
        str(row["group_type"]),
        str(row["group_level"]),
        str(row["metric"]),
    )


def finite_percentile(values: list[float], q: float) -> float:
    clean = np.asarray([v for v in values if not math.isnan(v) and not math.isinf(v)], dtype=float)
    if len(clean) == 0:
        return math.nan
    return float(np.percentile(clean, q))


def load_stage3_points(out_dir: Path) -> dict[MetricKey, float]:
    """Load existing point estimates for point-preservation audit."""

    stage3 = out_dir / "tables" / "stage3_translatability_map.csv"
    points: dict[MetricKey, float] = {}
    if not stage3.exists():
        return points
    with stage3.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row["source_algorithm"]
            target = row["target_algorithm"]
            for metric in ["mae", "rmse", "abs_error_p95", "mae_per_target_iqr"]:
                points[MetricKey("fullsample_oof_residual", source, target, HEADLINE_METHOD, "overall", "all", metric)] = parse_float_or_nan(row.get(metric))
            for group_type, prefix in [("age_group", "age"), ("sex", "sex"), ("bmi_group", "bmi")]:
                raw_col = f"{prefix}_p95_spread_raw"
                norm_col = f"{prefix}_p95_spread_over_target_iqr"
                points[MetricKey("subgroup_curve_spread", source, target, HEADLINE_METHOD, group_type, "stage3_group_levels", "p95_between_subgroup_spread_predicted_target")] = parse_float_or_nan(row.get(raw_col))
                points[MetricKey("subgroup_curve_spread", source, target, HEADLINE_METHOD, group_type, "stage3_group_levels", "p95_between_subgroup_spread_predicted_target_over_target_iqr")] = parse_float_or_nan(row.get(norm_col))
    return points


def build_ci_rows(
    point_rows: list[dict[str, object]],
    boot_values: dict[MetricKey, list[float]],
    requested_reps: int,
    existing_points: dict[MetricKey, float],
) -> tuple[list[dict[str, object]], list[float]]:
    output: list[dict[str, object]] = []
    point_diffs: list[float] = []
    for point_row in point_rows:
        key = key_from_row(point_row)
        values = boot_values.get(key, [])
        point_value = float(point_row["value"])
        existing = existing_points.get(key, math.nan)
        diff = abs(point_value - existing) if not math.isnan(existing) else math.nan
        if not math.isnan(diff):
            point_diffs.append(diff)
        output.append(
            {
                "metric_family": key.metric_family,
                "source_algorithm": key.source_algorithm,
                "target_algorithm": key.target_algorithm,
                "method": key.method,
                "group_type": key.group_type,
                "group_level": key.group_level,
                "metric": key.metric,
                "point_estimate": point_value,
                "ci_low": finite_percentile(values, CI_LOW_Q),
                "ci_high": finite_percentile(values, CI_HIGH_Q),
                "bootstrap_success": len(values),
                "bootstrap_requested": requested_reps,
                "n_point": point_row["n"],
                "target_iqr_point": point_row["target_iqr"],
                "existing_stage3_point_estimate": existing,
                "abs_diff_vs_stage3": diff,
                "ci_method": "PSU-in-stratum bootstrap; statistic recomputed with grouped 5-fold OOF isotonic crosswalk for residual metrics.",
                "note": point_row["note"],
            }
        )
    return output, point_diffs


def write_summary(
    path: Path,
    ci_rows: list[dict[str, object]],
    design_rows: list[dict[str, object]],
    requested_reps: int,
    point_diffs: list[float],
) -> None:
    headline = [
        row
        for row in ci_rows
        if row["metric_family"] == "fullsample_oof_residual"
        and row["group_type"] == "overall"
        and row["metric"] == "mae_per_target_iqr"
    ]
    headline_sorted = sorted(headline, key=lambda row: float(row["point_estimate"]))
    spread = [
        row
        for row in ci_rows
        if row["metric_family"] == "subgroup_curve_spread"
        and row["metric"] == "p95_between_subgroup_spread_predicted_target_over_target_iqr"
    ]
    spread_sorted = sorted(spread, key=lambda row: float(row["point_estimate"]), reverse=True)
    design_overall = next(row for row in design_rows if row["scope"] == "design_overall")
    max_point_diff = max(point_diffs) if point_diffs else math.nan

    def fmt(value: object, digits: int = 3) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "NA"
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# Stage 4 Crosswalk Uncertainty CI Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "- Scope: MSSE B4 reporting-compliance add-on for daily-steps crosswalk diagnostics.",
        f"- Bootstrap: {requested_reps} PSU-in-stratum design resamples from the D9-valid adult all-7 cohort.",
        f"- Cohort design audit: {design_overall['n_subjects']} subjects, {design_overall['n_psu']} stratum-PSU cells, weighted N={float(design_overall['weighted_n_wtmec4yr']):.1f}.",
        "- Point estimates remain the locked unweighted Stage 2/3 crosswalk diagnostics under SAP D7; this script adds uncertainty intervals only.",
        f"- Point-estimate audit vs Stage 3 where directly keyed: max absolute difference={fmt(max_point_diff, 6)}.",
        "- Guardrails: aggregate CI table only; no SEQN/person-day/minute output, no individual OOF predictions, no bootstrap replicate-level output.",
        "",
        "## Full-Sample MAE/Target-IQR With Bootstrap CI",
        "",
        "| rank | source -> target | point | 95% CI | successful reps |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(headline_sorted[:8], start=1):
        lines.append(
            f"| {rank} | {row['source_algorithm']}->{row['target_algorithm']} | {float(row['point_estimate']):.3f} | {float(row['ci_low']):.3f} to {float(row['ci_high']):.3f} | {row['bootstrap_success']} |"
        )
    lines.append("| ... | ... | ... | ... | ... |")
    for rank, row in enumerate(headline_sorted[-8:], start=max(len(headline_sorted) - 7, 1)):
        lines.append(
            f"| {rank} | {row['source_algorithm']}->{row['target_algorithm']} | {float(row['point_estimate']):.3f} | {float(row['ci_low']):.3f} to {float(row['ci_high']):.3f} | {row['bootstrap_success']} |"
        )

    lines.extend(
        [
            "",
            "## Largest Subgroup Spread/Target-IQR With Bootstrap CI",
            "",
            "| rank | source -> target | subgroup | point | 95% CI | successful reps |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(spread_sorted[:12], start=1):
        lines.append(
            f"| {rank} | {row['source_algorithm']}->{row['target_algorithm']} | {row['group_type']} | {float(row['point_estimate']):.3f} | {float(row['ci_low']):.3f} to {float(row['ci_high']):.3f} | {row['bootstrap_success']} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `outputs/tables/stage4_crosswalk_uncertainty_ci.csv`",
            "- `outputs/tables/stage4_crosswalk_uncertainty_design_audit.csv`",
            "- `outputs/tables/STAGE4_CROSSWALK_UNCERTAINTY_CI_SUMMARY.md`",
            "- `outputs/logs/stage4_crosswalk_uncertainty_ci_run.json`",
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

    log("rebuilding locked Stage 2c daily-steps cohort")
    cohort = build_daily_cohort(args.raw_dir)
    log("reading and aligning DEMO design variables")
    strata, psu, _weights, design_rows = align_design(cohort, args.raw_dir)

    all_indices = np.arange(cohort.subjects_n)
    log("computing original point metrics from rebuilt cohort")
    point_rows: list[dict[str, object]] = []
    pairs = [(source, target) for source in ALGORITHMS for target in ALGORITHMS if source != target]
    for source, target in pairs:
        point_rows.extend(compute_pair_metrics(cohort, all_indices, source, target, STAGE2_CROSSWALK_SEED))
    point_by_key = {key_from_row(row): row for row in point_rows}
    log(f"point metric rows={len(point_rows)} across {len(pairs)} directed pairs")

    rng = np.random.default_rng(args.seed)
    boot_values: dict[MetricKey, list[float]] = defaultdict(list)
    for rep in range(1, args.bootstrap_reps + 1):
        sample_idx = design_bootstrap_indices(strata, psu, rng)
        for source, target in pairs:
            rows = compute_pair_metrics(cohort, sample_idx, source, target, args.seed + rep)
            for row in rows:
                key = key_from_row(row)
                if key in point_by_key:
                    boot_values[key].append(float(row["value"]))
        if rep == 1 or rep == args.bootstrap_reps or rep % args.progress_every == 0:
            log(f"bootstrap replicate {rep}/{args.bootstrap_reps} complete")

    existing_points = load_stage3_points(args.out_dir)
    ci_rows, point_diffs = build_ci_rows(point_rows, boot_values, args.bootstrap_reps, existing_points)

    fieldnames = [
        "metric_family",
        "source_algorithm",
        "target_algorithm",
        "method",
        "group_type",
        "group_level",
        "metric",
        "point_estimate",
        "ci_low",
        "ci_high",
        "bootstrap_success",
        "bootstrap_requested",
        "n_point",
        "target_iqr_point",
        "existing_stage3_point_estimate",
        "abs_diff_vs_stage3",
        "ci_method",
        "note",
    ]
    write_csv(out_tables / "stage4_crosswalk_uncertainty_ci.csv", ci_rows, fieldnames)
    write_csv(
        out_tables / "stage4_crosswalk_uncertainty_design_audit.csv",
        design_rows,
        ["scope", "stratum", "n_subjects", "n_psu", "weighted_n_wtmec4yr", "note"],
    )
    write_summary(
        out_tables / "STAGE4_CROSSWALK_UNCERTAINTY_CI_SUMMARY.md",
        ci_rows,
        design_rows,
        args.bootstrap_reps,
        point_diffs,
    )

    finished = datetime.now(timezone.utc)
    run_log = {
        "script": "scripts/python/build_stage4_crosswalk_uncertainty_ci.py",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "algorithms": ALGORITHMS,
        "pairs": len(pairs),
        "cohort_subjects": cohort.subjects_n,
        "cohort_valid_days": cohort.valid_days_n,
        "metric_rows": len(ci_rows),
        "point_metric_rows": len(point_rows),
        "ci_method": "PSU-in-stratum design bootstrap; statistic recomputed in each replicate.",
        "outputs": [
            "outputs/tables/stage4_crosswalk_uncertainty_ci.csv",
            "outputs/tables/stage4_crosswalk_uncertainty_design_audit.csv",
            "outputs/tables/STAGE4_CROSSWALK_UNCERTAINTY_CI_SUMMARY.md",
            "outputs/logs/stage4_crosswalk_uncertainty_ci_run.json",
        ],
        "guardrails": [
            "raw public NHANES/PhysioNet read locally per human 2026-06-09 instruction",
            "aggregate CI table only",
            "no SEQN/person-day/minute-level output",
            "no individual OOF prediction output",
            "no bootstrap replicate-level output",
            "no SAP method change; point estimates/tier decisions unchanged",
        ],
    }
    with (out_logs / "stage4_crosswalk_uncertainty_ci_run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_log, handle, ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main()
