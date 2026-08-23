#!/usr/bin/env python3
"""Build MVP step exposures and aggregate exposure QC.

Scope:
- D9 main valid-day rule only: Troiano missing-as-nonwear, drop edge days,
  >=960 wear minutes/day, >=3 valid days, no weekend requirement.
- Adult filter comes from NHANES DEMO_G/H RIDAGEYR >=20.
- Outputs are aggregate tables, SVG QC figures, and a run log only.
- No SEQN/person-day/minute-level outputs, no outcomes, no models.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gate0_d9_wear_audit import (
    MINUTES_PER_DAY,
    day_type_for,
    edge_group,
    locate_file,
    normalize_seqn,
    open_xz_text,
    parse_float,
    parse_int,
    pct,
    quantile,
    read_day_minmax,
    summary_stats,
)


STEP_FILES = {
    "adept": "nhanes_1440_adeptsteps.csv.xz",
    "oak": "nhanes_1440_oaksteps.csv.xz",
}
TROIANO_FILE = "nhanes_1440_troianowear.csv.xz"
D9_WEAR_THRESHOLD_MINUTES = 960
D9_MIN_VALID_DAYS = 3
TRUE_TOKENS = {"TRUE", "True", "true", "T", "t", "1"}
MISSING_STEP_TOKENS = {"", "NA", "NaN", "nan", ".", "NULL"}


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


def read_adult_seqns_from_demo(raw_dir: Path) -> set[str]:
    """Read NHANES DEMO_G/H in R and return RIDAGEYR>=20 SEQN.

    Python has no XPT reader in this environment. The R call is captured
    internally and never written to project outputs.
    """

    r_raw_dir = json.dumps(str(raw_dir))
    code = f"""
suppressPackageStartupMessages(library(haven))
normalize_seqn <- function(x) as.character(as.integer(as.numeric(x)))
read_cycle <- function(cycle_dir, stub) {{
  path <- file.path({r_raw_dir}, cycle_dir, paste0(stub, ".XPT"))
  if (!file.exists(path)) stop("missing DEMO file: ", path)
  x <- read_xpt(path)
  x <- x[!is.na(x$RIDAGEYR) & x$RIDAGEYR >= 20, c("SEQN", "RIDAGEYR")]
  normalize_seqn(x$SEQN)
}}
seqns <- unique(c(
  read_cycle("nhanes_2011_2012", "DEMO_G"),
  read_cycle("nhanes_2013_2014", "DEMO_H")
))
cat(paste(seqns, collapse = "\\n"))
"""
    completed = subprocess.run(
        ["Rscript", "--vanilla", "-e", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def read_step_index(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    with open_xz_text(path) as handle:
        header = handle.readline().rstrip("\n\r").split(",")
        if header[:3] != ["SEQN", "PAXDAYM", "PAXDAYWM"]:
            raise ValueError(f"Unexpected step header in {path.name}")
        for line in handle:
            parts = line.rstrip("\n\r").split(",", 3)
            if len(parts) < 3:
                continue
            seqn = normalize_seqn(parts[0])
            day = parse_int(parts[1])
            if seqn and day is not None:
                keys.add((seqn, day))
    return keys


def collect_d9_valid_days(
    troiano_path: Path,
    step_common_keys: set[tuple[str, int]],
    minmax_by_seqn: dict[str, tuple[int, int]],
    adult_seqns: set[str],
) -> tuple[dict[str, list[tuple[str, int]]], set[str], set[tuple[str, int]], list[dict[str, object]]]:
    valid_days_by_subject: dict[str, list[tuple[str, int]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    with open_xz_text(troiano_path) as handle:
        header = handle.readline().rstrip("\n\r").split(",")
        if header[:3] != ["SEQN", "PAXDAYM", "PAXDAYWM"]:
            raise ValueError("Unexpected troianowear header.")
        for line in handle:
            parts = line.rstrip("\n\r").split(",")
            if len(parts) < 4:
                continue
            seqn = normalize_seqn(parts[0])
            day = parse_int(parts[1])
            if not seqn or day is None:
                continue
            key = (seqn, day)
            counters["troiano_person_days"] += 1
            if key not in step_common_keys:
                continue
            counters["troiano_intersect_step_person_days"] += 1
            day_type = day_type_for(seqn, day, minmax_by_seqn)
            if edge_group(day_type) == "edge_day":
                counters["edge_days_dropped"] += 1
                continue
            counters["interior_intersection_person_days"] += 1
            wear_minutes = sum(1 for value in parts[3:] if value in TRUE_TOKENS)
            if wear_minutes >= D9_WEAR_THRESHOLD_MINUTES:
                counters["d9_valid_person_days_before_min_days"] += 1
                valid_days_by_subject[seqn].append(key)

    d9_valid_subjects = {
        seqn for seqn, keys in valid_days_by_subject.items() if len(keys) >= D9_MIN_VALID_DAYS
    }
    final_subjects = d9_valid_subjects & adult_seqns
    final_valid_days = {
        key
        for seqn in final_subjects
        for key in valid_days_by_subject.get(seqn, [])
    }
    step_subjects = {seqn for seqn, _day in step_common_keys}
    troiano_subjects = set(minmax_by_seqn)
    alignment_rows = [
        {
            "metric": "mvp_step_common_subjects_adept_intersect_oak",
            "value": len(step_subjects),
            "notes": "Rows present in both adeptsteps and oaksteps.",
        },
        {
            "metric": "mvp_step_common_person_days_adept_intersect_oak",
            "value": len(step_common_keys),
            "notes": "Person-day rows present in both adeptsteps and oaksteps.",
        },
        {
            "metric": "troiano_subjects",
            "value": len(troiano_subjects),
            "notes": "Subjects present in troianowear.",
        },
        {
            "metric": "troiano_person_days",
            "value": counters["troiano_person_days"],
            "notes": "Rows present in troianowear.",
        },
        {
            "metric": "troiano_intersect_mvp_step_subjects",
            "value": len({seqn for seqn, _day in step_common_keys if seqn in troiano_subjects}),
            "notes": "Subject denominator before D9 rule.",
        },
        {
            "metric": "troiano_intersect_mvp_step_person_days",
            "value": counters["troiano_intersect_step_person_days"],
            "notes": "Person-day denominator before D9 rule.",
        },
        {
            "metric": "edge_person_days_dropped",
            "value": counters["edge_days_dropped"],
            "notes": "First/last PAXDAYM per SEQN dropped before valid-day assessment.",
        },
        {
            "metric": "interior_intersection_person_days",
            "value": counters["interior_intersection_person_days"],
            "notes": "Interior days in Troiano intersect MVP step person-days.",
        },
        {
            "metric": "d9_valid_person_days_before_min_days",
            "value": counters["d9_valid_person_days_before_min_days"],
            "notes": "Interior days with Troiano wear >=960 minutes.",
        },
        {
            "metric": "d9_valid_subjects_before_adult_filter",
            "value": len(d9_valid_subjects),
            "notes": "Subjects with >=3 D9-valid days; no weekend requirement.",
        },
        {
            "metric": "demo_adult_20plus_subjects",
            "value": len(adult_seqns),
            "notes": "DEMO_G/H RIDAGEYR >=20; used only for adult filter.",
        },
        {
            "metric": "analysis_subjects_d9_valid_adult_20plus",
            "value": len(final_subjects),
            "notes": "Exposure QC denominator.",
        },
        {
            "metric": "analysis_valid_person_days_d9_valid_adult_20plus",
            "value": len(final_valid_days),
            "notes": "Valid days used to construct exposures.",
        },
    ]
    return valid_days_by_subject, final_subjects, final_valid_days, alignment_rows


def collect_wear_masks(
    troiano_path: Path,
    final_valid_days: set[tuple[str, int]],
) -> dict[tuple[str, int], bytes]:
    masks: dict[tuple[str, int], bytes] = {}
    with open_xz_text(troiano_path) as handle:
        header = handle.readline().rstrip("\n\r").split(",")
        if header[:3] != ["SEQN", "PAXDAYM", "PAXDAYWM"]:
            raise ValueError("Unexpected troianowear header.")
        for line in handle:
            parts = line.rstrip("\n\r").split(",")
            if len(parts) < 4:
                continue
            seqn = normalize_seqn(parts[0])
            day = parse_int(parts[1])
            if not seqn or day is None:
                continue
            key = (seqn, day)
            if key not in final_valid_days:
                continue
            minute_values = parts[3:3 + MINUTES_PER_DAY]
            mask = bytearray(MINUTES_PER_DAY)
            for idx, value in enumerate(minute_values):
                if value in TRUE_TOKENS:
                    mask[idx] = 1
            masks[key] = bytes(mask)
    missing = len(final_valid_days) - len(masks)
    if missing:
        raise ValueError(f"Missing Troiano masks for {missing} final valid days.")
    return masks


def parse_step_value(value: str) -> float | None:
    if value in MISSING_STEP_TOKENS:
        return None
    try:
        return int(value)
    except ValueError:
        return parse_float(value)


def empty_subject_stat() -> dict[str, float | int]:
    return {
        "valid_days": 0,
        "daily_step_sum": 0.0,
        "wear_step_sum": 0.0,
        "nonwear_step_sum": 0.0,
        "missing_step_minutes": 0,
        "cadence_day_count": 0,
        "cadence_sum": 0.0,
        "cadence_missing_days": 0,
    }


def process_step_algorithm(
    algorithm: str,
    path: Path,
    wear_masks: dict[tuple[str, int], bytes],
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    stats_by_subject: dict[str, dict[str, float | int]] = defaultdict(empty_subject_stat)
    totals: Counter[str] = Counter()
    total_steps = 0.0
    wear_steps = 0.0
    nonwear_steps = 0.0

    with open_xz_text(path) as handle:
        header = handle.readline().rstrip("\n\r").split(",")
        if header[:3] != ["SEQN", "PAXDAYM", "PAXDAYWM"]:
            raise ValueError(f"Unexpected step header in {path.name}")
        for line in handle:
            parts = line.rstrip("\n\r").split(",")
            if len(parts) < 4:
                continue
            seqn = normalize_seqn(parts[0])
            day = parse_int(parts[1])
            if not seqn or day is None:
                continue
            key = (seqn, day)
            wear_mask = wear_masks.get(key)
            if wear_mask is None:
                continue

            totals["valid_days_processed"] += 1
            subject = stats_by_subject[seqn]
            subject["valid_days"] += 1

            minute_values = parts[3:3 + MINUTES_PER_DAY]
            day_total = 0.0
            day_wear = 0.0
            day_nonwear = 0.0
            day_missing = 0
            positive_wear_steps: list[float] = []

            for idx in range(MINUTES_PER_DAY):
                value = minute_values[idx] if idx < len(minute_values) else ""
                parsed = parse_step_value(value)
                if parsed is None:
                    day_missing += 1
                    step = 0.0
                else:
                    step = float(parsed)
                day_total += step
                if wear_mask[idx]:
                    day_wear += step
                    if step > 0:
                        positive_wear_steps.append(step)
                else:
                    day_nonwear += step

            if len(positive_wear_steps) >= 30:
                cadence = sum(heapq.nlargest(30, positive_wear_steps)) / 30.0
                subject["cadence_day_count"] += 1
                subject["cadence_sum"] += cadence
                totals["cadence_days_nonmissing"] += 1
            else:
                subject["cadence_missing_days"] += 1
                totals["cadence_days_missing_lt30_positive_wear_step_minutes"] += 1

            subject["daily_step_sum"] += day_total
            subject["wear_step_sum"] += day_wear
            subject["nonwear_step_sum"] += day_nonwear
            subject["missing_step_minutes"] += day_missing
            total_steps += day_total
            wear_steps += day_wear
            nonwear_steps += day_nonwear
            totals["missing_step_minutes"] += day_missing
            if day_nonwear > 0:
                totals["days_with_nonwear_steps_gt0"] += 1

    subject_metrics: dict[str, dict[str, float]] = {}
    for seqn, subject in stats_by_subject.items():
        valid_days = int(subject["valid_days"])
        cadence_day_count = int(subject["cadence_day_count"])
        daily_steps = float(subject["daily_step_sum"]) / valid_days if valid_days else math.nan
        wear_daily_steps = float(subject["wear_step_sum"]) / valid_days if valid_days else math.nan
        nonwear_daily_steps = float(subject["nonwear_step_sum"]) / valid_days if valid_days else math.nan
        nonwear_share = (
            float(subject["nonwear_step_sum"]) / float(subject["daily_step_sum"])
            if float(subject["daily_step_sum"]) > 0
            else math.nan
        )
        cadence = (
            float(subject["cadence_sum"]) / cadence_day_count
            if cadence_day_count
            else math.nan
        )
        subject_metrics[seqn] = {
            "daily_steps_whole_day_sum_mean": daily_steps,
            "wear_daily_steps_mean_qc": wear_daily_steps,
            "nonwear_daily_steps_mean_qc": nonwear_daily_steps,
            "nonwear_step_share": nonwear_share,
            "peak30_cadence_mean_secondary": cadence,
            "valid_days": float(valid_days),
            "cadence_day_count": float(cadence_day_count),
            "cadence_missing_days": float(subject["cadence_missing_days"]),
            "missing_step_minutes": float(subject["missing_step_minutes"]),
        }

    aggregate = {
        "algorithm": algorithm,
        "valid_days_processed": totals["valid_days_processed"],
        "subjects_processed": len(subject_metrics),
        "total_steps_whole_day": total_steps,
        "wear_steps": wear_steps,
        "nonwear_steps": nonwear_steps,
        "nonwear_step_share_all_valid_days": nonwear_steps / total_steps if total_steps > 0 else math.nan,
        "days_with_nonwear_steps_gt0": totals["days_with_nonwear_steps_gt0"],
        "pct_days_with_nonwear_steps_gt0": pct(
            totals["days_with_nonwear_steps_gt0"],
            totals["valid_days_processed"],
        ),
        "missing_step_minutes": totals["missing_step_minutes"],
        "cadence_days_nonmissing": totals["cadence_days_nonmissing"],
        "cadence_days_missing_lt30_positive_wear_step_minutes": totals[
            "cadence_days_missing_lt30_positive_wear_step_minutes"
        ],
        "pct_cadence_days_missing": pct(
            totals["cadence_days_missing_lt30_positive_wear_step_minutes"],
            totals["valid_days_processed"],
        ),
    }
    return subject_metrics, aggregate


def add_distribution_row(
    rows: list[dict[str, object]],
    algorithm: str,
    metric: str,
    values: list[float],
    denominator_n: int,
    calculation_note: str,
) -> None:
    clean = [value for value in values if not math.isnan(value)]
    stats = summary_stats(clean)
    rows.append(
        {
            "algorithm": algorithm,
            "metric": metric,
            "calculation_note": calculation_note,
            "n_subjects_denominator": denominator_n,
            "n_subjects_nonmissing": len(clean),
            "n_subjects_missing": denominator_n - len(clean),
            "pct_subjects_missing": pct(denominator_n - len(clean), denominator_n),
            "iqr": stats["p75"] - stats["p25"] if not math.isnan(float(stats["p25"])) else math.nan,
            **stats,
        }
    )


def build_distribution_rows(
    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]],
    final_subjects: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    denominator_n = len(final_subjects)
    metric_notes = {
        "daily_steps_whole_day_sum_mean": (
            "Main exposure. For each D9-valid day, sum all 1440 minute-level step counts "
            "for the day, including wear and nonwear minutes; then average valid days per person."
        ),
        "peak30_cadence_mean_secondary": (
            "Secondary exposure. For each D9-valid day, use wear minutes only; take the "
            "30 highest positive-step wear minutes and divide their sum by 30. Days with "
            "<30 positive-step wear minutes are missing, not zero."
        ),
        "nonwear_step_share": (
            "QC metric. Person-level nonwear steps divided by whole-day steps across D9-valid days; "
            "expected near zero if algorithms do not count steps in nonwear minutes."
        ),
        "wear_daily_steps_mean_qc": "QC metric. Valid-day mean steps restricted to Troiano wear minutes.",
        "nonwear_daily_steps_mean_qc": "QC metric. Valid-day mean steps in Troiano nonwear minutes.",
        "valid_days": "Number of D9-valid days contributing to daily steps.",
        "cadence_day_count": "Number of D9-valid days with >=30 positive-step wear minutes.",
        "cadence_missing_days": "Number of D9-valid days with <30 positive-step wear minutes.",
    }
    for algorithm, subject_metrics in sorted(metrics_by_algorithm.items()):
        for metric, note in metric_notes.items():
            values = [
                subject_metrics.get(seqn, {}).get(metric, math.nan)
                for seqn in final_subjects
            ]
            add_distribution_row(rows, algorithm, metric, values, denominator_n, note)
    return rows


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x <= 0 or denom_y <= 0:
        return math.nan
    return numerator / (denom_x * denom_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for _value, idx in indexed[i:j]:
            result[idx] = rank
        i = j
    return result


def spearman_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    return pearson_r(ranks(xs), ranks(ys))


def build_pairwise_rows(
    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]],
    final_subjects: set[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, list[float]]]]:
    rows: list[dict[str, object]] = []
    plot_data: dict[str, dict[str, list[float]]] = {}
    metric_labels = {
        "daily_steps_whole_day_sum_mean": "daily_steps",
        "peak30_cadence_mean_secondary": "peak30_cadence",
    }
    for metric, label in metric_labels.items():
        xs: list[float] = []
        ys: list[float] = []
        for seqn in final_subjects:
            x = metrics_by_algorithm["adept"].get(seqn, {}).get(metric, math.nan)
            y = metrics_by_algorithm["oak"].get(seqn, {}).get(metric, math.nan)
            if not math.isnan(x) and not math.isnan(y):
                xs.append(x)
                ys.append(y)
        diffs = [x - y for x, y in zip(xs, ys)]
        means = [(x + y) / 2 for x, y in zip(xs, ys)]
        diff_stats = summary_stats(diffs)
        sd_diff = float(diff_stats["sd"])
        mean_diff = float(diff_stats["mean"])
        rows.append(
            {
                "metric": label,
                "n_pair": len(xs),
                "adept_minus_oak_mean": mean_diff,
                "adept_minus_oak_sd": sd_diff,
                "adept_minus_oak_median": diff_stats["p50"],
                "adept_minus_oak_p05": diff_stats["p05"],
                "adept_minus_oak_p95": diff_stats["p95"],
                "pearson_r": pearson_r(xs, ys),
                "spearman_r": spearman_r(xs, ys),
                "bland_altman_mean_diff": mean_diff,
                "bland_altman_lower_loa": mean_diff - 1.96 * sd_diff if not math.isnan(sd_diff) else math.nan,
                "bland_altman_upper_loa": mean_diff + 1.96 * sd_diff if not math.isnan(sd_diff) else math.nan,
                "notes": "Pairwise subject-level comparison; no SEQN exported.",
            }
        )
        plot_data[label] = {"adept": xs, "oak": ys, "mean": means, "diff": diffs}
    return rows, plot_data


def build_nonwear_rows(
    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]],
    aggregate_by_algorithm: dict[str, dict[str, object]],
    final_subjects: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for algorithm in sorted(metrics_by_algorithm):
        aggregate = aggregate_by_algorithm[algorithm]
        person_shares = [
            metrics_by_algorithm[algorithm].get(seqn, {}).get("nonwear_step_share", math.nan)
            for seqn in final_subjects
        ]
        clean_shares = [value for value in person_shares if not math.isnan(value)]
        share_stats = summary_stats(clean_shares)
        rows.append(
            {
                "algorithm": algorithm,
                "scope": "all_valid_days_total_step_minutes",
                "n_subjects": len(final_subjects),
                "n_valid_days": aggregate["valid_days_processed"],
                "total_steps_whole_day": aggregate["total_steps_whole_day"],
                "wear_steps": aggregate["wear_steps"],
                "nonwear_steps": aggregate["nonwear_steps"],
                "nonwear_step_share": aggregate["nonwear_step_share_all_valid_days"],
                "days_with_nonwear_steps_gt0": aggregate["days_with_nonwear_steps_gt0"],
                "pct_days_with_nonwear_steps_gt0": aggregate["pct_days_with_nonwear_steps_gt0"],
                "person_nonwear_share_mean": share_stats["mean"],
                "person_nonwear_share_p50": share_stats["p50"],
                "person_nonwear_share_p95": share_stats["p95"],
                "person_nonwear_share_max": share_stats["max"],
                "note": "QC for possible nonwear step counting; expected near zero.",
            }
        )
    return rows


def build_cadence_missing_rows(
    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]],
    aggregate_by_algorithm: dict[str, dict[str, object]],
    final_subjects: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for algorithm in sorted(metrics_by_algorithm):
        aggregate = aggregate_by_algorithm[algorithm]
        n_person_missing_all = sum(
            1
            for seqn in final_subjects
            if math.isnan(
                metrics_by_algorithm[algorithm].get(seqn, {}).get(
                    "peak30_cadence_mean_secondary",
                    math.nan,
                )
            )
        )
        rows.append(
            {
                "algorithm": algorithm,
                "n_subjects": len(final_subjects),
                "n_subjects_with_no_nonmissing_cadence_days": n_person_missing_all,
                "pct_subjects_with_no_nonmissing_cadence_days": pct(n_person_missing_all, len(final_subjects)),
                "n_valid_days": aggregate["valid_days_processed"],
                "n_cadence_days_nonmissing": aggregate["cadence_days_nonmissing"],
                "n_cadence_days_missing_lt30_positive_wear_step_minutes": aggregate[
                    "cadence_days_missing_lt30_positive_wear_step_minutes"
                ],
                "pct_cadence_days_missing": aggregate["pct_cadence_days_missing"],
                "note": "D11: cadence day is missing if D9-valid day has <30 positive-step wear minutes; no zero padding.",
            }
        )
    return rows


def finite_range(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return 0.0, 1.0
    lo = min(clean)
    hi = max(clean)
    if lo == hi:
        return lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def scale(value: float, domain: tuple[float, float], target: tuple[float, float]) -> float:
    lo, hi = domain
    start, end = target
    if hi == lo:
        return (start + end) / 2
    return start + (value - lo) * (end - start) / (hi - lo)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial" font-size="16" fill="#222">{title}</text>',
    ]


def svg_axes(lines: list[str], width: int, height: int, xlabel: str, ylabel: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = 70, 48, width - 28, height - 58
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333" stroke-width="1"/>')
    lines.append(f'<text x="{(left + right) / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{xlabel}</text>')
    lines.append(f'<text x="18" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {(top + bottom) / 2:.1f})" font-family="Arial" font-size="12" fill="#333">{ylabel}</text>')
    return left, top, right, bottom


def write_scatter_svg(path: Path, xs: list[float], ys: list[float], xlabel: str, ylabel: str, title: str) -> None:
    width, height = 760, 560
    x_domain = finite_range(xs)
    y_domain = finite_range(ys)
    lines = svg_header(width, height, title)
    left, top, right, bottom = svg_axes(lines, width, height, xlabel, ylabel)
    identity_lo = max(x_domain[0], y_domain[0])
    identity_hi = min(x_domain[1], y_domain[1])
    if identity_hi > identity_lo:
        x1 = scale(identity_lo, x_domain, (left, right))
        y1 = scale(identity_lo, y_domain, (bottom, top))
        x2 = scale(identity_hi, x_domain, (left, right))
        y2 = scale(identity_hi, y_domain, (bottom, top))
        lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#666" stroke-dasharray="4 4" stroke-width="1"/>')
    for x, y in zip(xs, ys):
        cx = scale(x, x_domain, (left, right))
        cy = scale(y, y_domain, (bottom, top))
        lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="1.6" fill="#1f77b4" fill-opacity="0.28"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bland_altman_svg(
    path: Path,
    means: list[float],
    diffs: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    width, height = 760, 560
    x_domain = finite_range(means)
    y_domain = finite_range(diffs)
    diff_stats = summary_stats(diffs)
    mean_diff = float(diff_stats["mean"])
    sd_diff = float(diff_stats["sd"])
    loa = [
        ("mean", mean_diff, "#333"),
        ("lower LoA", mean_diff - 1.96 * sd_diff, "#b22222"),
        ("upper LoA", mean_diff + 1.96 * sd_diff, "#b22222"),
    ]
    y_values = diffs + [value for _label, value, _color in loa if not math.isnan(value)]
    y_domain = finite_range(y_values)
    lines = svg_header(width, height, title)
    left, top, right, bottom = svg_axes(lines, width, height, xlabel, ylabel)
    for label, value, color in loa:
        if math.isnan(value):
            continue
        y = scale(value, y_domain, (bottom, top))
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{color}" stroke-dasharray="5 4" stroke-width="1"/>')
        lines.append(f'<text x="{right - 4}" y="{y - 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="{color}">{label}</text>')
    for x, y_value in zip(means, diffs):
        cx = scale(x, x_domain, (left, right))
        cy = scale(y_value, y_domain, (bottom, top))
        lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="1.6" fill="#2ca02c" fill-opacity="0.28"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def histogram(values: list[float], bins: int, domain: tuple[float, float]) -> list[float]:
    counts = [0] * bins
    lo, hi = domain
    if hi <= lo:
        return [0.0] * bins
    for value in values:
        if math.isnan(value):
            continue
        idx = int((value - lo) / (hi - lo) * bins)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    total = sum(counts)
    width = (hi - lo) / bins
    if total == 0 or width <= 0:
        return [0.0] * bins
    return [count / total / width for count in counts]


def write_density_svg(
    path: Path,
    adept: list[float],
    oak: list[float],
    xlabel: str,
    title: str,
) -> None:
    width, height = 760, 560
    clean = [v for v in adept + oak if not math.isnan(v)]
    x_domain = finite_range(clean)
    bins = 60
    adept_density = histogram(adept, bins, x_domain)
    oak_density = histogram(oak, bins, x_domain)
    y_domain = finite_range(adept_density + oak_density + [0.0])
    lines = svg_header(width, height, title)
    left, top, right, bottom = svg_axes(lines, width, height, xlabel, "Density")
    colors = {"adept": "#1f77b4", "oak": "#d62728"}
    for label, densities in [("adept", adept_density), ("oak", oak_density)]:
        points = []
        for idx, density in enumerate(densities):
            x_value = x_domain[0] + (idx + 0.5) * (x_domain[1] - x_domain[0]) / bins
            x = scale(x_value, x_domain, (left, right))
            y = scale(density, y_domain, (bottom, top))
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[label]}" stroke-width="2"/>'
        )
    lines.append(f'<rect x="{right - 118}" y="{top + 8}" width="94" height="42" fill="white" stroke="#ddd"/>')
    lines.append(f'<line x1="{right - 108}" y1="{top + 22}" x2="{right - 82}" y2="{top + 22}" stroke="{colors["adept"]}" stroke-width="2"/>')
    lines.append(f'<text x="{right - 76}" y="{top + 26}" font-family="Arial" font-size="12" fill="#333">adept</text>')
    lines.append(f'<line x1="{right - 108}" y1="{top + 40}" x2="{right - 82}" y2="{top + 40}" stroke="{colors["oak"]}" stroke-width="2"/>')
    lines.append(f'<text x="{right - 76}" y="{top + 44}" font-family="Arial" font-size="12" fill="#333">oak</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    alignment_rows: list[dict[str, object]],
    distribution_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
    nonwear_rows: list[dict[str, object]],
    cadence_rows: list[dict[str, object]],
) -> None:
    align = {row["metric"]: row["value"] for row in alignment_rows}
    daily_rows = {
        row["algorithm"]: row
        for row in distribution_rows
        if row["metric"] == "daily_steps_whole_day_sum_mean"
    }
    cadence_dist_rows = {
        row["algorithm"]: row
        for row in distribution_rows
        if row["metric"] == "peak30_cadence_mean_secondary"
    }
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# MVP Exposure Construction QC Summary",
        "",
        f"- Generated: {generated}",
        "- Scope: exposure construction and exposure-quality checks only; no HbA1c/waist linkage and no models.",
        "- Denominator: D9-valid adults aged >=20 in `adeptsteps ∩ oaksteps ∩ troianowear`.",
        "- D9 main: Troiano missing-as-nonwear, drop first/last edge days, wear >=960 min/day, >=3 valid days, weekend not required.",
        "- Daily steps main exposure is explicitly a whole-day 1440-minute sum on each D9-valid day, then a person-level mean across valid days.",
        "- Peak-30 cadence is secondary: wear minutes only; days with <30 positive-step wear minutes are missing, not zero.",
        "- Guardrail: no SEQN/person-day/minute-level table is exported; plots have no IDs or labels.",
        "",
        "## Denominator",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for metric in [
        "mvp_step_common_subjects_adept_intersect_oak",
        "mvp_step_common_person_days_adept_intersect_oak",
        "troiano_intersect_mvp_step_subjects",
        "troiano_intersect_mvp_step_person_days",
        "edge_person_days_dropped",
        "d9_valid_subjects_before_adult_filter",
        "analysis_subjects_d9_valid_adult_20plus",
        "analysis_valid_person_days_d9_valid_adult_20plus",
    ]:
        lines.append(f"| {metric} | {align.get(metric, 'NA')} |")
    lines.extend(["", "## Exposure Distributions", "", "| algorithm | daily steps mean | daily steps median | cadence mean | cadence median | cadence missing % |", "|---|---:|---:|---:|---:|---:|"])
    cadence_missing_by_alg = {row["algorithm"]: row for row in cadence_rows}
    for algorithm in ["adept", "oak"]:
        daily = daily_rows[algorithm]
        cadence = cadence_dist_rows[algorithm]
        missing = cadence_missing_by_alg[algorithm]
        lines.append(
            "| {alg} | {daily_mean} | {daily_med} | {cad_mean} | {cad_med} | {cad_miss} |".format(
                alg=algorithm,
                daily_mean=fmt(float(daily["mean"]), 1),
                daily_med=fmt(float(daily["p50"]), 1),
                cad_mean=fmt(float(cadence["mean"]), 2),
                cad_med=fmt(float(cadence["p50"]), 2),
                cad_miss=fmt(float(missing["pct_cadence_days_missing"]) * 100, 2),
            )
        )
    lines.extend(["", "## Adept vs Oak", "", "| metric | n | Pearson r | Spearman r | mean adept-oak | BA lower LoA | BA upper LoA |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in pairwise_rows:
        lines.append(
            "| {metric} | {n} | {pearson} | {spearman} | {diff} | {lower} | {upper} |".format(
                metric=row["metric"],
                n=row["n_pair"],
                pearson=fmt(float(row["pearson_r"]), 4),
                spearman=fmt(float(row["spearman_r"]), 4),
                diff=fmt(float(row["adept_minus_oak_mean"]), 2),
                lower=fmt(float(row["bland_altman_lower_loa"]), 2),
                upper=fmt(float(row["bland_altman_upper_loa"]), 2),
            )
        )
    lines.extend(["", "## Nonwear Step QC", "", "| algorithm | nonwear share all valid days | days with nonwear steps >0 | pct days | person p95 share | person max share |", "|---|---:|---:|---:|---:|---:|"])
    for row in nonwear_rows:
        lines.append(
            "| {alg} | {share} | {days} | {pct_days} | {p95} | {max_share} |".format(
                alg=row["algorithm"],
                share=fmt(float(row["nonwear_step_share"]), 8),
                days=row["days_with_nonwear_steps_gt0"],
                pct_days=fmt(float(row["pct_days_with_nonwear_steps_gt0"]) * 100, 4),
                p95=fmt(float(row["person_nonwear_share_p95"]), 8),
                max_share=fmt(float(row["person_nonwear_share_max"]), 8),
            )
        )
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

    physio_dir = raw_dir / "physionet_v1.0.1"
    troiano_path = locate_file(physio_dir, TROIANO_FILE)
    step_paths = {
        algorithm: locate_file(physio_dir, filename)
        for algorithm, filename in STEP_FILES.items()
    }
    adult_seqns = read_adult_seqns_from_demo(raw_dir)
    step_indices = {algorithm: read_step_index(path) for algorithm, path in step_paths.items()}
    step_common_keys = step_indices["adept"] & step_indices["oak"]
    minmax_by_seqn = read_day_minmax(troiano_path)
    _valid_days_by_subject, final_subjects, final_valid_days, alignment_rows = collect_d9_valid_days(
        troiano_path=troiano_path,
        step_common_keys=step_common_keys,
        minmax_by_seqn=minmax_by_seqn,
        adult_seqns=adult_seqns,
    )
    alignment_rows.extend(
        [
            {
                "metric": "adept_only_person_days_vs_oak",
                "value": len(step_indices["adept"] - step_indices["oak"]),
                "notes": "Aggregate equality check only.",
            },
            {
                "metric": "oak_only_person_days_vs_adept",
                "value": len(step_indices["oak"] - step_indices["adept"]),
                "notes": "Aggregate equality check only.",
            },
        ]
    )
    wear_masks = collect_wear_masks(troiano_path, final_valid_days)

    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]] = {}
    aggregate_by_algorithm: dict[str, dict[str, object]] = {}
    for algorithm, path in step_paths.items():
        metrics, aggregate = process_step_algorithm(algorithm, path, wear_masks)
        metrics_by_algorithm[algorithm] = metrics
        aggregate_by_algorithm[algorithm] = aggregate

    distribution_rows = build_distribution_rows(metrics_by_algorithm, final_subjects)
    pairwise_rows, plot_data = build_pairwise_rows(metrics_by_algorithm, final_subjects)
    nonwear_rows = build_nonwear_rows(metrics_by_algorithm, aggregate_by_algorithm, final_subjects)
    cadence_rows = build_cadence_missing_rows(metrics_by_algorithm, aggregate_by_algorithm, final_subjects)

    write_csv(
        out_tables / "mvp_exposure_qc_denominator.csv",
        alignment_rows,
        ["metric", "value", "notes"],
    )
    write_csv(
        out_tables / "mvp_exposure_distribution.csv",
        distribution_rows,
        [
            "algorithm",
            "metric",
            "calculation_note",
            "n_subjects_denominator",
            "n_subjects_nonmissing",
            "n_subjects_missing",
            "pct_subjects_missing",
            "iqr",
            "n",
            "mean",
            "sd",
            "p01",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "min",
            "max",
        ],
    )
    write_csv(
        out_tables / "mvp_exposure_algorithm_pairwise.csv",
        pairwise_rows,
        [
            "metric",
            "n_pair",
            "adept_minus_oak_mean",
            "adept_minus_oak_sd",
            "adept_minus_oak_median",
            "adept_minus_oak_p05",
            "adept_minus_oak_p95",
            "pearson_r",
            "spearman_r",
            "bland_altman_mean_diff",
            "bland_altman_lower_loa",
            "bland_altman_upper_loa",
            "notes",
        ],
    )
    write_csv(
        out_tables / "mvp_exposure_nonwear_step_qc.csv",
        nonwear_rows,
        [
            "algorithm",
            "scope",
            "n_subjects",
            "n_valid_days",
            "total_steps_whole_day",
            "wear_steps",
            "nonwear_steps",
            "nonwear_step_share",
            "days_with_nonwear_steps_gt0",
            "pct_days_with_nonwear_steps_gt0",
            "person_nonwear_share_mean",
            "person_nonwear_share_p50",
            "person_nonwear_share_p95",
            "person_nonwear_share_max",
            "note",
        ],
    )
    write_csv(
        out_tables / "mvp_exposure_cadence_missingness.csv",
        cadence_rows,
        [
            "algorithm",
            "n_subjects",
            "n_subjects_with_no_nonmissing_cadence_days",
            "pct_subjects_with_no_nonmissing_cadence_days",
            "n_valid_days",
            "n_cadence_days_nonmissing",
            "n_cadence_days_missing_lt30_positive_wear_step_minutes",
            "pct_cadence_days_missing",
            "note",
        ],
    )

    write_summary(
        out_tables / "MVP_EXPOSURE_QC_SUMMARY.md",
        alignment_rows=alignment_rows,
        distribution_rows=distribution_rows,
        pairwise_rows=pairwise_rows,
        nonwear_rows=nonwear_rows,
        cadence_rows=cadence_rows,
    )

    for label, data in plot_data.items():
        if label == "daily_steps":
            xlab, ylab = "Adept daily steps", "Oak daily steps"
            ba_xlab, ba_ylab = "Mean daily steps", "Adept - Oak daily steps"
            density_xlab = "Person mean daily steps"
        else:
            xlab, ylab = "Adept peak-30 cadence", "Oak peak-30 cadence"
            ba_xlab, ba_ylab = "Mean peak-30 cadence", "Adept - Oak peak-30 cadence"
            density_xlab = "Person mean peak-30 cadence"
        write_scatter_svg(
            out_figures / f"mvp_exposure_{label}_adept_vs_oak_scatter.svg",
            data["adept"],
            data["oak"],
            xlab,
            ylab,
            f"{label}: adept vs oak",
        )
        write_bland_altman_svg(
            out_figures / f"mvp_exposure_{label}_bland_altman.svg",
            data["mean"],
            data["diff"],
            ba_xlab,
            ba_ylab,
            f"{label}: Bland-Altman",
        )
        write_density_svg(
            out_figures / f"mvp_exposure_{label}_density.svg",
            data["adept"],
            data["oak"],
            density_xlab,
            f"{label}: density",
        )

    finished = datetime.now(timezone.utc)
    log = {
        "script": "scripts/python/build_mvp_exposure_qc.py",
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "raw_dir": str(raw_dir),
        "d9_main_rule": {
            "wear_source": "troianowear",
            "missing_policy": "missing_as_nonwear",
            "edge_day_policy": "drop_first_last_edge_days",
            "daily_wear_threshold_minutes": D9_WEAR_THRESHOLD_MINUTES,
            "min_valid_days": D9_MIN_VALID_DAYS,
            "weekend_requirement": "none",
        },
        "exposure_definitions": {
            "daily_steps": "whole-day 1440-minute sum per D9-valid day, averaged per person",
            "peak30_cadence": "top 30 positive-step wear minutes per D9-valid day / 30, averaged across nonmissing days per person",
        },
        "denominator_subjects": len(final_subjects),
        "denominator_valid_days": len(final_valid_days),
        "outputs": [
            "outputs/tables/mvp_exposure_qc_denominator.csv",
            "outputs/tables/mvp_exposure_distribution.csv",
            "outputs/tables/mvp_exposure_algorithm_pairwise.csv",
            "outputs/tables/mvp_exposure_nonwear_step_qc.csv",
            "outputs/tables/mvp_exposure_cadence_missingness.csv",
            "outputs/tables/MVP_EXPOSURE_QC_SUMMARY.md",
            "outputs/figures/mvp_exposure_daily_steps_adept_vs_oak_scatter.svg",
            "outputs/figures/mvp_exposure_daily_steps_bland_altman.svg",
            "outputs/figures/mvp_exposure_daily_steps_density.svg",
            "outputs/figures/mvp_exposure_peak30_cadence_adept_vs_oak_scatter.svg",
            "outputs/figures/mvp_exposure_peak30_cadence_bland_altman.svg",
            "outputs/figures/mvp_exposure_peak30_cadence_density.svg",
            "outputs/logs/mvp_exposure_qc_run.json",
        ],
        "guardrails": [
            "aggregate tables only",
            "QC plots have no SEQN labels",
            "no SEQN/person-day/minute-level outputs",
            "no HbA1c or waist linkage",
            "no exposure-outcome models",
            "SAP not modified",
        ],
    }
    with (out_logs / "mvp_exposure_qc_run.json").open("w", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
