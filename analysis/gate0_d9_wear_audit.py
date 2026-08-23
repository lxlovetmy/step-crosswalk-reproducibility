#!/usr/bin/env python3
"""Gate 0 D9 wear-rule audit for PhysioNet NHANES step-count files.

This script produces aggregate evidence for selecting the unified valid-day
rule. It does not link metabolic outcomes, run exposure-outcome models, or
export participant/day/minute-level records.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


MISSING_TOKENS = {"", "NA", "NaN", "nan", ".", "NULL"}
MINUTES_PER_DAY = 1440
WEAR_THRESHOLDS = [600, 720, 960]
MIN_VALID_DAYS = [1, 3, 4]
WEEKEND_REQUIREMENTS = ["none", "at_least_1"]
EDGE_DAY_POLICIES = ["keep_edge_days", "drop_edge_days"]
PAXPREDM_POLICIES = ["unknown_as_nonwear", "unknown_excluded"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing PhysioNet v1.0.1 files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for aggregate outputs.",
    )
    return parser.parse_args()


def open_xz_text(path: Path):
    return lzma.open(path, mode="rt", encoding="utf-8", newline="")


def locate_file(raw_dir: Path, filename: str) -> Path:
    candidates = [raw_dir / filename, raw_dir / "csv" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required file not found: {filename}")


def is_missing(value: str) -> bool:
    return value in MISSING_TOKENS


def parse_int(value: str) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_bool(value: str) -> bool | None:
    if is_missing(value):
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1"}:
        return True
    if normalized in {"false", "f", "0"}:
        return False
    return None


def normalize_seqn(value: str) -> str:
    stripped = value.strip()
    if is_missing(stripped):
        return ""
    try:
        numeric = float(stripped)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return stripped


def quantile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summary_stats(values: list[float]) -> dict[str, float | int]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    clean.sort()
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "p01": quantile(clean, 0.01),
        "p05": quantile(clean, 0.05),
        "p10": quantile(clean, 0.10),
        "p25": quantile(clean, 0.25),
        "p50": quantile(clean, 0.50),
        "p75": quantile(clean, 0.75),
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "p99": quantile(clean, 0.99),
        "min": clean[0],
        "max": clean[-1],
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def pct(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else math.nan


def day_type_for(seqn: str, day: int | None, minmax_by_seqn: dict[str, tuple[int, int]]) -> str:
    if day is None or seqn not in minmax_by_seqn:
        return "unknown"
    min_day, max_day = minmax_by_seqn[seqn]
    if day == min_day and day == max_day:
        return "single_day_first_and_last"
    if day == min_day:
        return "first_day"
    if day == max_day:
        return "last_day"
    return "interior_day"


def edge_group(day_type: str) -> str:
    if day_type == "interior_day":
        return "interior_day"
    if day_type in {"first_day", "last_day", "single_day_first_and_last"}:
        return "edge_day"
    return "unknown"


def read_day_minmax(troiano_path: Path) -> dict[str, tuple[int, int]]:
    ranges: dict[str, list[int]] = {}
    with open_xz_text(troiano_path) as handle:
        header = handle.readline().rstrip("\n\r").split(",")
        if header[:3] != ["SEQN", "PAXDAYM", "PAXDAYWM"]:
            raise ValueError("Unexpected troianowear header.")
        for line in handle:
            parts = line.rstrip("\n\r").split(",", 3)
            if len(parts) < 3:
                continue
            seqn = normalize_seqn(parts[0])
            day = parse_int(parts[1])
            if not seqn or day is None:
                continue
            if seqn not in ranges:
                ranges[seqn] = [day, day]
            else:
                ranges[seqn][0] = min(ranges[seqn][0], day)
                ranges[seqn][1] = max(ranges[seqn][1], day)
    return {seqn: (days[0], days[1]) for seqn, days in ranges.items()}


def collect_troiano_daily(
    troiano_path: Path,
    minmax_by_seqn: dict[str, tuple[int, int]],
) -> tuple[dict[tuple[str, int], dict[str, object]], Counter, Counter]:
    daily: dict[tuple[str, int], dict[str, object]] = {}
    weekday_counts: Counter = Counter()
    day_type_counts: Counter = Counter()

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
            weekday = parse_int(parts[2])
            if not seqn or day is None:
                continue
            missing_minutes = 0
            wear_minutes = 0
            false_minutes = 0
            invalid_minutes = 0
            for value in parts[3:]:
                observed = parse_bool(value)
                if observed is None:
                    missing_minutes += 1
                elif observed:
                    wear_minutes += 1
                else:
                    false_minutes += 1
            n_cells = len(parts[3:])
            invalid_minutes = max(MINUTES_PER_DAY - n_cells, 0)
            missing_total = missing_minutes + invalid_minutes
            denominator_excluding_missing = max(MINUTES_PER_DAY - missing_total, 0)
            day_type = day_type_for(seqn, day, minmax_by_seqn)
            edge = edge_group(day_type)
            is_weekend_assumed = weekday in {1, 7} if weekday is not None else False
            weekday_counts[str(weekday) if weekday is not None else "missing"] += 1
            day_type_counts[day_type] += 1
            daily[(seqn, day)] = {
                "seqn": seqn,
                "paxdaym": day,
                "paxdaywm": weekday,
                "day_type": day_type,
                "edge_group": edge,
                "is_weekend_assumed": is_weekend_assumed,
                "troiano_wear_minutes": wear_minutes,
                "troiano_nonwear_minutes": false_minutes + missing_total,
                "troiano_missing_minutes": missing_total,
                "troiano_observed_minutes": denominator_excluding_missing,
                "troiano_wear_fraction_missing_as_nonwear": wear_minutes / MINUTES_PER_DAY,
                "troiano_wear_fraction_missing_excluded": wear_minutes / denominator_excluding_missing
                if denominator_excluding_missing
                else math.nan,
            }
    return daily, weekday_counts, day_type_counts


def add_paxpredm_daily(
    paxpredm_path: Path,
    daily: dict[tuple[str, int], dict[str, object]],
) -> Counter:
    value_counts: Counter = Counter()
    with open_xz_text(paxpredm_path) as handle:
        header = handle.readline().rstrip("\n\r").split(",")
        if header[:3] != ["SEQN", "PAXDAYM", "PAXDAYWM"]:
            raise ValueError("Unexpected PAXPREDM header.")
        for line in handle:
            parts = line.rstrip("\n\r").split(",")
            if len(parts) < 4:
                continue
            seqn = normalize_seqn(parts[0])
            day = parse_int(parts[1])
            if not seqn or day is None:
                continue
            counts = Counter()
            missing_minutes = 0
            for value in parts[3:]:
                if is_missing(value):
                    missing_minutes += 1
                    continue
                code = parse_int(value)
                if code is None:
                    missing_minutes += 1
                else:
                    counts[str(code)] += 1
                    value_counts[str(code)] += 1
            n_cells = len(parts[3:])
            missing_total = missing_minutes + max(MINUTES_PER_DAY - n_cells, 0)
            key = (seqn, day)
            if key not in daily:
                continue
            wear_12 = counts["1"] + counts["2"]
            unknown_4 = counts["4"]
            denom_unknown_excluded = max(MINUTES_PER_DAY - missing_total - unknown_4, 0)
            daily[key].update(
                {
                    "paxpredm_wear_minutes_12": wear_12,
                    "paxpredm_nonwear_minutes_3_unknown_as_nonwear": counts["3"] + unknown_4 + missing_total,
                    "paxpredm_nonwear_minutes_3_unknown_excluded": counts["3"] + missing_total,
                    "paxpredm_unknown_minutes_4": unknown_4,
                    "paxpredm_missing_minutes": missing_total,
                    "paxpredm_wear_fraction_unknown_as_nonwear": wear_12 / MINUTES_PER_DAY,
                    "paxpredm_wear_fraction_unknown_excluded": wear_12 / denom_unknown_excluded
                    if denom_unknown_excluded
                    else math.nan,
                    "paxpredm_denominator_unknown_excluded": denom_unknown_excluded,
                }
            )
    return value_counts


def read_subject_info(path: Path) -> dict[str, dict[str, object]]:
    subjects: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seqn = normalize_seqn(row.get("SEQN", ""))
            if not seqn:
                continue
            mec_weight_2yr = parse_float(row.get("full_sample_2_year_mec_exam_weight", ""))
            subjects[seqn] = {
                "data_release_cycle": row.get("data_release_cycle", ""),
                "gender": row.get("gender", ""),
                "age": parse_float(row.get("age_in_years_at_screening", "")),
                "mec_weight_4yr": mec_weight_2yr / 2 if mec_weight_2yr is not None else math.nan,
            }
    return subjects


def age_group(age: float | None) -> str:
    if age is None or math.isnan(age):
        return "missing"
    if age < 20:
        return "<20"
    if age < 40:
        return "20-39"
    if age < 60:
        return "40-59"
    return "60+"


def build_edge_day_rows(
    minmax_by_seqn: dict[str, tuple[int, int]],
    daily: dict[tuple[str, int], dict[str, object]],
) -> list[dict[str, object]]:
    subject_day_counts = Counter()
    for row in daily.values():
        subject_day_counts[row["seqn"]] += 1
    total_subjects = len(minmax_by_seqn)
    n_single_day_subjects = sum(1 for seqn, count in subject_day_counts.items() if count == 1)
    rows = []
    day_type_counts = Counter(row["day_type"] for row in daily.values())
    for day_type in ["first_day", "last_day", "single_day_first_and_last", "interior_day", "unknown"]:
        n_days = day_type_counts[day_type]
        rows.append(
            {
                "summary_level": "day_type",
                "category": day_type,
                "n_days": n_days,
                "pct_days": pct(n_days, len(daily)),
                "n_subjects": "",
                "pct_subjects": "",
                "notes": "first and last are defined from each SEQN's observed PAXDAYM min/max.",
            }
        )
    rows.append(
        {
            "summary_level": "subject",
            "category": "subjects_with_single_observed_day",
            "n_days": "",
            "pct_days": "",
            "n_subjects": n_single_day_subjects,
            "pct_subjects": pct(n_single_day_subjects, total_subjects),
            "notes": "A single observed day is both first and last edge day.",
        }
    )
    rows.append(
        {
            "summary_level": "subject",
            "category": "all_subjects_with_troiano_days",
            "n_days": "",
            "pct_days": "",
            "n_subjects": total_subjects,
            "pct_subjects": 1.0,
            "notes": "",
        }
    )
    return rows


def build_wear_distribution_rows(daily: dict[tuple[str, int], dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    missing_groups: dict[tuple[str, str], list[float]] = defaultdict(list)

    def group_labels(row: dict[str, object]) -> list[str]:
        labels = [str(row["edge_group"])]
        day_type = str(row["day_type"])
        if day_type not in labels:
            labels.append(day_type)
        return labels

    for row in daily.values():
        for label in group_labels(row):
            groups[("troianowear", "missing_as_nonwear", label)].append(float(row["troiano_wear_minutes"]))
            groups[("troianowear", "missing_excluded", label)].append(
                float(row["troiano_wear_fraction_missing_excluded"]) * MINUTES_PER_DAY
            )
            missing_groups[("troianowear", label)].append(float(row["troiano_missing_minutes"]))

        if "paxpredm_wear_minutes_12" in row:
            for label in group_labels(row):
                groups[("PAXPREDM_3_nonwear", "unknown_as_nonwear", label)].append(
                    float(row["paxpredm_wear_minutes_12"])
                )
                groups[("PAXPREDM_3_nonwear", "unknown_excluded", label)].append(
                    float(row["paxpredm_wear_fraction_unknown_excluded"]) * MINUTES_PER_DAY
                )
                missing_groups[("PAXPREDM_3_nonwear_unknown4", label)].append(
                    float(row["paxpredm_unknown_minutes_4"])
                )

    rows: list[dict[str, object]] = []
    for (wear_source, denominator_policy, day_group), values in sorted(groups.items()):
        stats = summary_stats(values)
        rows.append(
            {
                "wear_source": wear_source,
                "denominator_policy": denominator_policy,
                "day_group": day_group,
                **stats,
                "notes": "missing/unknown-as-nonwear rows are daily wear-minute counts; denominator-excluded rows are 1440-scaled wear equivalents. Retention tables are wear-only projections, not SAP section 3 final analytic N.",
            }
        )
    for (metric, day_group), values in sorted(missing_groups.items()):
        stats = summary_stats(values)
        rows.append(
            {
                "wear_source": metric,
                "denominator_policy": "minute_count_distribution",
                "day_group": day_group,
                **stats,
                "notes": "Distribution of missing or unknown minute counts for denominator sensitivity checks.",
            }
        )
    return rows


def is_day_valid(
    row: dict[str, object],
    wear_source: str,
    denominator_policy: str,
    threshold: int,
) -> bool:
    if wear_source == "troianowear":
        if denominator_policy == "missing_as_nonwear":
            return float(row["troiano_wear_minutes"]) >= threshold
        if denominator_policy == "missing_excluded":
            denominator = float(row["troiano_observed_minutes"])
            if denominator <= 0:
                return False
            return float(row["troiano_wear_fraction_missing_excluded"]) * MINUTES_PER_DAY >= threshold
    if wear_source == "PAXPREDM_3_nonwear":
        if "paxpredm_wear_minutes_12" not in row:
            return False
        if denominator_policy == "unknown_as_nonwear":
            return float(row["paxpredm_wear_minutes_12"]) >= threshold
        if denominator_policy == "unknown_excluded":
            denominator = float(row["paxpredm_denominator_unknown_excluded"])
            if denominator <= 0:
                return False
            return float(row["paxpredm_wear_fraction_unknown_excluded"]) * MINUTES_PER_DAY >= threshold
    raise ValueError(f"Unknown wear source/policy: {wear_source} / {denominator_policy}")


def build_candidate_rule_rows(
    daily: dict[tuple[str, int], dict[str, object]],
    subjects: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    all_subjects = {row["seqn"] for row in daily.values()}
    rule_inclusions: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []
    source_policies = [
        ("troianowear", "missing_as_nonwear", "primary"),
        (
            "troianowear",
            "missing_excluded",
            "missing-denominator sensitivity; threshold applied to 1440-scaled observed-minute wear fraction",
        ),
        ("PAXPREDM_3_nonwear", "unknown_as_nonwear", "PAXPREDM sensitivity; code 3=nonwear, code 4=unknown assumed"),
        (
            "PAXPREDM_3_nonwear",
            "unknown_excluded",
            "PAXPREDM sensitivity; code 3=nonwear, code 4=unknown assumed; threshold applied to 1440-scaled non-unknown-minute wear fraction",
        ),
    ]
    for wear_source, denominator_policy, source_notes in source_policies:
        for edge_policy in EDGE_DAY_POLICIES:
            eligible_rows = [
                row
                for row in daily.values()
                if not (edge_policy == "drop_edge_days" and row["edge_group"] == "edge_day")
            ]
            for threshold in WEAR_THRESHOLDS:
                valid_by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {"valid_days": 0, "weekend_days": 0})
                for row in eligible_rows:
                    if is_day_valid(row, wear_source, denominator_policy, threshold):
                        seqn = str(row["seqn"])
                        valid_by_subject[seqn]["valid_days"] += 1
                        if bool(row["is_weekend_assumed"]):
                            valid_by_subject[seqn]["weekend_days"] += 1
                for min_days in MIN_VALID_DAYS:
                    for weekend_requirement in WEEKEND_REQUIREMENTS:
                        included = {
                            seqn
                            for seqn, counts in valid_by_subject.items()
                            if counts["valid_days"] >= min_days
                            and (
                                weekend_requirement == "none"
                                or counts["weekend_days"] >= 1
                            )
                        }
                        rule_id = (
                            f"{wear_source}|{denominator_policy}|{edge_policy}|"
                            f"wear_ge_{threshold}|valid_days_ge_{min_days}|weekend_{weekend_requirement}"
                        )
                        rule_inclusions[rule_id] = included
                        rows.append(
                            {
                                "rule_id": rule_id,
                                "wear_source": wear_source,
                                "denominator_policy": denominator_policy,
                                "edge_day_policy": edge_policy,
                                "daily_wear_threshold_minutes": threshold,
                                "min_valid_days": min_days,
                                "weekend_requirement": weekend_requirement,
                                "n_subjects_wear_only": len(included),
                                "pct_subjects_wear_only": pct(len(included), len(all_subjects)),
                                "n_subjects_total_troiano_days": len(all_subjects),
                                "note_projection_not_final_n": "wear-only projection, not SAP section 3 final analytic intersection N",
                                "coding_caveat": source_notes,
                            }
                        )
    return rows, rule_inclusions


def build_weekday_rows(daily: dict[tuple[str, int], dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(row["paxdaywm"]) if row["paxdaywm"] is not None else "missing" for row in daily.values())
    total = sum(counts.values())
    rows: list[dict[str, object]] = []
    labels = {
        "1": "Sunday_assumed",
        "2": "Monday_assumed",
        "3": "Tuesday_assumed",
        "4": "Wednesday_assumed",
        "5": "Thursday_assumed",
        "6": "Friday_assumed",
        "7": "Saturday_assumed",
        "missing": "missing",
    }
    for code in ["1", "2", "3", "4", "5", "6", "7", "missing"]:
        n = counts[code]
        rows.append(
            {
                "paxdaywm_code": code,
                "assumed_label": labels[code],
                "n_person_days": n,
                "pct_person_days": pct(n, total),
                "weekend_assumed": code in {"1", "7"},
                "coding_caveat": "PAXDAYWM 1=Sunday/7=Saturday from dataset README; label kept as assumed pending dataset/CDC codebook confirmation.",
            }
        )
    weekend_n = counts["1"] + counts["7"]
    rows.append(
        {
            "paxdaywm_code": "1_or_7",
            "assumed_label": "weekend_assumed",
            "n_person_days": weekend_n,
            "pct_person_days": pct(weekend_n, total),
            "weekend_assumed": True,
            "coding_caveat": "Weekend indicator is internally consistent under the assumed coding; labels remain pending codebook confirmation.",
        }
    )
    return rows


def weighted_mean(values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _value, weight in values if not math.isnan(weight))
    if total_weight <= 0:
        return math.nan
    return sum(value * weight for value, weight in values if not math.isnan(weight)) / total_weight


def build_included_excluded_rows(
    subjects: dict[str, dict[str, object]],
    rule_inclusions: dict[str, set[str]],
    candidate_rule_id: str,
) -> list[dict[str, object]]:
    included = rule_inclusions.get(candidate_rule_id, set())
    all_subjects = set(subjects)
    rows: list[dict[str, object]] = []
    match_summary = [
        ("candidate_included_total", len(included)),
        ("candidate_included_with_subject_info", len(included & all_subjects)),
        ("candidate_included_without_subject_info", len(included - all_subjects)),
        ("subject_info_total", len(all_subjects)),
        ("subject_info_not_candidate_included", len(all_subjects - included)),
    ]
    for level, n_subjects in match_summary:
        rows.append(
            {
                "candidate_rule_id": candidate_rule_id,
                "included_excluded_status": "match_summary",
                "group_var": "subject_info_match_summary",
                "group_level": level,
                "unweighted_n": n_subjects,
                "weighted_n_mec4yr_approx": "",
                "weighted_mean_age_if_applicable": "",
                "caveat": "Illustrative only: D9 is not locked. Subject-info profile covers matched SEQN only and may not equal the candidate wear-only N. Effective wrist-step sample is not a MEC probability subsample; MEC-weighted contrast is approximate.",
            }
        )
    groups = [
        ("all", lambda _seqn, _info: "all"),
        ("data_release_cycle", lambda _seqn, info: str(info["data_release_cycle"]) or "missing"),
        ("gender", lambda _seqn, info: str(info["gender"]) or "missing"),
        ("age_group", lambda _seqn, info: age_group(info["age"])),
    ]
    for group_var, group_fn in groups:
        for status, seqns in [
            ("included", included),
            ("excluded", all_subjects - included),
        ]:
            bucket: dict[str, list[str]] = defaultdict(list)
            for seqn in seqns:
                info = subjects.get(seqn)
                if not info:
                    continue
                bucket[group_fn(seqn, info)].append(seqn)
            for level, level_seqns in sorted(bucket.items()):
                weights = [float(subjects[seqn]["mec_weight_4yr"]) for seqn in level_seqns]
                weighted_n = sum(w for w in weights if not math.isnan(w))
                ages = [
                    (float(subjects[seqn]["age"]), float(subjects[seqn]["mec_weight_4yr"]))
                    for seqn in level_seqns
                    if subjects[seqn]["age"] is not None and not math.isnan(float(subjects[seqn]["age"]))
                ]
                rows.append(
                    {
                        "candidate_rule_id": candidate_rule_id,
                        "included_excluded_status": status,
                        "group_var": group_var,
                        "group_level": level,
                        "unweighted_n": len(level_seqns),
                        "weighted_n_mec4yr_approx": weighted_n,
                        "weighted_mean_age_if_applicable": weighted_mean(ages) if group_var == "all" else "",
                        "caveat": "Illustrative only: D9 is not locked. Subject-info profile covers matched SEQN only and may not equal the candidate wear-only N. Effective wrist-step sample is not a MEC probability subsample; MEC-weighted contrast is approximate.",
                    }
                )
    return rows


def choose_illustrative_rule(candidate_rows: list[dict[str, object]]) -> str:
    preferred = [
        row
        for row in candidate_rows
        if row["wear_source"] == "troianowear"
        and row["denominator_policy"] == "missing_as_nonwear"
        and row["edge_day_policy"] == "drop_edge_days"
        and row["daily_wear_threshold_minutes"] == 960
        and row["min_valid_days"] == 3
        and row["weekend_requirement"] == "none"
    ]
    if preferred:
        return str(preferred[0]["rule_id"])
    return str(candidate_rows[0]["rule_id"]) if candidate_rows else ""


def write_brief(
    path: Path,
    edge_rows: list[dict[str, object]],
    wear_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    weekday_rows: list[dict[str, object]],
    illustrative_rule_id: str,
) -> None:
    primary_rows = [
        row
        for row in candidate_rows
        if row["wear_source"] == "troianowear" and row["denominator_policy"] == "missing_as_nonwear"
    ]
    primary_drop = [
        row
        for row in primary_rows
        if row["edge_day_policy"] == "drop_edge_days"
        and row["weekend_requirement"] == "none"
        and row["min_valid_days"] in {3, 4}
    ]
    primary_drop.sort(
        key=lambda row: (
            int(row["daily_wear_threshold_minutes"]),
            int(row["min_valid_days"]),
            str(row["weekend_requirement"]),
        )
    )
    weekend_comparisons = [
        row
        for row in primary_rows
        if row["edge_day_policy"] == "drop_edge_days"
        and row["daily_wear_threshold_minutes"] == 960
        and row["min_valid_days"] == 3
    ]
    generated = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        "# Gate 0 D9 Wear Audit Decision Brief",
        "",
        f"- Generated: {generated}",
        "- Scope: PhysioNet wear/quality fields only; no metabolic outcomes linked and no exposure-outcome models run.",
        "- Guardrail: all retention counts are **wear-only projections, not SAP section 3 final analytic intersection N**.",
        "- Primary wear口径: `troianowear == TRUE`; missing minutes counted as nonwear.",
        "- Sensitivities: Troiano missing minutes excluded from denominator; PAXPREDM assumes code 3=nonwear and code 4=unknown, with unknown-as-nonwear and unknown-excluded variants.",
        "- Coding caveats: PAXPREDM 3/4 and PAXDAYWM 1=Sunday/7=Saturday are treated as assumed pending dataset/CDC codebook confirmation.",
        "- Edge-day handling: first/last observed `PAXDAYM` per SEQN are edge days; single observed day is both first and last.",
        "- Weekend: assumed `PAXDAYWM in {1,7}`; day-of-week distribution is a sanity check, not proof of labels.",
        "",
        "## Edge-Day Counts",
        "",
        "| category | n days | pct days | n subjects | pct subjects |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in edge_rows:
        lines.append(
            f"| {row['category']} | {row['n_days']} | {fmt(row['pct_days'], 4)} | {row['n_subjects']} | {fmt(row['pct_subjects'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Wear Distribution Snapshot",
            "",
            "| wear source | policy | day group | n | p25 | median | p75 | p95 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    snapshot_groups = {
        ("troianowear", "missing_as_nonwear", "edge_day"),
        ("troianowear", "missing_as_nonwear", "interior_day"),
        ("troianowear", "missing_excluded", "edge_day"),
        ("troianowear", "missing_excluded", "interior_day"),
        ("PAXPREDM_3_nonwear", "unknown_as_nonwear", "edge_day"),
        ("PAXPREDM_3_nonwear", "unknown_as_nonwear", "interior_day"),
    }
    for row in wear_rows:
        key = (row["wear_source"], row["denominator_policy"], row["day_group"])
        if key in snapshot_groups:
            lines.append(
                f"| {row['wear_source']} | {row['denominator_policy']} | {row['day_group']} | {row['n']} | {fmt(row['p25'], 1)} | {fmt(row['p50'], 1)} | {fmt(row['p75'], 1)} | {fmt(row['p95'], 1)} |"
            )
    lines.extend(
        [
            "",
            "## Primary Candidate Grid Snapshot",
            "",
            "Primary snapshot below uses Troiano, missing-as-nonwear, edge days dropped, no weekend requirement unless stated.",
            "",
            "| wear >= min/day | min valid days | weekend | n subjects | pct subjects |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for row in primary_drop:
        lines.append(
            f"| {row['daily_wear_threshold_minutes']} | {row['min_valid_days']} | {row['weekend_requirement']} | {row['n_subjects_wear_only']} | {fmt(row['pct_subjects_wear_only'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Weekend Sensitivity Snapshot",
            "",
            "Snapshot uses Troiano, missing-as-nonwear, edge days dropped, wear >=960, min valid days >=3.",
            "",
            "| weekend requirement | n subjects | pct subjects |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(weekend_comparisons, key=lambda item: str(item["weekend_requirement"])):
        lines.append(
            f"| {row['weekend_requirement']} | {row['n_subjects_wear_only']} | {fmt(row['pct_subjects_wear_only'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Day-of-Week Sanity Check",
            "",
            "| PAXDAYWM | assumed label | n days | pct days | caveat |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in weekday_rows:
        lines.append(
            f"| {row['paxdaywm_code']} | {row['assumed_label']} | {row['n_person_days']} | {fmt(row['pct_person_days'], 4)} | assumed pending codebook |"
        )
    lines.extend(
        [
            "",
            "## Recommendation For Human Review",
            "",
            "- Do not lock D9 from this script alone; inspect the CSV grid and decide after aggregate review.",
            "- A wrist-appropriate starting candidate is usually the Troiano primary arm with edge days dropped and a high daily wear threshold (e.g. >=960 min), then choose min valid days/weekend requirement by retention and selection-missingness tradeoff.",
            "- Weekend requirement should remain a sensitivity unless retention loss is modest and the codebook confirmation is clean.",
            f"- Included/excluded contrast is illustrative only and uses candidate rule: `{illustrative_rule_id}`.",
            "- Included/excluded contrast profiles the subject-info matched subset only; see `gate0_d9_included_excluded_subjectinfo_weighted.csv` match-summary rows before interpreting weighted differences.",
            "- SAP v1.2 and the final D9 lock remain pending human decision.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_tables = out_dir / "tables"
    out_logs = out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    troiano_path = locate_file(raw_dir, "nhanes_1440_troianowear.csv.xz")
    paxpredm_path = locate_file(raw_dir, "nhanes_1440_PAXPREDM.csv.xz")
    subject_info_path = raw_dir / "subject-info.csv"

    started = datetime.now(timezone.utc)
    minmax_by_seqn = read_day_minmax(troiano_path)
    daily, _weekday_counts, _day_type_counts = collect_troiano_daily(troiano_path, minmax_by_seqn)
    paxpredm_value_counts = add_paxpredm_daily(paxpredm_path, daily)
    subjects = read_subject_info(subject_info_path)

    edge_rows = build_edge_day_rows(minmax_by_seqn, daily)
    wear_rows = build_wear_distribution_rows(daily)
    candidate_rows, rule_inclusions = build_candidate_rule_rows(daily, subjects)
    weekday_rows = build_weekday_rows(daily)
    illustrative_rule_id = choose_illustrative_rule(candidate_rows)
    included_excluded_rows = build_included_excluded_rows(subjects, rule_inclusions, illustrative_rule_id)

    write_csv(
        out_tables / "gate0_d9_edge_day_counts.csv",
        edge_rows,
        ["summary_level", "category", "n_days", "pct_days", "n_subjects", "pct_subjects", "notes"],
    )
    write_csv(
        out_tables / "gate0_d9_wear_distribution_by_day_type.csv",
        wear_rows,
        [
            "wear_source",
            "denominator_policy",
            "day_group",
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
            "notes",
        ],
    )
    write_csv(
        out_tables / "gate0_d9_candidate_rule_retention.csv",
        candidate_rows,
        [
            "rule_id",
            "wear_source",
            "denominator_policy",
            "edge_day_policy",
            "daily_wear_threshold_minutes",
            "min_valid_days",
            "weekend_requirement",
            "n_subjects_wear_only",
            "pct_subjects_wear_only",
            "n_subjects_total_troiano_days",
            "note_projection_not_final_n",
            "coding_caveat",
        ],
    )
    write_csv(
        out_tables / "gate0_d9_weekday_weekend_summary.csv",
        weekday_rows,
        [
            "paxdaywm_code",
            "assumed_label",
            "n_person_days",
            "pct_person_days",
            "weekend_assumed",
            "coding_caveat",
        ],
    )
    write_csv(
        out_tables / "gate0_d9_included_excluded_subjectinfo_weighted.csv",
        included_excluded_rows,
        [
            "candidate_rule_id",
            "included_excluded_status",
            "group_var",
            "group_level",
            "unweighted_n",
            "weighted_n_mec4yr_approx",
            "weighted_mean_age_if_applicable",
            "caveat",
        ],
    )
    write_brief(
        out_tables / "GATE0_D9_DECISION_BRIEF.md",
        edge_rows=edge_rows,
        wear_rows=wear_rows,
        candidate_rows=candidate_rows,
        weekday_rows=weekday_rows,
        illustrative_rule_id=illustrative_rule_id,
    )

    finished = datetime.now(timezone.utc)
    log = {
        "script": "scripts/python/gate0_d9_wear_audit.py",
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "raw_dir": str(raw_dir),
        "outputs": [
            "outputs/tables/gate0_d9_edge_day_counts.csv",
            "outputs/tables/gate0_d9_wear_distribution_by_day_type.csv",
            "outputs/tables/gate0_d9_candidate_rule_retention.csv",
            "outputs/tables/gate0_d9_weekday_weekend_summary.csv",
            "outputs/tables/gate0_d9_included_excluded_subjectinfo_weighted.csv",
            "outputs/tables/GATE0_D9_DECISION_BRIEF.md",
            "outputs/logs/gate0_d9_wear_audit_run.json",
        ],
        "guardrails": [
            "aggregate outputs only",
            "no metabolic outcomes linked",
            "no exposure-outcome models run",
            "retention counts are wear-only projections, not SAP section 3 final N",
            "D9 not locked and SAP not modified",
        ],
        "candidate_grid": {
            "wear_threshold_minutes": WEAR_THRESHOLDS,
            "min_valid_days": MIN_VALID_DAYS,
            "weekend_requirements": WEEKEND_REQUIREMENTS,
            "edge_day_policies": EDGE_DAY_POLICIES,
            "wear_sources_and_denominator_policies": [
                "troianowear/missing_as_nonwear",
                "troianowear/missing_excluded",
                "PAXPREDM_3_nonwear/unknown_as_nonwear",
                "PAXPREDM_3_nonwear/unknown_excluded",
            ],
        },
        "n_person_days": len(daily),
        "n_subjects_with_troiano_days": len(minmax_by_seqn),
        "n_subjects_in_subject_info": len(subjects),
        "paxpredm_value_counts": dict(sorted(paxpredm_value_counts.items())),
        "illustrative_included_excluded_rule": illustrative_rule_id,
    }
    with (out_logs / "gate0_d9_wear_audit_run.json").open("w", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
