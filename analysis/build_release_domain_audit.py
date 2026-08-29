#!/usr/bin/env python3
"""Audit performance inside fold-derived released source P05-P95 domains.

This bounded sensitivity analysis does not trim model fitting. For the
whole-cohort analysis, every isotonic model is fitted on its complete training
fold, the source P05-P95 interval is derived from that training fold only, and
the held-out error is evaluated both overall and among held-out participants
whose source value lies inside that interval. For cross-cycle application, the
range is derived from the complete training cycle and both the transported and
within-test-cycle OOF predictions are evaluated on the same covered test
participants. Only aggregate direction-level rows are written.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from build_mvp_crosswalk_stability import FOLDS, HEADLINE_METHOD, RANDOM_SEED, fit_crosswalk
from build_stage2_all7_crosswalk_matrix import ALGORITHMS
from build_stage2c_all7_subgroup_stability import DailyCohort, build_daily_cohort
from build_stage6_cycle_threshold import CYCLES, align_design, grouped_oof_predictions
from release_common import assert_pair_set, write_csv, write_json


WHOLE_FILE = "release_domain_whole_cohort.csv"
CROSS_CYCLE_FILE = "release_domain_cross_cycle.csv"
SUMMARY_FILE = "RELEASE_DOMAIN_AUDIT_SUMMARY.md"
RUN_LOG_FILE = "release_domain_audit_run.json"


def log(message: str) -> None:
    print(f"[release-domain {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def target_iqr(values: np.ndarray) -> float:
    return float(
        np.quantile(values, 0.75, method="linear")
        - np.quantile(values, 0.25, method="linear")
    )


def error_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    denominator_iqr: float,
) -> dict[str, float]:
    absolute = np.abs(prediction - target)
    mae = float(np.mean(absolute))
    return {
        "mae": mae,
        "abs_error_p95": float(np.quantile(absolute, 0.95, method="linear")),
        "E": safe_divide(mae, denominator_iqr),
    }


def subject_grouped_folds(subject_ids: np.ndarray, seed: int):
    unique_subjects = np.asarray(sorted(set(subject_ids.tolist())), dtype=object)
    if len(unique_subjects) < FOLDS:
        raise RuntimeError(f"At least {FOLDS} unique participants are required.")
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for train_unique, test_unique in splitter.split(unique_subjects):
        train_subjects = set(unique_subjects[train_unique].tolist())
        test_subjects = set(unique_subjects[test_unique].tolist())
        train_mask = np.fromiter(
            (item in train_subjects for item in subject_ids), dtype=bool, count=len(subject_ids)
        )
        test_mask = np.fromiter(
            (item in test_subjects for item in subject_ids), dtype=bool, count=len(subject_ids)
        )
        if np.any(train_mask & test_mask):
            raise RuntimeError("A participant was assigned to both training and test data.")
        yield train_mask, test_mask


def whole_cohort_row(
    cohort: DailyCohort,
    source: str,
    target: str,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    x_all = np.asarray(cohort.daily_by_alg[source], dtype=float)
    y_all = np.asarray(cohort.daily_by_alg[target], dtype=float)
    ids_all = np.asarray(cohort.subject_ids, dtype=object)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    x, y, subject_ids = x_all[finite], y_all[finite], ids_all[finite]
    prediction = np.full(len(x), np.nan, dtype=float)
    covered = np.zeros(len(x), dtype=bool)
    fold_p05: list[float] = []
    fold_p95: list[float] = []

    for train_mask, test_mask in subject_grouped_folds(subject_ids, seed):
        lower, upper = np.quantile(x[train_mask], [0.05, 0.95], method="linear")
        fitted = fit_crosswalk(HEADLINE_METHOD, x[train_mask], y[train_mask])
        prediction[test_mask] = fitted.predict(x[test_mask])
        covered[test_mask] = (x[test_mask] >= lower) & (x[test_mask] <= upper)
        fold_p05.append(float(lower))
        fold_p95.append(float(upper))

    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Non-finite OOF prediction for {source}->{target}.")
    if not np.any(covered):
        raise RuntimeError(f"Empty release-domain evaluation for {source}->{target}.")

    denominator = target_iqr(y)
    global_result = error_summary(prediction, y, denominator)
    domain_result = error_summary(prediction[covered], y[covered], denominator)
    return {
        "scope": "whole_cohort_subject_grouped_5fold_oof",
        "method": HEADLINE_METHOD,
        "source_algorithm": source,
        "target_algorithm": target,
        "n_total": int(len(y)),
        "n_covered": int(np.sum(covered)),
        "coverage_pct": float(100 * np.mean(covered)),
        "complete_sample_target_iqr": denominator,
        "global_oof_mae": global_result["mae"],
        "global_oof_abs_error_p95": global_result["abs_error_p95"],
        "global_oof_E": global_result["E"],
        "release_domain_oof_mae": domain_result["mae"],
        "release_domain_oof_abs_error_p95": domain_result["abs_error_p95"],
        "release_domain_oof_E": domain_result["E"],
        "delta_E_domain_minus_global": domain_result["E"] - global_result["E"],
        "training_fold_source_p05_min": float(np.min(fold_p05)),
        "training_fold_source_p05_median": float(np.median(fold_p05)),
        "training_fold_source_p05_max": float(np.max(fold_p05)),
        "training_fold_source_p95_min": float(np.min(fold_p95)),
        "training_fold_source_p95_median": float(np.median(fold_p95)),
        "training_fold_source_p95_max": float(np.max(fold_p95)),
        "range_rule": "inclusive source P05-P95 derived from the training fold only",
        "fit_rule": "isotonic fit used the complete training fold without trimming",
        "interpretation": "release-domain sensitivity; not a validated range or performance improvement test",
    }


def cross_cycle_row(
    cohort: DailyCohort,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    source: str,
    target: str,
    train_cycle: str,
    test_cycle: str,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    x_train_all = np.asarray(cohort.daily_by_alg[source][train_idx], dtype=float)
    y_train_all = np.asarray(cohort.daily_by_alg[target][train_idx], dtype=float)
    x_test_all = np.asarray(cohort.daily_by_alg[source][test_idx], dtype=float)
    y_test_all = np.asarray(cohort.daily_by_alg[target][test_idx], dtype=float)
    ids_test_all = np.asarray(cohort.subject_ids[test_idx], dtype=object)
    train_finite = np.isfinite(x_train_all) & np.isfinite(y_train_all)
    test_finite = np.isfinite(x_test_all) & np.isfinite(y_test_all)
    x_train, y_train = x_train_all[train_finite], y_train_all[train_finite]
    x_test, y_test = x_test_all[test_finite], y_test_all[test_finite]
    test_ids = ids_test_all[test_finite]

    lower, upper = np.quantile(x_train, [0.05, 0.95], method="linear")
    covered = (x_test >= lower) & (x_test <= upper)
    if not np.any(covered):
        raise RuntimeError(f"Empty cross-cycle release domain for {source}->{target}.")
    locked_model = fit_crosswalk(HEADLINE_METHOD, x_train, y_train)
    transported = locked_model.predict(x_test)
    within_test_oof = grouped_oof_predictions(x_test, y_test, test_ids, seed)
    denominator = target_iqr(y_test)
    transport_all = error_summary(transported, y_test, denominator)
    within_all = error_summary(within_test_oof, y_test, denominator)
    transport_domain = error_summary(transported[covered], y_test[covered], denominator)
    within_domain = error_summary(within_test_oof[covered], y_test[covered], denominator)

    return {
        "scope": "cross_cycle_training_range_applied_to_test_cycle",
        "method": HEADLINE_METHOD,
        "source_algorithm": source,
        "target_algorithm": target,
        "train_cycle": train_cycle,
        "test_cycle": test_cycle,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_covered": int(np.sum(covered)),
        "coverage_pct": float(100 * np.mean(covered)),
        "train_source_p05": float(lower),
        "train_source_p95": float(upper),
        "complete_test_cycle_target_iqr": denominator,
        "transport_all_test_E": transport_all["E"],
        "transport_release_domain_mae": transport_domain["mae"],
        "transport_release_domain_abs_error_p95": transport_domain["abs_error_p95"],
        "transport_release_domain_E": transport_domain["E"],
        "transport_delta_E_domain_minus_all_test": transport_domain["E"] - transport_all["E"],
        "within_test_oof_all_test_E": within_all["E"],
        "within_test_oof_release_domain_mae": within_domain["mae"],
        "within_test_oof_release_domain_abs_error_p95": within_domain["abs_error_p95"],
        "within_test_oof_release_domain_E": within_domain["E"],
        "within_test_oof_delta_E_domain_minus_all_test": within_domain["E"] - within_all["E"],
        "transport_penalty_E_on_same_covered_test": transport_domain["E"] - within_domain["E"],
        "range_rule": "inclusive source P05-P95 derived from the complete training cycle only",
        "fit_rule": "transport isotonic fit used the complete training cycle without trimming",
        "comparison_rule": "transported and within-test OOF predictions evaluated on identical covered test participants",
        "interpretation": "release-domain sensitivity; not external validation or a validated range",
    }


def describe(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.min(array)), float(np.median(array)), float(np.max(array))


def write_summary(path: Path, whole_rows: list[dict[str, object]], cycle_rows: list[dict[str, object]]) -> None:
    whole_coverage = describe([float(row["coverage_pct"]) for row in whole_rows])
    whole_delta = describe([float(row["delta_E_domain_minus_global"]) for row in whole_rows])
    cycle_coverage = describe([float(row["coverage_pct"]) for row in cycle_rows])
    cycle_penalty = describe([float(row["transport_penalty_E_on_same_covered_test"]) for row in cycle_rows])
    lines = [
        "# P05-P95 Release-Domain Audit Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "- Scope: 42 whole-cohort OOF directions and 84 cross-cycle train-test directions.",
        "- Model fitting used complete training folds or cycles; P05-P95 restricted evaluation only.",
        "- All ranges were derived without access to the corresponding held-out observations.",
        "- Target-IQR denominators remain those of the complete cohort or complete test cycle.",
        "- No bootstrap, alternative estimator, repeated cross-validation, or external cohort was added.",
        "- These are release-domain sensitivities, not validated ranges or tests of improvement.",
        "",
        "## Aggregate checks",
        "",
        f"- Whole-cohort coverage range/median: {whole_coverage[0]:.2f}% / {whole_coverage[1]:.2f}% / {whole_coverage[2]:.2f}%.",
        f"- Whole-cohort domain-minus-global E range/median: {whole_delta[0]:.4f} / {whole_delta[1]:.4f} / {whole_delta[2]:.4f}.",
        f"- Cross-cycle coverage range/median: {cycle_coverage[0]:.2f}% / {cycle_coverage[1]:.2f}% / {cycle_coverage[2]:.2f}%.",
        f"- Cross-cycle transport penalty on the same covered test participants, range/median: {cycle_penalty[0]:.4f} / {cycle_penalty[1]:.4f} / {cycle_penalty[2]:.4f}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    out_tables = args.out_dir / "tables"
    out_logs = args.out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    log("rebuilding frozen D9 all-seven adult cohort")
    cohort = build_daily_cohort(args.raw_dir)
    if (cohort.subjects_n, cohort.valid_days_n) != (8646, 57080):
        raise RuntimeError(
            f"Cohort drift: expected 8646/57080, got {cohort.subjects_n}/{cohort.valid_days_n}."
        )
    cycles, _strata, _psus, _weights, _audit = align_design(cohort, args.raw_dir)
    cycle_indices = {cycle: np.flatnonzero(cycles == cycle) for cycle in CYCLES}
    pairs = list(itertools.permutations(ALGORITHMS, 2))

    log("computing 42 whole-cohort fold-derived release-domain rows")
    whole_rows = [whole_cohort_row(cohort, source, target) for source, target in pairs]
    assert_pair_set(whole_rows, "release-domain whole-cohort")

    log("computing 84 training-cycle release-domain rows")
    cycle_rows: list[dict[str, object]] = []
    for train_cycle, test_cycle in ((CYCLES[0], CYCLES[1]), (CYCLES[1], CYCLES[0])):
        for source, target in pairs:
            cycle_rows.append(
                cross_cycle_row(
                    cohort,
                    cycle_indices[train_cycle],
                    cycle_indices[test_cycle],
                    source,
                    target,
                    train_cycle,
                    test_cycle,
                )
            )
    if len(cycle_rows) != 84:
        raise RuntimeError(f"Expected 84 cross-cycle rows, got {len(cycle_rows)}.")

    write_csv(out_tables / WHOLE_FILE, whole_rows)
    write_csv(out_tables / CROSS_CYCLE_FILE, cycle_rows)
    write_summary(out_tables / SUMMARY_FILE, whole_rows, cycle_rows)
    write_json(
        out_logs / RUN_LOG_FILE,
        {
            "status": "PASS",
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "cohort_subjects": cohort.subjects_n,
            "cohort_valid_person_days": cohort.valid_days_n,
            "folds": FOLDS,
            "seed": RANDOM_SEED,
            "whole_cohort_rows": len(whole_rows),
            "cross_cycle_rows": len(cycle_rows),
            "quantile_method": "linear",
            "bootstrap_replicates": 0,
            "guardrails": [
                "fit complete training folds/cycles without trimming",
                "derive evaluation range from training data only",
                "aggregate rows only",
                "no participant IDs, folds, predictions, or bootstrap rows written",
                "not a validated range or improvement test",
            ],
        },
    )
    log("done")


if __name__ == "__main__":
    main()
