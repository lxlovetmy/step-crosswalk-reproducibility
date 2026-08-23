#!/usr/bin/env python3
"""Build exact isotonic mappings and one direction-metadata row per mapping.

Each direction is refitted from the frozen all-seven-algorithm cohort. The
fitted isotonic thresholds are the sole formal conversion parameters. Outputs
are aggregate fitted-model resources only; no participant-level rows or
predictions are written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_mvp_crosswalk_stability import HEADLINE_METHOD, fit_crosswalk
from build_stage2c_all7_subgroup_stability import build_daily_cohort
from release_common import (
    ALGORITHM_LABELS,
    DIRECTED_PAIRS,
    assert_pair_set,
    sha256,
    write_csv,
    write_json,
)


RESOURCE_VERSION = "1.1.1"
NUMERIC_TOLERANCE = 1e-8


def log(message: str) -> None:
    print(f"[crosswalk-resource {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pair-summary-csv",
        type=Path,
        help="Existing E/H pair summary; defaults to OUT_DIR/tables/stage3_translatability_map.csv.",
    )
    parser.add_argument("--resource-version", default=RESOURCE_VERSION)
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_pair_summary(path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(require(path))
    assert_pair_set(pairs.to_dict("records"), "E/H pair summary")
    if len(pairs) != len(DIRECTED_PAIRS):
        raise RuntimeError(f"Expected 42 E/H summary rows, got {len(pairs)}.")
    if pairs.groupby(["source_algorithm", "target_algorithm"]).size().ne(1).any():
        raise RuntimeError("Expected one E/H summary row per direction.")
    return pairs


def validate_exact_mapping(estimator: object, exact_x: np.ndarray, exact_y: np.ndarray) -> float:
    if len(exact_x) < 2 or len(exact_x) != len(exact_y):
        raise RuntimeError("Exact isotonic resource must contain at least two paired thresholds.")
    if np.any(~np.isfinite(exact_x)) or np.any(~np.isfinite(exact_y)):
        raise RuntimeError("Exact isotonic thresholds must be finite.")
    if np.any(np.diff(exact_x) <= 0) or np.any(np.diff(exact_y) < -NUMERIC_TOLERANCE):
        raise RuntimeError("Exact isotonic thresholds must be strictly ordered and monotone.")
    if np.any(exact_y < 0):
        raise RuntimeError("Exact isotonic predictions must be nonnegative.")
    model_predict = getattr(estimator, "predict", None)
    if model_predict is None:
        raise TypeError("Fitted model does not expose predict().")
    error = np.abs(np.asarray(model_predict(exact_x), dtype=float) - exact_y)
    maximum = float(np.max(error))
    if maximum > NUMERIC_TOLERANCE:
        raise RuntimeError(f"Exact thresholds failed model reconstruction: {maximum} steps.")
    return maximum


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    tables = args.out_dir / "tables"
    logs = args.out_dir / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    pair_path = args.pair_summary_csv or tables / "stage3_translatability_map.csv"
    pair_summary = load_pair_summary(pair_path)
    pair_lookup = pair_summary.set_index(["source_algorithm", "target_algorithm"])

    log("rebuilding the locked all-seven D9-valid cohort")
    cohort = build_daily_cohort(args.raw_dir)
    if cohort.subjects_n != 8646 or cohort.valid_days_n != 57080:
        raise RuntimeError(
            f"Crosswalk-resource cohort drift: expected 8646/57080, got "
            f"{cohort.subjects_n}/{cohort.valid_days_n}."
        )

    knot_rows: list[dict[str, object]] = []
    metadata_seed_rows: list[dict[str, object]] = []
    reconstruction_errors: list[float] = []

    for pair_number, (source, target) in enumerate(DIRECTED_PAIRS, start=1):
        x_all = np.asarray(cohort.daily_by_alg[source], dtype=float)
        y_all = np.asarray(cohort.daily_by_alg[target], dtype=float)
        valid = np.isfinite(x_all) & np.isfinite(y_all)
        x = x_all[valid]
        y = y_all[valid]
        fitted = fit_crosswalk(HEADLINE_METHOD, x, y)
        estimator = fitted.model
        if not hasattr(estimator, "X_thresholds_") or not hasattr(estimator, "y_thresholds_"):
            raise TypeError("Headline isotonic estimator does not expose fitted thresholds.")
        exact_x = np.asarray(estimator.X_thresholds_, dtype=float)
        exact_y = np.asarray(estimator.y_thresholds_, dtype=float)
        reconstruction_errors.append(validate_exact_mapping(estimator, exact_x, exact_y))

        released_low = float(np.quantile(x, 0.05))
        released_high = float(np.quantile(x, 0.95))
        if not (exact_x[0] <= released_low < released_high <= exact_x[-1]):
            raise RuntimeError(f"{source}->{target}: released P05-P95 support exceeds fitted knots.")

        for knot_index, (source_value, target_value) in enumerate(zip(exact_x, exact_y), start=1):
            knot_rows.append(
                {
                    "source_algorithm": source,
                    "target_algorithm": target,
                    "knot_index": knot_index,
                    "source_input_steps": float(source_value),
                    "predicted_target_steps": float(target_value),
                    "within_released_p05_p95_support": bool(
                        released_low <= source_value <= released_high
                    ),
                }
            )

        pair_row = pair_lookup.loc[(source, target)]
        metadata_seed_rows.append(
            {
                "resource_version": args.resource_version,
                "mapping_id": f"{source}_to_{target}",
                "source_algorithm": source,
                "source_algorithm_label": ALGORITHM_LABELS[source],
                "target_algorithm": target,
                "target_algorithm_label": ALGORITHM_LABELS[target],
                "direction_reversible": False,
                "input_unit": "steps/day",
                "output_unit": "steps/day",
                "analysis_level": "participant_multi_day_mean_daily_steps",
                "fit_method": "nonnegative_nondecreasing_isotonic_regression",
                "n_fit": int(len(x)),
                "training_source_min_steps": float(np.min(x)),
                "training_source_max_steps": float(np.max(x)),
                "released_source_p05_steps": released_low,
                "released_source_p95_steps": released_high,
                "released_support_rule": "pooled_source_P05_to_P95",
                "out_of_range_policy": "reject_no_formal_conversion",
                "exact_knot_nodes": int(len(exact_x)),
                "exact_knots_within_released_support": int(
                    np.sum((exact_x >= released_low) & (exact_x <= released_high))
                ),
                "conversion_interpolation_rule": "linear_between_exact_isotonic_thresholds",
                "rounding_rule": "round_after_interpolation_only_when_whole_steps_are_required",
                "oof_mae_target_steps": float(pair_row["mae"]),
                "target_iqr": float(pair_row["target_iqr"]),
                "E_oof_mae_per_target_iqr": float(pair_row["E"]),
                "H_full_range_max_subgroup_spread_per_target_iqr": float(pair_row["H"]),
                "H_driver": str(pair_row["max_subgroup_p95_spread_group_type"]),
                "exact_knots_file": "crosswalk_exact_knots.csv",
                "strict_converter_file": "analysis/crosswalk_converter.py",
                "table_s2_role": "direction_index_and_use_boundaries",
                "use_boundary": (
                    "Direction-specific re-expression of participant multi-day mean daily "
                    "steps only; not true-step accuracy, health-threshold equivalence, or "
                    "sample-marginal prevalence alignment."
                ),
            }
        )
        if pair_number == 1 or pair_number == len(DIRECTED_PAIRS) or pair_number % 7 == 0:
            log(f"direction {pair_number}/{len(DIRECTED_PAIRS)} complete")

    knots_path = tables / "crosswalk_exact_knots.csv"
    metadata_path = tables / "crosswalk_direction_metadata.csv"
    write_csv(knots_path, knot_rows, list(knot_rows[0]))
    knots_hash = sha256(knots_path)
    metadata_rows = [
        {**row, "exact_knots_sha256": knots_hash}
        for row in metadata_seed_rows
    ]
    write_csv(metadata_path, metadata_rows, list(metadata_rows[0]))

    max_exact_error = max(reconstruction_errors)
    write_json(
        logs / "crosswalk_resource_run.json",
        {
            "script": "analysis/build_crosswalk_resource.py",
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "resource_version": args.resource_version,
            "cohort_subjects": cohort.subjects_n,
            "cohort_valid_days": cohort.valid_days_n,
            "directed_pairs": len(DIRECTED_PAIRS),
            "exact_knot_rows": len(knot_rows),
            "metadata_rows": len(metadata_rows),
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "max_exact_knot_reconstruction_abs_error_steps": max_exact_error,
            "exact_knots_sha256": knots_hash,
            "outputs": [
                "tables/crosswalk_exact_knots.csv",
                "tables/crosswalk_direction_metadata.csv",
                "logs/crosswalk_resource_run.json",
            ],
            "guardrails": [
                "aggregate fitted-model resource only",
                "no participant identifiers or participant-level predictions",
                "42 directions are asymmetric and independently fitted",
                "formal conversion restricted to the released source P05-P95 range",
                "exact fitted thresholds are the sole formal conversion parameters",
            ],
        },
    )
    log(
        f"done; exact knots={len(knot_rows)}, metadata={len(metadata_rows)}, "
        f"max reconstruction error={max_exact_error:.3g} steps"
    )


if __name__ == "__main__":
    main()
