#!/usr/bin/env python3
"""Build the 21-row sample-marginal Oak anchor alignment table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_stage2c_all7_subgroup_stability import build_daily_cohort
from release_common import ALGORITHMS, OAK_ANCHORS, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = build_daily_cohort(args.raw_dir)
    if (cohort.subjects_n, cohort.valid_days_n) != (8646, 57080):
        raise RuntimeError(f"Cohort drift: {cohort.subjects_n}/{cohort.valid_days_n}")
    oak = np.asarray(cohort.daily_by_alg["oak"], dtype=float)
    rows: list[dict[str, object]] = []
    for anchor in OAK_ANCHORS:
        percentile = float(np.mean(oak <= anchor))
        for algorithm in ALGORITHMS:
            values = np.asarray(cohort.daily_by_alg[algorithm], dtype=float)
            equivalent = anchor if algorithm == "oak" else float(np.quantile(values, percentile))
            rows.append(
                {
                    "reference_algorithm": "oak",
                    "reference_threshold_steps": anchor,
                    "oak_percentile": percentile,
                    "algorithm": algorithm,
                    "equivalent_threshold_steps": equivalent,
                    "conversion_method": "unweighted_empirical_quantile",
                    "use_case": "sample_marginal_position_alignment_only",
                }
            )
    if len(rows) != 21:
        raise RuntimeError("Sample alignment must contain 21 rows")
    write_csv(args.out_dir / "tables" / "sample_alignment.csv", rows)


if __name__ == "__main__":
    main()
