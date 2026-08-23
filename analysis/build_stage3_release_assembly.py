#!/usr/bin/env python3
"""Assemble the continuous direction-level E/H pair summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from release_common import ALGORITHMS, assert_pair_set


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    tables = args.out_dir / "tables"
    logs = args.out_dir / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    residual = pd.read_csv(require(tables / "stage2_all7_crosswalk_residual_matrix.csv"))
    spread = pd.read_csv(require(tables / "stage2c_all7_subgroup_curve_spread.csv"))
    divergence = pd.read_csv(require(tables / "stage2c_all7_subgroup_curve_divergence.csv"))

    residual = residual.loc[residual["method"].eq("isotonic")].copy()
    if len(residual) != 42:
        raise RuntimeError(f"Expected 42 isotonic residual rows, got {len(residual)}")
    assert_pair_set(residual.to_dict("records"), "residual matrix")

    spread = spread.loc[spread["group_type"].isin(["age_group", "sex", "bmi_group"])].copy()
    if len(spread) != 126:
        raise RuntimeError(f"Expected 126 subgroup spread rows, got {len(spread)}")
    raw_col = "p95_between_subgroup_spread_predicted_target"
    norm_col = "p95_between_subgroup_spread_predicted_target_over_target_iqr"
    spread_wide = spread.pivot(index=["source_algorithm", "target_algorithm"], columns="group_type", values=[raw_col, norm_col])
    spread_wide.columns = [f"{group.replace('_group', '')}_{'p95_spread_raw' if metric == raw_col else 'p95_spread_over_target_iqr'}" for metric, group in spread_wide.columns]
    spread_wide = spread_wide.reset_index()

    max_spread_rows = []
    for (source, target), part in spread.groupby(["source_algorithm", "target_algorithm"], sort=False):
        row = part.loc[part[norm_col].astype(float).idxmax()]
        max_spread_rows.append({
            "source_algorithm": source,
            "target_algorithm": target,
            "max_subgroup_p95_spread_group_type": row["group_type"],
            "max_subgroup_p95_spread_raw": float(row[raw_col]),
            "max_subgroup_p95_spread_over_target_iqr": float(row[norm_col]),
        })
    max_spread = pd.DataFrame(max_spread_rows)

    div_col = "p95_abs_subgroup_minus_full_predicted_target_over_target_iqr"
    div_rows = []
    for (source, target), part in divergence.groupby(["source_algorithm", "target_algorithm"], sort=False):
        row = part.loc[part[div_col].astype(float).idxmax()]
        div_rows.append({
            "source_algorithm": source,
            "target_algorithm": target,
            "max_divergence_group_type": row["group_type"],
            "max_divergence_group_level": row["group_level"],
            "max_divergence_p95_abs_raw": float(row["p95_abs_subgroup_minus_full_predicted_target"]),
            "max_divergence_p95_abs_over_target_iqr": float(row[div_col]),
        })
    max_div = pd.DataFrame(div_rows)

    pair = residual[[
        "exposure", "source_algorithm", "target_algorithm", "method", "n", "mae",
        "rmse", "abs_error_p95", "target_iqr", "mae_per_target_iqr",
    ]].merge(spread_wide, on=["source_algorithm", "target_algorithm"], validate="one_to_one")
    pair = pair.merge(max_spread, on=["source_algorithm", "target_algorithm"], validate="one_to_one")
    pair = pair.merge(max_div, on=["source_algorithm", "target_algorithm"], validate="one_to_one")
    pair["E"] = pair["mae_per_target_iqr"]
    pair["H"] = pair["max_subgroup_p95_spread_over_target_iqr"]
    pair["interpretation"] = "continuous_evidence_without_acceptability_grade"
    pair = pair.sort_values(["source_algorithm", "target_algorithm"]).reset_index(drop=True)
    assert_pair_set(pair.to_dict("records"), "pair summary")

    pair.to_csv(tables / "stage3_translatability_map.csv", index=False)
    (logs / "stage3_release_assembly_run.json").write_text(
        json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm_order": ALGORITHMS,
            "row_counts": {"pair_summary": len(pair)},
            "formal_crosswalk_resource": "built separately from exact fitted isotonic thresholds",
            "acceptability_grades_generated": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
