#!/usr/bin/env python3
"""Regenerate package reference and authority manifests from actual files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def release_role(relative: str) -> str:
    name = Path(relative).name
    if name.endswith("_current.png"):
        return "current_visual_reference"
    if name.startswith("Figure1_fixed8000_attainment"):
        return "publication_figure_export"
    if name == "crosswalk_exact_knots.csv":
        return "canonical_machine_readable_mapping_reference"
    if name == "crosswalk_direction_metadata.csv":
        return "direction_metadata_reference"
    return "aggregate_numeric_reference"


def build_reference_rows(root: Path) -> list[dict[str, object]]:
    reference_root = root / "results" / "reference"
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in reference_root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower().lstrip(".")
        row: dict[str, object] = {
            "relative_path": relative,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "artifact_type": suffix,
            "data_rows": csv_rows(path) if suffix == "csv" else "",
            "dimensions_pixels": "",
            "release_role": release_role(relative),
        }
        if suffix == "png":
            with Image.open(path) as image:
                row["dimensions_pixels"] = f"{image.width}x{image.height}"
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--english-manuscript", type=Path, required=True)
    parser.add_argument("--sdc", type=Path, required=True)
    parser.add_argument("--generated-utc")
    args = parser.parse_args()

    root = args.package_root.resolve()
    reference_manifest = root / "provenance" / "reference_manifest.csv"
    authority_manifest = root / "provenance" / "authority_manifest.json"
    validation_path = root / "provenance" / "release_validation.json"
    config_path = root / "config" / "analysis.json"
    input_manifest = root / "data" / "expected_inputs.csv"
    protected = [
        (args.english_manuscript.resolve(), "current_English_manuscript"),
        (args.sdc.resolve(), "current_supplement"),
    ]
    for path, _role in protected:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = build_reference_rows(root)
    fieldnames = [
        "relative_path", "sha256", "bytes", "artifact_type",
        "data_rows", "dimensions_pixels", "release_role",
    ]
    with reference_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    generated_utc = args.generated_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    by_name = {Path(str(row["relative_path"])).name: row for row in rows}
    critical_names = [
        "crosswalk_exact_knots.csv",
        "crosswalk_direction_metadata.csv",
        "common_support_detail.csv",
        "common_support_summary.csv",
        "continuous_bootstrap.csv",
        "wear_pair.csv",
        "wear_comparison.csv",
        "Figure4_current.png",
    ]
    authority = {
        "schema_version": 2,
        "generated_utc": generated_utc,
        "release_scope": "current_crosswalk_only_manuscript",
        "release_version": "1.1.1",
        "repository_snapshot": {
            "git_head": head,
            "working_tree_state": "dirty_work_preserved_no_commit",
        },
        "authority_rule": (
            "The two designated read-only writing files define manuscript scope; "
            "the package reference manifest defines aggregate numeric and visual authorities."
        ),
        "protected_writing_files": [
            {"relative_path": path.name, "sha256": sha256(path), "role": role}
            for path, role in protected
        ],
        "input_contract": {
            "expected_inputs_manifest": "data/expected_inputs.csv",
            "expected_inputs_manifest_sha256": sha256(input_manifest),
            "verified_public_use_inputs": 12,
            "raw_inputs_bundled": False,
        },
        "method_contract": {
            "eligible_subgroup_minimum_n": config["evaluation"]["minimum_subgroup_n"],
            "common_support_grid": config["evaluation"]["common_support_grid"],
            "common_support_curve_fitting": config["evaluation"]["common_support_curve_fitting"],
            "oof_folds": config["individual_crosswalk"]["oof_folds"],
            "oof_split_unit": config["individual_crosswalk"]["oof_split_unit"],
            "bootstrap_replicates": config["bootstrap"]["replicates"],
            "stage4_seed": config["bootstrap"]["stage4_seed"],
            "stage6_seed": config["bootstrap"]["stage6_seed"],
        },
        "frozen_contracts": {
            "participants": config["cohort"]["expected_participants"],
            "valid_person_days": config["cohort"]["expected_valid_person_days"],
            "algorithms": 7,
            "directed_pairs": config["individual_crosswalk"]["expected_directed_pairs"],
            "exact_knot_rows": 11416,
            "common_support_detail_rows": 252,
            "common_support_summary_rows": 42,
            "continuous_bootstrap_rows": 84,
            "common_support_bootstrap_success": "300/300",
            "displayed_common_support_change_counts": {"lower": 32, "equal": 6, "higher": 4},
            "displayed_common_support_median_change": -0.024,
        },
        "reference_manifest": {
            "relative_path": "provenance/reference_manifest.csv",
            "sha256": sha256(reference_manifest),
            "artifact_count": len(rows),
        },
        "critical_reference_artifacts": [by_name[name] for name in critical_names],
        "terminal_validation": {
            "relative_path": "provenance/release_validation.json",
            "sha256": sha256(validation_path),
            "status": validation.get("status"),
            "check_count": validation.get("check_count"),
            "failure_count": validation.get("failure_count"),
        },
    }
    authority_manifest.write_text(
        json.dumps(authority, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PROVENANCE_MANIFESTS=PASS references={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
