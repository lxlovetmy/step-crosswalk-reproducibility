#!/usr/bin/env python3
"""Strict direction-specific converter for the released exact crosswalk resource.

The converter refuses unsupported directions and values outside the released
source P05-P95 range. It never silently clips or extrapolates.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


STATUS_IN_RANGE = "IN_RANGE"
STATUS_BELOW_RANGE = "OUT_OF_RANGE_BELOW"
STATUS_ABOVE_RANGE = "OUT_OF_RANGE_ABOVE"
STATUS_NONFINITE = "INVALID_NONFINITE_INPUT"


@dataclass(frozen=True)
class ConversionResult:
    source_algorithm: str
    target_algorithm: str
    source_value_steps: float
    converted_target_steps: float | None
    status: str
    released_source_low_steps: float
    released_source_high_steps: float
    resource_version: str


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid or missing numeric field {field!r}.") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def round_half_up_nonnegative(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("Only finite nonnegative crosswalk predictions can be rounded.")
    return int(math.floor(value + 0.5))


class CrosswalkResource:
    def __init__(self, metadata_path: Path, exact_knots_path: Path) -> None:
        metadata_rows = _read_csv(metadata_path)
        knot_rows = _read_csv(exact_knots_path)
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.knots: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

        for row in metadata_rows:
            key = (row.get("source_algorithm", ""), row.get("target_algorithm", ""))
            if not all(key) or key[0] == key[1]:
                raise ValueError(f"Invalid direction in metadata: {key!r}")
            if key in self.metadata:
                raise ValueError(f"Duplicate metadata direction: {key!r}")
            self.metadata[key] = row

        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in knot_rows:
            key = (row.get("source_algorithm", ""), row.get("target_algorithm", ""))
            grouped.setdefault(key, []).append(row)
        if set(grouped) != set(self.metadata):
            raise ValueError(
                "Metadata/exact-knot direction mismatch: "
                f"metadata={len(self.metadata)}, knots={len(grouped)}"
            )

        for key, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: int(row["knot_index"]))
            x = np.asarray([_float(row, "source_input_steps") for row in ordered], dtype=float)
            y = np.asarray([_float(row, "predicted_target_steps") for row in ordered], dtype=float)
            if len(x) < 2 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
                raise ValueError(f"Invalid exact knots for {key!r}.")
            if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < -1e-8) or np.any(y < 0):
                raise ValueError(f"Non-monotone exact knots for {key!r}.")
            expected = int(self.metadata[key]["exact_knot_nodes"])
            if len(x) != expected:
                raise ValueError(f"Exact-knot count mismatch for {key!r}: {len(x)} != {expected}.")
            low = _float(self.metadata[key], "released_source_p05_steps")
            high = _float(self.metadata[key], "released_source_p95_steps")
            if not (x[0] <= low < high <= x[-1]):
                raise ValueError(f"Released support is outside fitted knots for {key!r}.")
            self.knots[key] = (x, y)

    def available_directions(self) -> list[tuple[str, str]]:
        return sorted(self.metadata)

    def convert_one(
        self,
        source_algorithm: str,
        target_algorithm: str,
        source_value_steps: float,
        *,
        round_whole_step: bool = False,
    ) -> ConversionResult:
        key = (source_algorithm, target_algorithm)
        if source_algorithm == target_algorithm:
            raise ValueError("Source and target algorithms must differ.")
        if key not in self.metadata:
            raise KeyError(
                f"Unsupported direction {source_algorithm}->{target_algorithm}; "
                "A->B and B->A are separate fitted resources."
            )
        metadata = self.metadata[key]
        low = _float(metadata, "released_source_p05_steps")
        high = _float(metadata, "released_source_p95_steps")
        value = float(source_value_steps)
        converted: float | None = None
        if not math.isfinite(value):
            status = STATUS_NONFINITE
        elif value < low:
            status = STATUS_BELOW_RANGE
        elif value > high:
            status = STATUS_ABOVE_RANGE
        else:
            x, y = self.knots[key]
            converted_value = float(np.interp(value, x, y))
            converted = (
                float(round_half_up_nonnegative(converted_value))
                if round_whole_step
                else converted_value
            )
            status = STATUS_IN_RANGE
        return ConversionResult(
            source_algorithm=source_algorithm,
            target_algorithm=target_algorithm,
            source_value_steps=value,
            converted_target_steps=converted,
            status=status,
            released_source_low_steps=low,
            released_source_high_steps=high,
            resource_version=metadata["resource_version"],
        )

    def convert_many(
        self,
        source_algorithm: str,
        target_algorithm: str,
        source_values_steps: Iterable[float],
        *,
        round_whole_step: bool = False,
    ) -> list[ConversionResult]:
        return [
            self.convert_one(
                source_algorithm,
                target_algorithm,
                value,
                round_whole_step=round_whole_step,
            )
            for value in source_values_steps
        ]


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    default_tables = package_root / "results" / "reference" / "tables"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=default_tables / "crosswalk_direction_metadata.csv",
    )
    parser.add_argument(
        "--exact-knots",
        type=Path,
        default=default_tables / "crosswalk_exact_knots.csv",
    )
    parser.add_argument("--source", required=True, help="Source algorithm code, for example acti.")
    parser.add_argument("--target", required=True, help="Target algorithm code, for example oak.")
    parser.add_argument(
        "--value",
        required=True,
        type=float,
        action="append",
        help="Source-scale multi-day mean daily steps; repeat the option for multiple values.",
    )
    parser.add_argument("--round-whole-step", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resource = CrosswalkResource(args.metadata, args.exact_knots)
    try:
        results = resource.convert_many(
            args.source,
            args.target,
            args.value,
            round_whole_step=args.round_whole_step,
        )
    except (KeyError, ValueError) as error:
        print(f"CONVERSION_ERROR: {error}", file=sys.stderr)
        return 2
    fieldnames = list(ConversionResult.__dataclass_fields__)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result.__dict__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
